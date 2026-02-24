"""适配器：用 SuffixDecodingCache 实现 SuffixCache 的 API，以使用 rllm 的高效 C++ 实现"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Hashable, List, Optional, Sequence, Tuple, Union
import time
from arctic_inference.suffix_decoding import SuffixDecodingCache, SuffixDecodingDraft
from arctic_inference.suffix_decoding._C import SuffixTree

# Do NOT import from .suffix_cache - it loads suffix_cache._C which conflicts with
# suffix_decoding._C (both register "Candidate" in pybind11). Use SuffixDecodingDraft
# as the result type; __init__.py aliases it as SuffixSpecResult when adapter is enabled.


class SuffixCacheAdapter(SuffixDecodingCache):
    """将 SuffixDecodingCache 适配为 SuffixCache 接口"""
    
    def __init__(self, max_depth=64, thread_safe=False, max_threads=None, **kwargs):
        super().__init__(max_tree_depth=max_depth, thread_safe=thread_safe, 
                         max_threads=max_threads or 8, max_cached_requests=-1, **kwargs)
    
    @property
    def max_depth(self):
        return self.max_tree_depth
    
    def has_cached_prompt(self, req_id):
        return req_id in self._local_trees

    def cache_prompt(
        self,
        req_id: Hashable,
        prompt_token_ids: Sequence[int],
        problem_id: Optional[Hashable] = None,
    ):
        """
        Cache a prompt for a request. When problem_id is provided (recommended),
        uses it for start_request. Otherwise uses req_id as placeholder; the
        problem tree will be correctly updated when update_response is called.
        """
        effective_pid = problem_id if problem_id is not None else req_id
        self.start_request(req_id, problem_id=effective_pid, prompt_token_ids=prompt_token_ids)

    def evict_prompt(self, req_id):
        self.stop_request(req_id)

    def update_response(
        self,
        req_id: Hashable,
        problem_id: Hashable,
        token_ids: Union[int, Sequence[int]],
    ):
        """Update local_trees only."""
        super().add_active_response(req_id, problem_id, token_ids)
    
    def speculate(self, req_id, problem_id, pattern, max_spec_tokens=None,
                  max_spec_factor=1.0, max_spec_offset=-1, min_token_prob=0.1, **kwargs):
        """Returns SuffixDecodingDraft (aliased as SuffixSpecResult in __init__)."""
        result, _ = super().speculate(req_id, pattern, max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor, max_spec_offset=max_spec_offset,
            min_token_prob=min_token_prob, problem_id=problem_id, **kwargs)
        return result

    def clear_all_cache(self):
        """Clear all cached data. Compatible with SuffixCache interface."""
        self.clear_cache(problem_ids=None)

    def _normalize_prebuild_input(
        self,
        data: List[Union[dict, Tuple[Hashable, List[int], List[List[int]]]]],
    ) -> List[dict]:
        """
        Normalize input to the canonical format. Accepts both:
        - New format: [{'problem_id': pid, 'sequences': [{'seq_id', 'prompt_tokens', 'response_tokens'}, ...]}, ...]
        - Old format: [(problem_id, prompt_tokens, sequences), ...] where sequences is List[List[int]]
        """
        if not data:
            return []
        first = data[0]
        if isinstance(first, dict) and "problem_id" in first:
            return list(data)
        # Old format: (problem_id, prompt_tokens, sequences)
        normalized = []
        for problem_id, prompt_tokens, sequences in data:
            normalized.append({
                "problem_id": problem_id,
                "sequences": [
                    {
                        "seq_id": -i - 1,
                        "prompt_tokens": list(prompt_tokens) if prompt_tokens else [],
                        "response_tokens": list(token_ids) if token_ids else [],
                    }
                    for i, token_ids in enumerate(sequences)
                ],
            })
        return normalized

    def prebuild_problems_parallel(
        self,
        problem_data: List[Union[dict, Tuple[Hashable, List[int], List[List[int]]]]],
    ) -> dict:
        """
        Pre-build multiple problem trees in parallel using ThreadPoolExecutor.
        Logic matches arctic_inference/suffix_decoding/cache.py.

        Args:
            problem_data: Either format:
                - New: [{'problem_id': pid, 'sequences': [{'seq_id', 'prompt_tokens', 'response_tokens'}, ...]}, ...]
                - Old: [(problem_id, prompt_tokens, sequences), ...] where sequences is List[List[int]]

        Returns:
            dict: Results containing success status and statistics
        """
        start_time = time.time()
        if not problem_data:
            end_time = time.time()
            print(f"Time taken to prebuild problems: {end_time - start_time} seconds")
            return {"success": True, "problems_built": 0, "total_problems": 0}

        problem_data = self._normalize_prebuild_input(problem_data)

        # Validate and normalize input data with assertions
        # Thread grouping and load balancing (round-robin per problem)
        thread_groups = [[] for _ in range(self._max_threads)]

        for i, item in enumerate(problem_data):
            assert isinstance(item, dict), f"Expected dict at index {i}, got {type(item)}"
            assert "problem_id" in item, f"Missing 'problem_id' key at index {i}"

            problem_id = item["problem_id"]
            sequences = item.get("sequences", [])
            assert isinstance(sequences, list), (
                f"'sequences' must be list at index {i}, got {type(sequences)}"
            )

            thread_idx = i % self._max_threads

            for seq_idx, seq_data in enumerate(sequences):
                assert isinstance(seq_data, dict), (
                    f"Sequence at index {i}.{seq_idx} must be dict, got {type(seq_data)}"
                )

                seq_id = seq_data.get("seq_id", seq_idx)
                prompt_tokens = seq_data.get("prompt_tokens", [])
                response_tokens = seq_data.get("response_tokens", [])

                assert isinstance(seq_id, int), (
                    f"seq_id must be int at {i}.{seq_idx}, got {type(seq_id)}"
                )
                assert isinstance(prompt_tokens, list), (
                    f"prompt_tokens must be list at {i}.{seq_idx}, got {type(prompt_tokens)}"
                )
                assert isinstance(response_tokens, list), (
                    f"response_tokens must be list at {i}.{seq_idx}, got {type(response_tokens)}"
                )

                thread_groups[thread_idx].append(
                    (seq_id, problem_id, prompt_tokens, response_tokens)
                )

        def prebuild_problem_group(group_data):
            """Pre-build a group of problems in one thread"""
            built_count = 0

            for seq_id, problem_id, prompt_tokens, response_tokens in group_data:
                try:
                    if problem_id not in self._problem_tree:
                        self._problem_tree[problem_id] = SuffixTree(self._max_tree_depth)

                    tree = self._problem_tree[problem_id]

                    if prompt_tokens or response_tokens:
                        if self._thread_safe:
                            tree.extend_safe(seq_id, prompt_tokens + response_tokens)
                        else:
                            tree.extend(seq_id, prompt_tokens + response_tokens)

                    built_count += 1

                except Exception as e:
                    print(f"Error building problem {problem_id}: {e}")
                    continue

            return {"built": built_count}

        max_workers = len([g for g in thread_groups if g])
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for group in thread_groups:
                if group:
                    future = executor.submit(prebuild_problem_group, group)
                    futures.append(future)

            results = []
            for future in as_completed(futures):
                results.append(future.result())

        total_built = sum(r["built"] for r in results)

        end_time = time.time()
        print(f"[SUFFIX_CACHE_PREBUILD] Time taken to prebuild problems: {end_time - start_time} seconds")
        return {
            "success": True,
            "problems_built": total_built,
            "total_problems": len(problem_data),
        }