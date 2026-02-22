"""适配器：用 SuffixDecodingCache 实现 SuffixCache 的 API，以使用 rllm 的高效 C++ 实现"""
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Hashable, List, Optional, Sequence, Tuple, Union

from arctic_inference.suffix_decoding import SuffixDecodingCache, SuffixDecodingDraft
from arctic_inference.suffix_decoding._C import SuffixTree

# Do NOT import from .suffix_cache - it loads suffix_cache._C which conflicts with
# suffix_decoding._C (both register "Candidate" in pybind11). Use SuffixDecodingDraft
# as the result type; __init__.py aliases it as SuffixSpecResult when adapter is enabled.

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


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

    def _prebuild_problemtree(
        self,
        seq_id: int,
        problem_id: Hashable,
        prompt_token_ids: List[int],
        token_ids: List[int],
    ):
        """Build a single problem tree entry. Uses rllm's SuffixTree (C++ backend)."""
        if problem_id not in self._problem_tree:
            self._problem_tree[problem_id] = SuffixTree(self._max_tree_depth)
        tree = self._problem_tree[problem_id]
        tokens = token_ids if token_ids is not None else []
        tree.extend(seq_id, tokens)

    def prebuild_problems_parallel(
        self,
        problems_data: List[Tuple[Hashable, List[int], List[List[int]]]],
    ) -> dict:
        """
        Build multiple problem trees in parallel. Uses vsrivatsa's logic
        (hash-based load balancing, serial fallback) with rllm's C++ SuffixTree.

        Args:
            problems_data: List of (problem_id, prompt_tokens, sequences)
                in vsrivatsa format.

        Returns:
            Dictionary with performance statistics.
        """
        if not self._thread_safe:
            return self._build_problems_serial(problems_data)

        if not problems_data:
            return {"total_problems": 0, "total_time": 0.0, "method": "no_data"}

        start_time = time.perf_counter()

        # Hash-based load balancing (vsrivatsa style)
        thread_groups = [[] for _ in range(self._max_threads)]
        for problem_id, prompt_tokens, sequences in problems_data:
            problem_hash = hashlib.md5(str(problem_id).encode()).hexdigest()
            thread_id = int(problem_hash, 16) % self._max_threads
            thread_groups[thread_id].append((problem_id, prompt_tokens, sequences))

        def process_thread_group(group_data):
            processed = 0
            operations = 0
            thread_start = time.perf_counter()
            for problem_id, prompt_tokens, sequences in group_data:
                for i, token_ids in enumerate(sequences):
                    seq_id = -i - 1
                    self._prebuild_problemtree(seq_id, problem_id, prompt_tokens, token_ids)
                    operations += 2
                processed += 1
            return {
                "processed": processed,
                "operations": operations,
                "time": time.perf_counter() - thread_start,
            }

        workers = [g for g in thread_groups if g]
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = [
                executor.submit(process_thread_group, group)
                for group in thread_groups
                if group
            ]
            results = []
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc="Building with C++ object-level locking"):
                results.append(future.result())

        total_time = time.perf_counter() - start_time
        total_processed = sum(r["processed"] for r in results)
        total_operations = sum(r["operations"] for r in results)
        thread_times = [r["time"] for r in results]
        parallel_time = max(thread_times) if thread_times else 0.0
        sequential_equivalent_time = sum(thread_times)
        theoretical_speedup = sequential_equivalent_time / parallel_time if parallel_time > 0 else 1.0
        actual_speedup = sequential_equivalent_time / total_time if total_time > 0 else 1.0

        print("🚀 C++ Object-Level Locking Results (rllm backend):")
        print(f"  Total time: {total_time:.4f}秒")
        print(f"  Parallel time: {parallel_time:.4f}秒")
        print(f"  Processed problems: {total_processed}")
        print(f"  Total operations: {total_operations}")
        print(f"  Theoretical speedup: {theoretical_speedup:.2f}x")
        print(f"  Actual speedup: {actual_speedup:.2f}x")
        print(f"  Active threads: {len(results)}")

        return {
            "method": f"cpp_object_locking_{self._max_threads}",
            "total_problems": len(problems_data),
            "successful_problems": total_processed,
            "total_operations": total_operations,
            "total_time": total_time,
            "parallel_time": parallel_time,
            "theoretical_speedup": theoretical_speedup,
            "actual_speedup": actual_speedup,
            "active_threads": len(results),
            "thread_safe": True,
        }

    def _build_problems_serial(
        self,
        problems_data: List[Tuple[Hashable, List[int], List[List[int]]]],
    ) -> dict:
        """Fallback serial processing when thread_safe=False."""
        start_time = time.perf_counter()
        processed = 0
        operations = 0
        for problem_id, prompt_tokens, sequences in problems_data:
            for i, token_ids in enumerate(sequences):
                seq_id = -i - 1
                self._prebuild_problemtree(seq_id, problem_id, prompt_tokens, token_ids)
                operations += 2
            processed += 1
        total_time = time.perf_counter() - start_time
        return {
            "method": "serial_fallback",
            "total_problems": len(problems_data),
            "successful_problems": processed,
            "total_operations": operations,
            "total_time": total_time,
            "parallel_time": total_time,
            "theoretical_speedup": 1.0,
            "actual_speedup": 1.0,
            "active_threads": 1,
            "thread_safe": False,
        }