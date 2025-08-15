import numpy as np
import pytest
from dataclasses import dataclass, field


class _SpecConfig:
    def __init__(self,
                 enable_suffix_decoding: bool = True,
                 suffix_cache_max_depth: int = 64,
                 suffix_max_spec_factor: float = 1.0,
                 suffix_max_spec_offset: float = 0.0,
                 suffix_min_token_prob: float = 0.1):
        self.enable_suffix_decoding = enable_suffix_decoding
        self.suffix_cache_max_depth = suffix_cache_max_depth
        self.suffix_max_spec_factor = suffix_max_spec_factor
        self.suffix_max_spec_offset = suffix_max_spec_offset
        self.suffix_min_token_prob = suffix_min_token_prob


class _InputBatch:
    def __init__(self, req_ids, token_ids_cpu, num_prompt_tokens, num_tokens_no_spec):
        self.req_ids = req_ids
        self.token_ids_cpu = token_ids_cpu
        self.num_prompt_tokens = num_prompt_tokens
        self.num_tokens_no_spec = num_tokens_no_spec


@dataclass
class SuffixSpecResult:
    token_ids: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    score: float = 0.0
    match_len: int = 0


class FakeCache:
    def __init__(self):
        self.calls: list[tuple] = []

    def cache_prompt(self, req_id, prompt):
        # No-op for fake cache
        pass

    def speculate(self, req_id,
                  pattern,
                  max_spec_tokens,
                  max_spec_factor,
                  max_spec_offset,
                  min_token_prob):
        # Deterministic fake: return last k tokens, with simple parent chain
        k = max(0, min(max_spec_tokens, len(pattern)))
        toks = pattern[-k:]
        parents = [-1] + list(range(0, max(0, k - 1))) if k > 0 else []
        probs = [min(1.0, 0.5 + 0.01 * (t % 50)) for t in toks]
        score = float(sum(probs))
        match_len = len(pattern) - k
        return SuffixSpecResult(token_ids=toks, parents=parents, probs=probs, score=score, match_len=match_len)


def _propose_serial(input_batch: _InputBatch,
                    suffix_cache: FakeCache,
                    spec_config: _SpecConfig,
                    sampled_token_ids: list[list[int]],
                    spec_token_ids: list[list[int]] | None,
                    max_model_len: int,
                    max_spec_len: int) -> list[SuffixSpecResult]:
    results: list[SuffixSpecResult] = []
    for i, sampled_ids in enumerate(sampled_token_ids):
        spec_ids = spec_token_ids[i] if spec_token_ids is not None else []
        if not sampled_ids:
            results.append(SuffixSpecResult())
            continue

        req_id = input_batch.req_ids[i]

        start_idx = input_batch.num_tokens_no_spec[i]
        end_idx = start_idx + len(sampled_ids)
        input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids

        size = min(end_idx, spec_config.suffix_cache_max_depth)
        pattern = input_batch.token_ids_cpu[i, end_idx - size:end_idx]
        pattern = pattern.tolist() + spec_ids
        if len(pattern) > spec_config.suffix_cache_max_depth:
            pattern = pattern[-spec_config.suffix_cache_max_depth:]

        max_spec_tokens = min(
            max_spec_len - len(spec_ids),
            spec_config.suffix_cache_max_depth,
            max_model_len - end_idx - 1,
        )
        max_spec_factor = spec_config.suffix_max_spec_factor
        max_spec_offset = (
            spec_config.suffix_max_spec_offset - len(spec_ids) * (max_spec_factor + 1)
        )

        result = suffix_cache.speculate(
            req_id,
            pattern,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            min_token_prob=spec_config.suffix_min_token_prob,
        )
        results.append(result)
    return results


def _propose_parallel(input_batch: _InputBatch,
                      suffix_cache: FakeCache,
                      spec_config: _SpecConfig,
                      sampled_token_ids: list[list[int]],
                      spec_token_ids: list[list[int]] | None,
                      max_model_len: int,
                      max_spec_len: int) -> list[SuffixSpecResult]:
    from concurrent.futures import ThreadPoolExecutor

    results: list[SuffixSpecResult] = [SuffixSpecResult() for _ in sampled_token_ids]
    tasks: list[tuple[int, object, list[int], int, float, float, float]] = []

    for i, sampled_ids in enumerate(sampled_token_ids):
        spec_ids = spec_token_ids[i] if spec_token_ids is not None else []
        if not sampled_ids:
            continue

        req_id = input_batch.req_ids[i]

        start_idx = input_batch.num_tokens_no_spec[i]
        end_idx = start_idx + len(sampled_ids)
        input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids

        size = min(end_idx, spec_config.suffix_cache_max_depth)
        base_seg = input_batch.token_ids_cpu[i, end_idx - size:end_idx]
        pattern = base_seg.tolist() + spec_ids
        if len(pattern) > spec_config.suffix_cache_max_depth:
            pattern = pattern[-spec_config.suffix_cache_max_depth:]

        max_spec_tokens = min(
            max_spec_len - len(spec_ids),
            spec_config.suffix_cache_max_depth,
            max_model_len - end_idx - 1,
        )
        max_spec_factor = spec_config.suffix_max_spec_factor
        max_spec_offset = (
            spec_config.suffix_max_spec_offset - len(spec_ids) * (max_spec_factor + 1)
        )
        tasks.append(
            (
                i,
                req_id,
                pattern,
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                spec_config.suffix_min_token_prob,
            )
        )

    def _spec(task):
        idx, req_id, pattern, max_spec_tokens, max_spec_factor, max_spec_offset, min_token_prob = task
        res = suffix_cache.speculate(
            req_id,
            pattern,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            min_token_prob=min_token_prob,
        )
        return idx, res

    if tasks:
        with ThreadPoolExecutor() as ex:
            for idx, res in ex.map(_spec, tasks):
                results[idx] = res

    return results


@pytest.mark.parametrize("batch_size", [1, 4,16,128,256])
def test_parallel_matches_serial(batch_size):
    max_model_len = 4096
    max_spec_len = 8
    spec = _SpecConfig(
        enable_suffix_decoding=True,
        suffix_cache_max_depth=16,
        suffix_max_spec_factor=1.0,
        suffix_max_spec_offset=0.0,
        suffix_min_token_prob=0.1,
    )

    # Create input batch buffers
    max_tokens_per_seq = 64
    token_ids_cpu = np.zeros((batch_size, max_tokens_per_seq), dtype=np.int32)
    num_prompt_tokens = np.zeros((batch_size,), dtype=np.int32)
    num_tokens_no_spec = np.zeros((batch_size,), dtype=np.int32)

    req_ids = [f"req_{i}" for i in range(batch_size)]
    inp = _InputBatch(req_ids, token_ids_cpu, num_prompt_tokens, num_tokens_no_spec)

    cache = FakeCache()

    # Prepare prompts and cache them
    for i in range(batch_size):
        prompt = [100 + i, 101 + i, 102 + i, 200, 201, 202, 200, 201, 202]
        n = len(prompt)
        inp.token_ids_cpu[i, :n] = prompt
        inp.num_prompt_tokens[i] = n
        inp.num_tokens_no_spec[i] = n
        cache.cache_prompt(inp.req_ids[i], prompt)

    # Prepare sampled token ids with some empty rows as edge cases
    sampled = []
    for i in range(batch_size):
        if i % 2 == 0:
            sampled.append([300 + i, 301 + i])
        else:
            sampled.append([])

    serial = _propose_serial(inp, cache, spec, sampled, None, max_model_len, max_spec_len)
    # Recreate fresh input buffers for the parallel run to avoid in-place side effects
    token_ids_cpu2 = np.copy(token_ids_cpu)
    inp2 = _InputBatch(list(req_ids), token_ids_cpu2, np.copy(num_prompt_tokens), np.copy(num_tokens_no_spec))
    parallel = _propose_parallel(inp2, cache, spec, sampled, None, max_model_len, max_spec_len)

    assert len(serial) == len(parallel)
    for a, b in zip(serial, parallel):
        assert a.token_ids == b.token_ids
        assert a.parents == b.parents
        assert a.probs == b.probs
        assert pytest.approx(a.score) == b.score
        assert a.match_len == b.match_len

