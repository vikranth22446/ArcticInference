# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Suffix cache lifecycle management for speculative decoding.
#
# Handles building, prebuilding, activating, and clearing SuffixDecodingCache
# instances used by the vLLM model runner.  Also provides utilities for
# loading historical sequence data from step files and classifying problems
# into difficulty tiers (hard / medium / easy) for distribution-aware
# speculation (DAS).
#
# This module is loaded only in GPU worker processes (never in the trainer).

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from arctic_inference.suffix_decoding import SuffixDecodingCache
from arctic_inference.vllm.context_managers import ProblemIdContextManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone utilities (easily testable)
# ---------------------------------------------------------------------------

def load_problem_data_from_file_refs(
    payload: dict[str, Any],
    generation_id: int,
) -> list[dict]:
    """Build canonical ``problems_data`` from lightweight file references.

    Payload format::

        {
            "mode": "file_ref_v1",
            "token_ids_dir": "<...>/generation_token_ids",
            "steps_needed_by_pid": {"pid": [step1, step2, ...], ...}
        }

    Returns a list of dicts suitable for
    ``SuffixDecodingCache.prebuild_problems_parallel``.
    """
    mode = payload.get("mode")
    if mode != "file_ref_v1":
        return payload  # type: ignore[return-value]

    token_ids_dir = payload.get("token_ids_dir")
    steps_needed_by_pid = payload.get("steps_needed_by_pid", {})
    if not token_ids_dir or not isinstance(steps_needed_by_pid, dict):
        print(
            f"[SUFFIX_CACHE] Invalid file_ref payload for generation_id={generation_id}: "
            f"token_ids_dir={token_ids_dir}, steps_needed_by_pid_type={type(steps_needed_by_pid)}",
            flush=True,
        )
        return []

    if not os.path.isdir(token_ids_dir):
        print(
            f"[SUFFIX_CACHE] token_ids_dir does not exist for generation_id={generation_id}: {token_ids_dir}",
            flush=True,
        )
        return []

    normalized_steps_by_pid: dict[str, set[int]] = {}
    for pid, steps in steps_needed_by_pid.items():
        pid_str = str(pid)
        if not isinstance(steps, list):
            continue
        normalized_steps: set[int] = set()
        for s in steps:
            if not isinstance(s, (int, str)):
                continue
            try:
                normalized_steps.add(int(s))
            except (TypeError, ValueError):
                continue
        normalized_steps_by_pid[pid_str] = normalized_steps

    if not normalized_steps_by_pid:
        return []

    step_to_pids: dict[int, set[str]] = {}
    for pid, steps in normalized_steps_by_pid.items():
        for step in steps:
            step_to_pids.setdefault(step, set()).add(pid)

    all_steps = sorted(step_to_pids.keys())
    pid_to_sequences: dict[str, list[list[int]]] = {
        pid: [] for pid in normalized_steps_by_pid
    }

    io_start = time.perf_counter()
    for step in all_steps:
        file_path = os.path.join(token_ids_dir, f"{step}.jsonl")
        if not os.path.exists(file_path):
            continue
        required_pids = step_to_pids[step]
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_pid = str(entry.get("problem_id", ""))
                if entry_pid not in required_pids:
                    continue
                response_ids = entry.get("response_token_ids")
                if isinstance(response_ids, list) and response_ids:
                    pid_to_sequences[entry_pid].append(response_ids)

    io_end = time.perf_counter()
    problems_data_from_files: list[dict] = []
    total_seqs = 0
    for pid, seqs in pid_to_sequences.items():
        if seqs:
            problems_data_from_files.append({
                "problem_id": pid,
                "sequences": [
                    {"seq_id": -i - 1, "prompt_tokens": [], "response_tokens": s}
                    for i, s in enumerate(seqs)
                ],
            })
            total_seqs += len(seqs)
    print(
        f"[PREBUILD_TIMING] generation_id={generation_id} worker_file_ref_load_s="
        f"{(io_end - io_start):.3f} problems={len(problems_data_from_files)} sequences={total_seqs} "
        f"steps={len(all_steps)}",
        flush=True,
    )
    return problems_data_from_files


def classify_and_set_problem_difficulty(
    resolved_problem_data: list[dict],
    das_long_ratio: float,
    das_medium_ratio: float,
    generation_id: int,
) -> None:
    """Classify problems into hard/medium/easy by sequence-length statistics
    and set the result on ``ProblemIdContextManager``.

    For each problem, computes ``(mean_seq_len, max_seq_len)`` over its
    historical sequences.  Problems are sorted by ``(mean, max)`` in
    descending order, then the top *das_long_ratio* fraction is labelled
    **hard**, the next *das_medium_ratio* fraction **medium**, and the
    remainder **easy**.
    """
    pid_stats: dict[str, tuple[float, int]] = {}
    for entry in resolved_problem_data:
        pid = str(entry["problem_id"])
        lengths = [
            len(seq["response_tokens"])
            for seq in entry.get("sequences", [])
            if seq.get("response_tokens")
        ]
        if lengths:
            pid_stats[pid] = (float(np.mean(lengths)), max(lengths))

    if not pid_stats:
        return

    ranked_pids = sorted(
        pid_stats.keys(),
        key=lambda p: pid_stats[p],
        reverse=True,
    )

    n = len(ranked_pids)
    n_hard = max(1, int(n * das_long_ratio)) if n > 0 else 0
    n_medium = max(1, int(n * das_medium_ratio)) if n > 0 else 0
    hard_ids = ranked_pids[:n_hard]
    medium_ids = ranked_pids[n_hard:n_hard + n_medium]
    easy_ids = ranked_pids[n_hard + n_medium:]

    ProblemIdContextManager.set_hard_medium_ids(hard_ids, medium_ids, easy_ids)
    print(
        f"[DIFFICULTY] generation_id={generation_id} worker-local: "
        f"hard={len(hard_ids)}, medium={len(medium_ids)}, easy={len(easy_ids)} "
        f"(total={n}, das_long_ratio={das_long_ratio}, das_medium_ratio={das_medium_ratio})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Cache lifecycle manager
# ---------------------------------------------------------------------------

class SuffixCacheWorkerManager:
    """Manages the lifecycle of ``SuffixDecodingCache`` instances on a single
    vLLM worker.

    Supports synchronous rebuilds, asynchronous prebuilds via a background
    thread-pool, and activation / cleanup of generation-keyed caches.
    """

    def __init__(self, max_workers: int = 8):
        self._current_cache = None
        self._current_generation_id = -1
        self._prebuilt_caches: dict[int, SuffixDecodingCache] = {}
        self._stale_caches: list[SuffixDecodingCache] = []
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="SuffixCache",
        )
        self._lock = threading.Lock()
        self._build_futures: dict = {}

    # -- Accessors ----------------------------------------------------------

    def get_current_cache(self):
        with self._lock:
            return self._current_cache

    def get_current_generation_id(self):
        with self._lock:
            return self._current_generation_id

    # -- Synchronous rebuild ------------------------------------------------

    def rebuild_cache_sync(self, generation_id, cache_params, problems_data):
        if generation_id <= self._current_generation_id:
            return False

        new_cache = SuffixDecodingCache(**cache_params)
        if problems_data:
            new_cache.prebuild_problems_parallel(problems_data)

        with self._lock:
            old_cache = self._current_cache
            self._current_cache = new_cache
            self._current_generation_id = generation_id

            if old_cache:
                self._executor.submit(old_cache.clear_all_cache)
        return True

    # -- Asynchronous prebuild ----------------------------------------------

    def prebuild_cache_async(
        self,
        generation_id: int,
        cache_params: dict,
        problems_data,
        das_long_ratio: float = 0.2,
        das_medium_ratio: float = 0.4,
    ):
        if generation_id in self._build_futures:
            old_future = self._build_futures.pop(generation_id)
            old_future.cancel()
        enqueue_ts = time.perf_counter()

        def _build():
            build_start_ts = time.perf_counter()
            queue_wait_s = build_start_ts - enqueue_ts
            print(
                f"[PREBUILD_TIMING] generation_id={generation_id} worker_build_queue_wait_s={queue_wait_s:.3f}",
                flush=True,
            )
            try:
                cache = SuffixDecodingCache(**cache_params)
                resolved_problem_data = (
                    load_problem_data_from_file_refs(problems_data, generation_id)
                    if isinstance(problems_data, dict)
                    else problems_data
                )
                if resolved_problem_data:
                    classify_and_set_problem_difficulty(
                        resolved_problem_data, das_long_ratio, das_medium_ratio, generation_id,
                    )
                    cache.prebuild_problems_parallel(resolved_problem_data)

                with self._lock:
                    self._prebuilt_caches[generation_id] = cache
                build_end_ts = time.perf_counter()
                print(
                    f"[PREBUILD_TIMING] generation_id={generation_id} worker_build_total_s="
                    f"{(build_end_ts - build_start_ts):.3f}",
                    flush=True,
                )
                return cache
            except Exception as e:
                logger.error(
                    "Failed to prebuild cache for generation %s: %s",
                    generation_id, e,
                )
                return None
            finally:
                current_future = self._build_futures.get(generation_id)
                if current_future is future:
                    self._build_futures.pop(generation_id, None)

        future = self._executor.submit(_build)
        self._build_futures[generation_id] = future
        return future

    # -- Activation / cleanup -----------------------------------------------

    def activate_prebuilt_cache(self, generation_id):
        with self._lock:
            if generation_id in self._prebuilt_caches:
                old_cache = self._current_cache
                self._current_cache = self._prebuilt_caches.pop(generation_id)
                self._current_generation_id = generation_id
                if old_cache:
                    self._stale_caches.append(old_cache)
                return True
        return False

    def clear_old_suffix_cache(self, generation_id):
        """Clean up stale caches and any leftover prebuilt caches in background."""
        with self._lock:
            stale_prebuilt_ids = [
                gid for gid in self._prebuilt_caches if gid < generation_id
            ]
            caches_to_clear = [
                self._prebuilt_caches.pop(gid) for gid in stale_prebuilt_ids
            ]
            caches_to_clear.extend(self._stale_caches)
            self._stale_caches.clear()
        for cache in caches_to_clear:
            self._executor.submit(cache.clear_all_cache)

    def shutdown(self):
        for future in self._build_futures.values():
            future.cancel()
        self._executor.shutdown(wait=True)

        with self._lock:
            if self._current_cache:
                self._current_cache.clear_all_cache()
            for cache in self._prebuilt_caches.values():
                cache.clear_all_cache()
            self._prebuilt_caches.clear()
