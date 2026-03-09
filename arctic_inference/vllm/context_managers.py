# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight context managers for ArcticInference. This module has NO dependencies
# on vLLM GPU code (gpu_model_runner) or Triton, so it can be safely imported
# in the train_agent process before workers acquire GPU.
# The Triton-triggering imports happen in model_runner.py, which is only loaded
# in worker processes (via GPUModelRunnerPatch deferred to WorkerBase.__init__).

import contextlib
import threading
from typing import List, Optional

# Thread-local storage for per-batch / per-request data (batch problem_ids,
# req_id mapping).
_problem_id_context = threading.local()

# Process-level (cross-thread) storage for difficulty classification
# (hard/medium/easy IDs).  These are set from a background prebuild thread
# and read from the main worker thread during execute_model(), so they MUST
# NOT live in threading.local().
_difficulty_lock = threading.Lock()
_difficulty_data: dict = {
    'hard_ids': None,
    'medium_ids': None,
    'easy_ids': None,
    'version': 0,
}


class ProblemIdContextManager:
    """Context manager for problem_ids with req_id mapping support."""

    @staticmethod
    def set_current_batch_problem_ids(problem_ids: list[Optional[str]]):
        """Set problem_ids for the current batch."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        _problem_id_context.data['problem_ids'] = problem_ids

    @staticmethod
    def get_current_batch_problem_ids() -> list[Optional[str]]:
        """Get problem_ids for the current batch."""
        if not hasattr(_problem_id_context, 'data'):
            return []
        return _problem_id_context.data.get('problem_ids', [])

    @staticmethod
    def set_req_id_to_problem_id_mapping(mapping: dict[str, Optional[str]]):
        """Set req_id to problem_id mapping."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        _problem_id_context.data['req_id_to_problem_id'] = mapping

    @staticmethod
    def get_req_id_to_problem_id_mapping() -> dict[str, Optional[str]]:
        """Get the req_id to problem_id mapping."""
        if not hasattr(_problem_id_context, 'data'):
            return {}
        return _problem_id_context.data.get('req_id_to_problem_id', {})

    @staticmethod
    def get_problem_id_for_req_id(req_id: str) -> Optional[str]:
        """Get problem_id for a specific req_id."""
        if not hasattr(_problem_id_context, 'data'):
            return None

        mapping = _problem_id_context.data.get('req_id_to_problem_id', {})
        return mapping.get(req_id)

    @staticmethod
    def atomic_update_req_id_mapping(req_id: str, problem_id: Optional[str]):
        """Update req_id to problem_id mapping (thread-local, no lock needed)."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        current_mapping = _problem_id_context.data.get('req_id_to_problem_id', {})
        current_mapping[req_id] = problem_id
        _problem_id_context.data['req_id_to_problem_id'] = current_mapping

    @staticmethod
    def clear_context():
        """Clear the current context."""
        if hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}

    @staticmethod
    def clear_req_id_mapping():
        """Clear only the req_id mapping, keep problem_ids."""
        if hasattr(_problem_id_context, 'data'):
            _problem_id_context.data.pop('req_id_to_problem_id', None)

    @staticmethod
    def set_hard_medium_ids(
        hard_ids: Optional[List[str]] = None,
        medium_ids: Optional[List[str]] = None,
        easy_ids: Optional[List[str]] = None,
    ):
        """Set hard, medium, and easy problem IDs for distribution-aware processing.

        Uses process-level storage so the classification set from a background
        prebuild thread is visible to the main worker thread.
        Increments version so consumers can detect changes and rebuild caches.
        """
        with _difficulty_lock:
            _difficulty_data['hard_ids'] = hard_ids or []
            _difficulty_data['medium_ids'] = medium_ids or []
            _difficulty_data['easy_ids'] = easy_ids or []
            _difficulty_data['version'] += 1

    @staticmethod
    def get_hard_medium_ids() -> tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]]]:
        """Get hard, medium, and easy problem IDs (process-level, cross-thread safe)."""
        with _difficulty_lock:
            return (
                _difficulty_data.get('hard_ids'),
                _difficulty_data.get('medium_ids'),
                _difficulty_data.get('easy_ids'),
            )

    @staticmethod
    def get_difficulty_version() -> int:
        """Return the monotonically increasing version counter for difficulty data.

        Consumers can compare this with a cached version to decide whether to
        rebuild derived structures (e.g. difficulty sets).
        """
        with _difficulty_lock:
            return _difficulty_data['version']

    @staticmethod
    def set_dynamic_config(hard_problems=None, max_quota=None):
        """Set dynamic configuration for hard problems and quota."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        if hard_problems is not None:
            _problem_id_context.data['hard_problems'] = hard_problems
        if max_quota is not None:
            _problem_id_context.data['max_quota'] = max_quota

    @staticmethod
    def set_hard_problems(hard_problems):
        """Set hard problems only."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        _problem_id_context.data['hard_problems'] = hard_problems

    @staticmethod
    @contextlib.contextmanager
    def batch_context(problem_ids: list[Optional[str]]):
        """Context manager for a batch of problem_ids."""
        try:
            ProblemIdContextManager.set_current_batch_problem_ids(problem_ids)
            yield
        finally:
            ProblemIdContextManager.clear_context()
