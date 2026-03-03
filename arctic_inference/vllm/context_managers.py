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

# Thread-local storage for problem_ids context and global lock for atomic operations
_problem_id_context = threading.local()
_problem_id_context_lock = threading.Lock()


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
        """Atomically update req_id to problem_id mapping."""
        with _problem_id_context_lock:
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
        """Set hard, medium, and easy problem IDs for distribution-aware processing."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        _problem_id_context.data['hard_ids'] = hard_ids or []
        _problem_id_context.data['medium_ids'] = medium_ids or []
        _problem_id_context.data['easy_ids'] = easy_ids or []
        # Invalidate cached indices so they are recomputed for the next batch
        _problem_id_context.data.pop('hard_indices', None)
        _problem_id_context.data.pop('medium_indices', None)
        _problem_id_context.data.pop('easy_indices', None)
        _problem_id_context.data.pop('allowed_indices', None)

    @staticmethod
    def get_hard_medium_ids() -> tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]]]:
        """Get hard, medium, and easy problem IDs."""
        if not hasattr(_problem_id_context, 'data'):
            return None, None, None
        return (
            _problem_id_context.data.get('hard_ids'),
            _problem_id_context.data.get('medium_ids'),
            _problem_id_context.data.get('easy_ids'),
        )

    @staticmethod
    def set_hard_medium_indices(
        hard_indices: List[int],
        medium_indices: List[int],
        easy_indices: List[int],
        allowed_indices: List[int],
    ):
        """Cache hard, medium, and easy indices for the current batch."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        _problem_id_context.data['hard_indices'] = hard_indices
        _problem_id_context.data['medium_indices'] = medium_indices
        _problem_id_context.data['easy_indices'] = easy_indices
        _problem_id_context.data['allowed_indices'] = allowed_indices

    @staticmethod
    def get_hard_medium_indices() -> tuple[List[int], List[int], List[int], List[int]]:
        """Get cached hard, medium, and easy indices."""
        if not hasattr(_problem_id_context, 'data'):
            return [], [], [], []
        return (
            _problem_id_context.data.get('hard_indices', []),
            _problem_id_context.data.get('medium_indices', []),
            _problem_id_context.data.get('easy_indices', []),
            _problem_id_context.data.get('allowed_indices', []),
        )

    @staticmethod
    def has_hard_medium_indices() -> bool:
        """Check if hard/medium indices are cached."""
        if not hasattr(_problem_id_context, 'data'):
            return False
        return 'hard_indices' in _problem_id_context.data

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
    def get_dynamic_hard_problems():
        """Get dynamic hard problems configuration."""
        if not hasattr(_problem_id_context, 'data'):
            return None
        return _problem_id_context.data.get('hard_problems')

    @staticmethod
    def get_dynamic_max_quota():
        """Get dynamic max quota configuration."""
        if not hasattr(_problem_id_context, 'data'):
            return None
        return _problem_id_context.data.get('max_quota')

    @staticmethod
    @contextlib.contextmanager
    def batch_context(problem_ids: list[Optional[str]]):
        """Context manager for a batch of problem_ids."""
        try:
            ProblemIdContextManager.set_current_batch_problem_ids(problem_ids)
            yield
        finally:
            ProblemIdContextManager.clear_context()
