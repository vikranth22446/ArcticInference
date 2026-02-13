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
from typing import Optional

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
