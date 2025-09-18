# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, Sequence, AsyncGenerator, Mapping
from vllm.logger import init_logger
from vllm.inputs import PromptType
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.outputs import RequestOutput

from arctic_inference.vllm.model_runner import ProblemIdContextManager

logger = init_logger(__name__)


class AsyncLLMPatch:
    _orig_generate = None
    _orig_validate_and_add_requests = None
    _orig_add_request = None


def apply_async_llm_patches():
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        
        if hasattr(AsyncLLM, '_arctic_async_problem_id_patched'):
            logger.debug("AsyncLLM already patched for problem_id support")
            return
        
        AsyncLLMPatch._orig_generate = AsyncLLM.generate
        # Store additional methods for compatibility (even if AsyncLLM doesn't have them)
        AsyncLLMPatch._orig_validate_and_add_requests = getattr(AsyncLLM, '_validate_and_add_requests', None)
        AsyncLLMPatch._orig_add_request = getattr(AsyncLLM, '_add_request', None)
        
        async def generate_patch(
            self,
            prompt: PromptType,
            sampling_params: SamplingParams,
            request_id: str,
            lora_request: Optional[LoRARequest] = None,
            trace_headers: Optional[Mapping[str, str]] = None,
            prompt_adapter_request: Optional[PromptAdapterRequest] = None,
            priority: int = 0,
            data_parallel_rank: Optional[int] = None,
            problem_ids: Optional[Union[str, Sequence[str]]] = None,
            **kwargs
        ) -> AsyncGenerator[RequestOutput, None]:
            
            if isinstance(problem_ids, str):
                problem_ids = [problem_ids]
            if isinstance(problem_ids, int):
                problem_ids = [problem_ids]
            hard_problems = getattr(self, '_arctic_hard_problems', None)
            max_quota = getattr(self, '_arctic_max_spec_quota', None)
            
            if problem_ids is not None:
                # TODO: Clear the requests
                if len(problem_ids) == 1:
                    current_problem_id = problem_ids[0]
                    try:
                        # Propagate to all workers via collective RPC (including this process)
                        await self.collective_rpc("atomic_update_req_id_mapping", 
                                                args=(request_id, current_problem_id))
                        
                        logger.debug(f"AsyncLLM: Mapped req_id {request_id} -> problem_id {current_problem_id}")
                    except Exception as e:
                        logger.warning(f"AsyncLLM: Failed to record req_id mapping for {request_id}: {e}")
            
            if hard_problems is not None or max_quota is not None:
                try:
                    await self.collective_rpc("set_dynamic_config", 
                                            args=(hard_problems, max_quota))
                except Exception as e:
                    logger.warning(f"AsyncLLM: Failed to set dynamic config: {e}")
            
            async for output in AsyncLLMPatch._orig_generate(
                self,
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                **kwargs
            ):
                yield output
        
        def set_hard_problems(self, hard_problems):
            self._arctic_hard_problems = set(hard_problems)
        
        def add_hard_problems(self, problem_ids):
            """Add problem IDs to the existing hard problems set incrementally."""
            if not hasattr(self, '_arctic_hard_problems') or self._arctic_hard_problems is None:
                self._arctic_hard_problems = set()
            
            if isinstance(problem_ids, (str, int)):
                problem_ids = [problem_ids]
            
            self._arctic_hard_problems.update(problem_ids)
            logger.debug(f"AsyncLLM: Added {len(problem_ids)} hard problems. Total: {len(self._arctic_hard_problems)}")
        
        def set_max_spec_quota(self, quota):
            self._arctic_max_spec_quota = quota
        
        async def clear_reqs(self):
            """Clear request ID mappings via collective RPC."""
            try:
                await self.collective_rpc("clear_problem_id_cache")
                logger.debug("AsyncLLM: Cleared request ID mappings via RPC")
            except Exception as e:
                logger.warning(f"AsyncLLM: Failed to clear cache via RPC: {e}")
        
        AsyncLLM.generate = generate_patch
        AsyncLLM.set_hard_problems = set_hard_problems
        AsyncLLM.add_hard_problems = add_hard_problems
        AsyncLLM.set_max_spec_quota = set_max_spec_quota
        AsyncLLM.clear_reqs = clear_reqs
        AsyncLLM._arctic_async_problem_id_patched = True
        
        logger.info("AsyncLLM patches applied successfully for problem_id support")
        
    except ImportError as e:
        logger.warning(f"Failed to apply AsyncLLM patches - vLLM V1 AsyncLLM not available: {e}")
    except Exception as e:
        logger.error(f"Error applying AsyncLLM patches: {e}")
        raise

def unapply_async_llm_patches():
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        
        if not hasattr(AsyncLLM, '_arctic_async_problem_id_patched'):
            return
        
        if AsyncLLMPatch._orig_generate is not None:
            AsyncLLM.generate = AsyncLLMPatch._orig_generate
            AsyncLLMPatch._orig_generate = None
            AsyncLLMPatch._orig_validate_and_add_requests = None  
            AsyncLLMPatch._orig_add_request = None
        
        delattr(AsyncLLM, '_arctic_async_problem_id_patched')
        logger.info("AsyncLLM patches removed successfully")
        
    except Exception as e:
        logger.error(f"Error removing AsyncLLM patches: {e}")