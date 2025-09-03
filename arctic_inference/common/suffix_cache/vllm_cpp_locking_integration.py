# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM C++对象级锁定集成

一行替换解决方案：使用真正的per-problem-ID并行
"""

import time
from typing import List, Dict, Tuple, Any, Hashable

from .thread_safe_suffix_cache import ThreadSafeSuffixCache


def replace_vllm_suffix_prebuild_with_cpp_locking(
    vllm_rollout_instance,
    unique_problem_ids: List[Hashable],
    problem_id_to_sequences: Dict[Hashable, List[List[int]]],
    vllm_inputs: List[Dict],
    max_threads: int = None
) -> Dict[str, Any]:
    """
    使用C++对象级锁定替换vLLM中的prebuild代码
    
    在vllm_rollout_spmd.py中替换第625-735行：
    
    # 原代码（有问题的多进程版本）:
    # suffix_cache = self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache
    # ... 复杂的、有序列化问题的多进程代码 ...
    
    # 替换为：
    from arctic_inference.common.suffix_cache.vllm_cpp_locking_integration import replace_vllm_suffix_prebuild_with_cpp_locking
    result = replace_vllm_suffix_prebuild_with_cpp_locking(
        self, unique_problem_ids, problem_id_to_sequences, vllm_inputs
    )
    print(f"✅ C++对象级锁定完成: {result}")
    """
    
    if not unique_problem_ids or not problem_id_to_sequences:
        return {"total_problems": 0, "total_time": 0.0, "method": "cpp_locking_no_data"}
    
    # 获取原始SuffixCache
    original_suffix_cache = vllm_rollout_instance.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache
    max_depth = original_suffix_cache._max_depth
    
    # 准备数据
    problems_data = []
    for problem_id in unique_problem_ids:
        if problem_id not in problem_id_to_sequences:
            continue
            
        prompt_tokens = vllm_rollout_instance.get_prompt_token_ids(vllm_inputs, problem_id)
        if prompt_tokens is None:
            continue
            
        sequences = problem_id_to_sequences[problem_id]
        if sequences:
            problems_data.append((problem_id, list(prompt_tokens), sequences))
    
    if not problems_data:
        return {"method": "cpp_locking", "total_problems": 0, "total_time": 0.0}
    
    print(f"🚀 使用C++对象级锁定处理{len(problems_data)}个problems")
    
    # 使用线程安全SuffixCache
    thread_safe_cache = ThreadSafeSuffixCache(
        max_depth=max_depth, 
        max_threads=max_threads or 8
    )
    
    # 执行并行构建
    result = thread_safe_cache.build_problems_parallel(problems_data)
    
    # 将结果合并到原始SuffixCache
    print("合并结果到原始SuffixCache...")
    merge_start = time.perf_counter()
    
    for problem_id in thread_safe_cache.main_cache._problem_tree:
        original_suffix_cache._problem_tree[problem_id] = thread_safe_cache.main_cache._problem_tree[problem_id]
    
    merge_time = time.perf_counter() - merge_start
    
    # 更新结果统计
    result.update({
        "merge_time": merge_time,
        "original_cache_trees": len(original_suffix_cache._problem_tree)
    })
    
    print(f"✅ C++对象级锁定完成:")
    print(f"  方法: 真正的per-problem-ID并行")
    print(f"  处理问题数: {result.get('successful_problems', 0)}")
    print(f"  并行时间: {result.get('parallel_time', 0.0):.4f}秒")
    print(f"  总时间: {result.get('total_time', 0.0):.4f}秒")
    print(f"  加速比: {result.get('actual_speedup', 1.0):.2f}x")
    print(f"  🎯 解决了GIL争夺和序列化问题")
    
    return result


def create_thread_safe_suffix_cache_replacement(
    original_suffix_cache,
    max_threads: int = None
):
    """
    创建原SuffixCache的线程安全替代版本
    
    使用方法：
    # 在vLLM初始化时替换SuffixCache
    original_cache = self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache
    thread_safe_cache = create_thread_safe_suffix_cache_replacement(original_cache)
    self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache = thread_safe_cache
    """
    from .thread_safe_suffix_cache import ThreadSafeSuffixCacheAdapter
    
    return ThreadSafeSuffixCacheAdapter(
        max_depth=original_suffix_cache.max_depth,
        max_threads=max_threads
    )
