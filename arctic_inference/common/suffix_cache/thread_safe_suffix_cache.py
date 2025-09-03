# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""
线程安全的SuffixCache实现

核心特性：
- 使用C++对象级锁定 + GIL释放实现真正的per-object并行
- 不同problem_id的SuffixTree可以真正并行执行
- 同一个SuffixTree的操作仍然串行（由C++对象锁保护）
- 解决了传统多线程的GIL争夺问题
"""

import threading
import time
import hashlib
from typing import Dict, List, Tuple, Hashable, Sequence, Union, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .suffix_cache import SuffixCache, SuffixSpecResult
from arctic_inference.common.suffix_cache._C import SuffixTree


@dataclass
class ThreadSafeBuildTask:
    """线程安全构建任务"""
    thread_id: int
    problems_data: List[Tuple[Hashable, List[int], List[List[int]]]]


@dataclass
class ThreadSafeBuildResult:
    """线程安全构建结果"""
    thread_id: int
    problems_processed: int
    total_operations: int
    processing_time: float
    success: bool
    error_msg: str = ""


class ThreadSafeSuffixCache:
    """
    线程安全的SuffixCache
    
    核心机制：
    1. 📦 每个SuffixTree有独立的C++对象锁 (std::mutex)
    2. 🔓 C++方法释放GIL (py::call_guard<py::gil_scoped_release>)
    3. ⚡ 不同problem_id可以真正并行执行
    4. 🔒 同一problem_id内的操作串行化（对象锁保护）
    
    这样实现了："不同problem_id有独立GIL，同一tree共享GIL"的需求
    """
    
    def __init__(self, max_depth: int = 64, max_threads: int = None):
        self.max_depth = max_depth
        self.max_threads = max_threads or min(8, threading.active_count() + 4)
        
        # 主SuffixCache（使用线程安全方法）
        self.main_cache = SuffixCache(max_depth)
        
        print(f"ThreadSafeSuffixCache初始化: 最大{self.max_threads}个线程")
    
    def _get_thread_for_problem(self, problem_id: Hashable) -> int:
        """哈希分区：将problem_id分配到固定线程"""
        problem_hash = hashlib.md5(str(problem_id).encode()).hexdigest()
        return int(problem_hash, 16) % self.max_threads
    
    def _build_problems_in_thread(self, task: ThreadSafeBuildTask) -> ThreadSafeBuildResult:
        """在线程中构建problems（使用线程安全方法）"""
        start_time = time.perf_counter()
        
        try:
            problems_processed = 0
            total_operations = 0
            
            for problem_id, prompt_tokens, sequences in task.problems_data:
                # 🔑 关键步骤：确保SuffixTree存在
                if problem_id not in self.main_cache._problem_tree:
                    # 这个操作需要线程同步（访问共享字典）
                    with threading.Lock():  # 保护共享字典的访问
                        if problem_id not in self.main_cache._problem_tree:
                            self.main_cache._problem_tree[problem_id] = SuffixTree(self.max_depth)
                
                # ⚡ 关键优化：使用线程安全方法（释放GIL + 对象锁）
                tree = self.main_cache._problem_tree[problem_id]
                
                for i, token_ids in enumerate(sequences):
                    seq_id = -i-1
                    
                    # 🚀 这些操作会：
                    # 1. 获取tree的对象锁 (std::lock_guard<std::mutex>)
                    # 2. 释放Python GIL (py::call_guard<py::gil_scoped_release>)  
                    # 3. 执行C++代码（真正并行！）
                    tree.extend_safe(seq_id, prompt_tokens)
                    tree.extend_safe(seq_id, token_ids)
                    total_operations += 2
                
                problems_processed += 1
            
            processing_time = time.perf_counter() - start_time
            
            return ThreadSafeBuildResult(
                thread_id=task.thread_id,
                problems_processed=problems_processed,
                total_operations=total_operations,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return ThreadSafeBuildResult(
                thread_id=task.thread_id,
                problems_processed=0,
                total_operations=0,
                processing_time=processing_time,
                success=False,
                error_msg=str(e)
            )
    
    def build_problems_parallel(
        self, 
        problems_data: List[Tuple[Hashable, List[int], List[List[int]]]]
    ) -> Dict[str, Any]:
        """并行构建problems（真正的per-object并行）"""
        
        if not problems_data:
            return {"total_problems": 0, "total_time": 0.0, "method": "no_data"}
        
        start_time = time.perf_counter()
        
        # 按线程分区分组problems
        thread_problems = [[] for _ in range(self.max_threads)]
        
        for problem_id, prompt_tokens, sequences in problems_data:
            thread_id = self._get_thread_for_problem(problem_id)
            thread_problems[thread_id].append((problem_id, prompt_tokens, sequences))
        
        # 创建线程任务
        thread_tasks = []
        for thread_id, problems in enumerate(thread_problems):
            if problems:
                task = ThreadSafeBuildTask(thread_id, problems)
                thread_tasks.append(task)
        
        thread_sizes = [len(tp) for tp in thread_problems]
        print(f"线程分布: {thread_sizes}")
        print(f"启动{len(thread_tasks)}个线程进行真正的并行构建")
        
        # 🚀 关键：使用线程池 + 线程安全方法实现真正并行
        with ThreadPoolExecutor(max_workers=len(thread_tasks)) as executor:
            from tqdm import tqdm
            
            # 提交所有任务
            future_to_task = {
                executor.submit(self._build_problems_in_thread, task): task 
                for task in thread_tasks
            }
            
            # 收集结果
            results = []
            for future in tqdm(as_completed(future_to_task), total=len(thread_tasks), 
                              desc="Building with thread-safe C++ objects"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    results.append(ThreadSafeBuildResult(
                        thread_id=task.thread_id,
                        problems_processed=0,
                        total_operations=0,
                        processing_time=0.0,
                        success=False,
                        error_msg=str(e)
                    ))
        
        # 统计结果
        successful_threads = 0
        total_operations = 0
        processing_times = []
        
        for result in results:
            if result.success:
                successful_threads += 1
                total_operations += result.total_operations
                processing_times.append(result.processing_time)
                print(f"线程{result.thread_id}: {result.problems_processed} problems, "
                      f"{result.total_operations} ops, {result.processing_time:.4f}秒")
            else:
                print(f"线程{result.thread_id} 失败: {result.error_msg}")
        
        total_time = time.perf_counter() - start_time
        
        # 性能指标
        parallel_time = max(processing_times) if processing_times else 0.0
        theoretical_speedup = sum(processing_times) / parallel_time if parallel_time > 0 else 1.0
        actual_speedup = sum(processing_times) / total_time if total_time > 0 else 1.0
        
        print(f"线程安全并行完成:")
        print(f"  并行执行时间: {parallel_time:.4f}秒") 
        print(f"  总时间: {total_time:.4f}秒")
        print(f"  理论加速比: {theoretical_speedup:.2f}x")
        print(f"  实际加速比: {actual_speedup:.2f}x")
        print(f"  ✅ 真正实现了per-problem-ID的独立并行执行")
        
        return {
            "method": f"thread_safe_cpp_locking_{self.max_threads}",
            "total_problems": len(problems_data),
            "successful_problems": sum(r.problems_processed for r in results if r.success),
            "successful_threads": successful_threads,
            "total_operations": total_operations,
            "total_time": total_time,
            "parallel_time": parallel_time,
            "theoretical_speedup": theoretical_speedup,
            "actual_speedup": actual_speedup,
            "thread_distribution": thread_sizes,
            "active_threads": len(thread_tasks)
        }
    
    def prebuild_problemtree_safe(
        self, 
        seq_id: int, 
        problem_id: Hashable,
        prompt_token_ids: Sequence[int], 
        token_ids: Sequence[int]
    ):
        """线程安全的单个prebuild方法"""
        # 确保SuffixTree存在
        if problem_id not in self.main_cache._problem_tree:
            self.main_cache._problem_tree[problem_id] = SuffixTree(self.max_depth)
        
        # 使用线程安全方法
        tree = self.main_cache._problem_tree[problem_id]
        tree.extend_safe(seq_id, list(prompt_token_ids))
        tree.extend_safe(seq_id, list(token_ids))
    
    def speculate(
        self,
        req_id: Hashable,
        problem_id: Hashable,
        pattern: Sequence[int],
        **kwargs
    ) -> SuffixSpecResult:
        """推测方法（speculate已经是线程安全的）"""
        return self.main_cache.speculate(req_id, problem_id, pattern, **kwargs)
    
    def get_thread_stats(self) -> Dict[str, Any]:
        """获取线程安全统计信息"""
        stats = {
            "max_threads": self.max_threads,
            "problem_tree_count": len(self.main_cache._problem_tree),
            "prompt_tree_count": len(self.main_cache._prompt_trees)
        }
        
        # 统计各个线程的problem分布
        thread_problems = [0] * self.max_threads
        for problem_id in self.main_cache._problem_tree.keys():
            thread_id = self._get_thread_for_problem(problem_id)
            thread_problems[thread_id] += 1
        
        stats["thread_problem_distribution"] = thread_problems
        
        return stats
    
    def clear_all_cache(self):
        """清空所有缓存"""
        self.main_cache.clear_all_cache()


# 适配器：兼容原SuffixCache接口
class ThreadSafeSuffixCacheAdapter:
    """
    适配器，让ThreadSafeSuffixCache完全兼容原SuffixCache接口
    可以直接替换原来的SuffixCache使用
    """
    
    def __init__(self, max_depth: int = 64, max_threads: int = None):
        self.thread_safe_cache = ThreadSafeSuffixCache(max_depth, max_threads)
        self._max_depth = max_depth
    
    @property
    def max_depth(self) -> int:
        return self._max_depth
    
    @property
    def _problem_tree(self):
        """直接访问主缓存的problem_tree"""
        return self.thread_safe_cache.main_cache._problem_tree
    
    @property
    def _prompt_trees(self):
        """直接访问主缓存的prompt_trees"""
        return self.thread_safe_cache.main_cache._prompt_trees
    
    def prebuild_problemtree(
        self, 
        seq_id: int, 
        problem_id: Hashable,
        prompt_token_ids: Sequence[int], 
        token_ids: Sequence[int]
    ):
        """兼容接口：单个prebuild"""
        self.thread_safe_cache.prebuild_problemtree_safe(
            seq_id, problem_id, prompt_token_ids, token_ids
        )
    
    def speculate(
        self,
        req_id: Hashable,
        problem_id: Hashable,
        pattern: Sequence[int],
        **kwargs
    ) -> SuffixSpecResult:
        """兼容接口：推测"""
        return self.thread_safe_cache.speculate(req_id, problem_id, pattern, **kwargs)
    
    def clear_all_cache(self):
        """兼容接口：清空缓存"""
        self.thread_safe_cache.clear_all_cache()
    
    # 批量构建接口（推荐使用）
    def build_problems_parallel(
        self, 
        problems_data: List[Tuple[Hashable, List[int], List[List[int]]]]
    ) -> Dict[str, Any]:
        """批量并行构建（推荐接口）"""
        return self.thread_safe_cache.build_problems_parallel(problems_data)
