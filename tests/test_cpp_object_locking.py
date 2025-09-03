#!/usr/bin/env python3
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""
测试C++对象级锁定方案

验证：
1. 线程安全方法是否存在
2. GIL是否真正被释放
3. 不同problem_id是否能真正并行
4. 同一problem_id是否保持串行
5. 性能提升是否显著
"""

import sys
import os
import time
import random
import threading
import multiprocessing
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

# 添加项目路径
sys.path.append('/data/zshao/rllm/ArcticInference')

def test_thread_safe_methods_exist():
    """测试线程安全方法是否存在"""
    print("🔍 测试1: 检查线程安全方法是否存在")
    
    try:
        from arctic_inference.common.suffix_cache._C import SuffixTree
        
        # 创建测试树
        tree = SuffixTree(64)
        
        # 检查线程安全方法
        safe_methods = ['append_safe', 'extend_safe', 'num_seqs_safe']
        missing_methods = []
        
        for method in safe_methods:
            if hasattr(tree, method):
                print(f"  ✅ {method} 方法存在")
            else:
                missing_methods.append(method)
                print(f"  ❌ {method} 方法缺失")
        
        if missing_methods:
            print(f"  ⚠️ 缺失方法: {missing_methods}")
            print(f"  请确保已重新编译C++代码：pip install -e .[vllm] -v")
            return False
        else:
            print(f"  ✅ 所有线程安全方法都存在")
            return True
            
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_gil_release_effectiveness():
    """测试GIL释放的有效性"""
    print(f"\n⚡ 测试2: 验证GIL释放效果")
    
    try:
        from arctic_inference.common.suffix_cache._C import SuffixTree
        
        def cpu_intensive_task_safe(tree, seq_id_start, num_ops=1000):
            """使用线程安全方法的CPU密集型任务"""
            for i in range(num_ops):
                seq_id = seq_id_start + i
                tokens = [random.randint(1, 1000) for _ in range(50)]
                tree.extend_safe(seq_id, tokens)
            return num_ops
        
        def cpu_intensive_task_unsafe(tree, seq_id_start, num_ops=1000):
            """使用非线程安全方法的CPU密集型任务"""
            for i in range(num_ops):
                seq_id = seq_id_start + i
                tokens = [random.randint(1, 1000) for _ in range(50)]
                tree.extend(seq_id, tokens)
            return num_ops
        
        # 创建多个独立的树（避免对象锁冲突）
        trees_safe = [SuffixTree(64) for _ in range(4)]
        trees_unsafe = [SuffixTree(64) for _ in range(4)]
        
        # 测试线程安全版本
        print("  测试线程安全版本（应该有并行效果）...")
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(cpu_intensive_task_safe, trees_safe[i], i * 1000, 500) 
                for i in range(4)
            ]
            results_safe = [f.result() for f in futures]
        
        safe_time = time.perf_counter() - start_time
        
        # 测试非线程安全版本
        print("  测试非线程安全版本（应该串行化）...")
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(cpu_intensive_task_unsafe, trees_unsafe[i], i * 1000, 500)
                for i in range(4)
            ]
            results_unsafe = [f.result() for f in futures]
        
        unsafe_time = time.perf_counter() - start_time
        
        # 计算加速比
        speedup = unsafe_time / safe_time if safe_time > 0 else 1.0
        
        print(f"  线程安全版本时间: {safe_time:.4f}秒")
        print(f"  非线程安全版本时间: {unsafe_time:.4f}秒") 
        print(f"  加速比: {speedup:.2f}x")
        
        if speedup > 1.5:
            print(f"  ✅ GIL释放生效！线程安全版本有明显加速")
            return True
        elif speedup > 1.1:
            print(f"  ⚠️ GIL释放部分生效，有一定加速")
            return True
        else:
            print(f"  ❌ GIL释放可能未生效，无明显加速")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_per_object_locking():
    """测试per-object锁定机制"""
    print(f"\n🔒 测试3: 验证per-object锁定机制")
    
    try:
        from arctic_inference.common.suffix_cache._C import SuffixTree
        
        shared_tree = SuffixTree(64)
        results = []
        lock = threading.Lock()
        
        def worker_on_same_tree(worker_id, num_ops=500):
            """多个线程操作同一个树（应该串行化）"""
            ops_completed = 0
            for i in range(num_ops):
                seq_id = worker_id * 1000 + i
                tokens = [worker_id, i] * 10  # 可区分的token序列
                shared_tree.extend_safe(seq_id, tokens)
                ops_completed += 1
            
            with lock:
                results.append((worker_id, ops_completed))
            return ops_completed
        
        # 多个线程操作同一个树
        print("  多个线程操作同一个SuffixTree...")
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(worker_on_same_tree, worker_id, 300)
                for worker_id in range(4)
            ]
            thread_results = [f.result() for f in futures]
        
        same_tree_time = time.perf_counter() - start_time
        
        # 验证结果正确性
        total_seqs = shared_tree.num_seqs_safe()
        expected_seqs = sum(thread_results)
        
        print(f"  执行时间: {same_tree_time:.4f}秒")
        print(f"  期望序列数: {expected_seqs}")
        print(f"  实际序列数: {total_seqs}")
        
        if total_seqs == expected_seqs:
            print(f"  ✅ Per-object锁定工作正常，数据一致性保持")
            return True
        else:
            print(f"  ❌ 数据不一致，per-object锁定可能有问题")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_performance_comparison():
    """性能对比测试"""
    print(f"\n📊 测试4: 完整性能对比")
    
    try:
        from arctic_inference.common.suffix_cache.thread_safe_suffix_cache import ThreadSafeSuffixCache
        from arctic_inference.common.suffix_cache.suffix_cache import SuffixCache
        
        # 生成测试数据
        def generate_test_data(num_problems=16, seqs_per_problem=8, tokens_per_seq=1000):
            problems_data = []
            for problem_id in range(num_problems):
                prompt_tokens = [random.randint(1, 5000) for _ in range(200)]
                sequences = []
                for seq_id in range(seqs_per_problem):
                    token_ids = [random.randint(1, 5000) for _ in range(tokens_per_seq)]
                    sequences.append(token_ids)
                problems_data.append((f"problem_{problem_id}", prompt_tokens, sequences))
            return problems_data
        
        problems_data = generate_test_data(20, 6, 1500)
        total_operations = len(problems_data) * 6 * 2  # problems * seqs * (extend prompt + extend tokens)
        
        print(f"  测试配置: {len(problems_data)} problems, 总操作数: {total_operations}")
        
        # 测试传统串行方法
        print("  测试传统SuffixCache（串行）...")
        traditional_cache = SuffixCache(max_depth=64)
        
        start_time = time.perf_counter()
        for problem_id, prompt_tokens, sequences in problems_data:
            for i, token_ids in enumerate(sequences):
                traditional_cache.prebuild_problemtree(-i-1, problem_id, prompt_tokens, token_ids)
        serial_time = time.perf_counter() - start_time
        
        # 测试线程安全并行方法
        print("  测试ThreadSafeSuffixCache（并行）...")
        thread_safe_cache = ThreadSafeSuffixCache(max_depth=64, max_threads=8)
        
        result = thread_safe_cache.build_problems_parallel(problems_data)
        parallel_time = result['total_time']
        
        # 性能对比
        speedup = serial_time / parallel_time if parallel_time > 0 else 1.0
        
        print(f"\n  📈 性能对比结果:")
        print(f"    串行时间: {serial_time:.4f}秒")
        print(f"    并行时间: {parallel_time:.4f}秒")
        print(f"    加速比: {speedup:.2f}x")
        print(f"    理论加速: {result.get('theoretical_speedup', 1.0):.2f}x")
        print(f"    成功处理: {result.get('successful_problems', 0)} problems")
        
        if speedup > 2.0:
            print(f"    ✅ 显著加速！C++对象级锁定方案成功")
            return True
        elif speedup > 1.3:
            print(f"    ⚡ 有一定加速效果")
            return True
        else:
            print(f"    ⚠️ 加速效果有限")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 C++对象级锁定方案验证测试")
    print("=" * 60)
    
    test_results = []
    
    # 执行所有测试
    test_results.append(("线程安全方法检查", test_thread_safe_methods_exist()))
    
    if test_results[-1][1]:  # 如果方法存在才继续测试
        test_results.append(("GIL释放效果验证", test_gil_release_effectiveness()))
        test_results.append(("Per-object锁定验证", test_per_object_locking()))
        test_results.append(("完整性能对比", test_performance_comparison()))
    
    # 汇总结果
    print(f"\n" + "=" * 60)
    print(f"🎯 测试结果汇总:")
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n总计: {passed_tests}/{len(test_results)} 测试通过")
    
    if passed_tests == len(test_results):
        print(f"🎉 所有测试通过！C++对象级锁定方案工作完美")
        print(f"✅ 可以在生产环境中使用ThreadSafeSuffixCache")
    elif passed_tests >= len(test_results) - 1:
        print(f"⚡ 大部分测试通过，方案基本可用")  
        print(f"⚠️ 请检查失败的测试项")
    else:
        print(f"❌ 多项测试失败，需要检查实现")
        print(f"💡 请确保：")
        print(f"   1. 重新编译了C++代码")
        print(f"   2. 线程安全方法正确实现")
        print(f"   3. Python绑定正确配置")
    
    return passed_tests == len(test_results)


if __name__ == "__main__":
    success = main()
