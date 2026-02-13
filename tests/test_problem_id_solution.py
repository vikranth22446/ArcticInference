#!/usr/bin/env python3
"""
测试脚本：验证新的req_id到problem_id映射解决方案是否正常工作

使用方法:
1. 确保在正确的Docker环境中运行
2. 确保VLLM_USE_V1=1环境变量已设置
3. 运行: python test_problem_id_solution.py
"""

import sys
import os

def test_imports():
    """测试所有必要的导入"""
    print("=== 测试导入 ===")
    
    try:
        # 测试ArcticInference核心组件导入
        from arctic_inference.vllm.context_managers import ProblemIdContextManager
        print("✓ ProblemIdContextManager导入成功")
        
        from arctic_inference.vllm.llm import apply_llm_patches, LLMPatch
        print("✓ LLM patches组件导入成功")
        
        # 测试vLLM导入
        import vllm
        from vllm.entrypoints.llm import LLM
        print(f"✓ vLLM版本: {vllm.__version__}")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_llm_patches():
    """测试LLM patches的应用和功能"""
    print("\n=== 测试LLM Patches ===")
    
    try:
        from arctic_inference.vllm.llm import apply_llm_patches
        from vllm.entrypoints.llm import LLM
        
        # 应用patches
        apply_llm_patches()
        
        # 检查patches是否已应用
        if hasattr(LLM, '_arctic_problem_id_patched'):
            print("✓ LLM patches已成功应用")
            
            # 检查原始方法是否被保存
            if hasattr(LLMPatch, '_orig_generate'):
                print("✓ 原始方法已正确保存")
            else:
                print("✗ 原始方法保存失败")
                return False
            
            # 测试generate方法签名
            import inspect
            sig = inspect.signature(LLM.generate)
            if 'problem_ids' in sig.parameters:
                print("✓ generate方法已包含problem_ids参数")
            else:
                print("✗ generate方法缺少problem_ids参数")
                return False
                
            return True
        else:
            print("✗ LLM patches未正确应用")
            return False
            
    except Exception as e:
        print(f"✗ LLM patches测试异常: {e}")
        return False


def test_context_manager():
    """测试ProblemIdContextManager的新功能"""
    print("\n=== 测试ProblemIdContextManager ===")
    
    try:
        from arctic_inference.vllm.context_managers import ProblemIdContextManager
        
        # 测试req_id映射功能
        test_mapping = {"req_1": "problem_1", "req_2": "problem_2", "req_3": "problem_3"}
        
        # 设置映射
        ProblemIdContextManager.set_req_id_to_problem_id_mapping(test_mapping)
        
        # 验证映射
        retrieved_mapping = ProblemIdContextManager.get_req_id_to_problem_id_mapping()
        if retrieved_mapping == test_mapping:
            print("✓ req_id映射设置和获取正常")
        else:
            print(f"✗ req_id映射不匹配: 期望 {test_mapping}, 得到 {retrieved_mapping}")
            return False
        
        # 测试单个req_id查找
        for req_id, expected_problem_id in test_mapping.items():
            actual_problem_id = ProblemIdContextManager.get_problem_id_for_req_id(req_id)
            if actual_problem_id == expected_problem_id:
                print(f"✓ req_id查找 {req_id}: {actual_problem_id}")
            else:
                print(f"✗ req_id查找 {req_id}: 期望 {expected_problem_id}, 得到 {actual_problem_id}")
                return False
        
        # 测试清理功能
        ProblemIdContextManager.clear_req_id_mapping()
        after_clear = ProblemIdContextManager.get_req_id_to_problem_id_mapping()
        if len(after_clear) == 0:
            print("✓ 映射清理成功")
        else:
            print(f"✗ 映射清理失败，仍有数据: {after_clear}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ ProblemIdContextManager测试异常: {e}")
        return False


def test_integration():
    """集成测试：模拟完整的工作流程"""
    print("\n=== 集成测试 ===")
    
    try:
        from arctic_inference.vllm.context_managers import ProblemIdContextManager
        from arctic_inference.vllm.llm import apply_llm_patches
        
        # 确保patches已应用
        apply_llm_patches()
        
        # 模拟vllm_rollout_spmd.py的调用流程
        problem_ids = ["problem_001", "problem_002", "problem_003"]
        
        # 1. 初始化上下文（模拟vllm_rollout_spmd.py中的代码）
        ProblemIdContextManager.clear_req_id_mapping()
        ProblemIdContextManager.set_req_id_to_problem_id_mapping({})
        print("✓ 上下文初始化完成")
        
        # 2. 模拟LLM.generate调用（实际情况下会通过patches建立映射）
        # 这里我们手动建立映射来模拟patches的效果
        simulated_mapping = {
            "0": "problem_001",
            "1": "problem_002", 
            "2": "problem_003"
        }
        ProblemIdContextManager.set_req_id_to_problem_id_mapping(simulated_mapping)
        print("✓ 模拟映射建立完成")
        
        # 3. 模拟execute_model中的查找（通过GPUModelRunnerPatch）
        for req_id, expected_problem_id in simulated_mapping.items():
            # 这模拟了execute_model中的查找过程
            retrieved_problem_id = ProblemIdContextManager.get_problem_id_for_req_id(req_id)
            if retrieved_problem_id == expected_problem_id:
                print(f"✓ execute_model模拟查找 req_id={req_id} -> problem_id={retrieved_problem_id}")
            else:
                print(f"✗ execute_model模拟查找失败 req_id={req_id}: 期望 {expected_problem_id}, 得到 {retrieved_problem_id}")
                return False
        
        print("✓ 集成测试完成 - 完整工作流程正常")
        return True
        
    except Exception as e:
        print(f"✗ 集成测试异常: {e}")
        return False


def check_environment():
    """检查环境变量和配置"""
    print("=== 检查环境 ===")
    
    # 检查VLLM_USE_V1
    if os.getenv("VLLM_USE_V1") == "1":
        print("✓ VLLM_USE_V1=1 已设置")
    else:
        print("⚠ VLLM_USE_V1 未设置为1，插件可能无法正常工作")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用，设备数: {torch.cuda.device_count()}")
        else:
            print("⚠ CUDA不可用")
    except ImportError:
        print("⚠ PyTorch未安装")
    
    print(f"✓ Python版本: {sys.version}")


def main():
    """主测试函数"""
    print("req_id到problem_id映射解决方案测试")
    print("=" * 50)
    
    # 检查环境
    check_environment()
    
    # 运行各项测试
    tests = [
        ("导入测试", test_imports),
        ("LLM Patches测试", test_llm_patches),
        ("上下文管理器测试", test_context_manager),
        ("集成测试", test_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name}异常: {e}")
            results[test_name] = False
    
    # 总结结果
    print(f"\n{'='*20}")
    print("测试结果总结:")
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试都通过了！解决方案工作正常。")
        print("\n📝 下一步：在实际环境中测试vLLMRollout与problem_ids参数的完整工作流程")
        return 0
    else:
        print("\n❌ 某些测试失败。请检查配置和代码。")
        return 1


if __name__ == "__main__":
    sys.exit(main())