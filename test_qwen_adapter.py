import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_adapters.qwen_native_adapter import QwenNativeAdapter

def test_qwen_plus_text_model():
    """测试 qwen-plus 文本模型"""
    print("=" * 60)
    print("测试 1: qwen-plus 文本模型")
    print("=" * 60)
    
    # 从环境变量获取 API Key，如果没有则使用测试 key
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    if not api_key:
        print("❌ 错误: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("请先设置环境变量: set DASHSCOPE_API_KEY=your_api_key")
        return False
    
    try:
        adapter = QwenNativeAdapter(api_key=api_key, model="qwen-plus")
        print("✓ QwenNativeAdapter 初始化成功")
        print(f"✓ 默认模型: {adapter.default_model}")
        
        # 测试模型类型检测
        is_text = adapter._is_text_model("qwen-plus")
        print(f"✓ 模型类型检测: qwen-plus 是文本模型 = {is_text}")
        
        if not is_text:
            print("❌ 错误: 模型类型检测失败")
            return False
        
        # 测试流式对话
        print("\n开始流式对话测试...")
        messages = [
            {"role": "user", "content": "你好，请用一句话介绍你自己"}
        ]
        
        response_text = ""
        for chunk_type, content in adapter.stream_chat("你好，请用一句话介绍你自己", messages=messages):
            if chunk_type == "answer":
                response_text += content
                print(content, end="", flush=True)
            elif chunk_type == "error":
                print(f"\n❌ 错误: {content}")
                return False
            elif chunk_type == "usage":
                print(f"\n\n✓ Token 使用情况: {content}")
        
        if response_text:
            print(f"\n✓ 文本模型测试成功，获得响应")
            return True
        else:
            print("\n❌ 错误: 未获得响应")
            return False
            
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_qwen_vl_plus_multimodal_model():
    """测试 qwen-vl-plus 多模态模型"""
    print("\n" + "=" * 60)
    print("测试 2: qwen-vl-plus 多模态模型")
    print("=" * 60)
    
    # 从环境变量获取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    if not api_key:
        print("❌ 错误: 未设置 DASHSCOPE_API_KEY 环境变量")
        return False
    
    try:
        adapter = QwenNativeAdapter(api_key=api_key, model="qwen-vl-plus")
        print("✓ QwenNativeAdapter 初始化成功")
        print(f"✓ 默认模型: {adapter.default_model}")
        
        # 测试模型类型检测
        is_text = adapter._is_text_model("qwen-vl-plus")
        print(f"✓ 模型类型检测: qwen-vl-plus 是文本模型 = {is_text}")
        
        if is_text:
            print("❌ 错误: 模型类型检测失败，qwen-vl-plus 应该是多模态模型")
            return False
        
        # 测试流式对话（仅文本）
        print("\n开始流式对话测试（仅文本）...")
        messages = [
            {"role": "user", "content": "你好，请用一句话介绍你自己"}
        ]
        
        response_text = ""
        for chunk_type, content in adapter.stream_chat("你好，请用一句话介绍你自己", messages=messages):
            if chunk_type == "answer":
                response_text += content
                print(content, end="", flush=True)
            elif chunk_type == "error":
                print(f"\n❌ 错误: {content}")
                return False
            elif chunk_type == "usage":
                print(f"\n\n✓ Token 使用情况: {content}")
        
        if response_text:
            print(f"\n✓ 多模态模型测试成功，获得响应")
            return True
        else:
            print("\n❌ 错误: 未获得响应")
            return False
            
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_type_detection():
    """测试模型类型检测方法"""
    print("\n" + "=" * 60)
    print("测试 3: 模型类型检测方法")
    print("=" * 60)
    
    api_key = os.getenv("DASHSCOPE_API_KEY", "test_key")
    
    try:
        adapter = QwenNativeAdapter(api_key=api_key, model="qwen-plus")
        
        # 测试各种模型
        test_cases = [
            ("qwen-plus", True),
            ("qwen-turbo", True),
            ("qwen-max", True),
            ("qwen3-plus", True),
            ("qwen3-max", True),
            ("qwen-vl-plus", False),
            ("qwen-vl-max", False),
            ("qwen-vl-plus-v2", False),
        ]
        
        all_passed = True
        for model, expected_is_text in test_cases:
            is_text = adapter._is_text_model(model)
            status = "✓" if is_text == expected_is_text else "❌"
            print(f"{status} {model}: 文本模型={is_text}, 预期={expected_is_text}")
            if is_text != expected_is_text:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Qwen Native Adapter 测试套件")
    print("=" * 60)
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("⚠️  警告: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("API 连接测试将被跳过")
        print("请使用以下命令设置 API Key:")
        print("set DASHSCOPE_API_KEY=your_api_key")
        print()
    
    # 运行测试
    results = []
    
    # 测试 1: 模型类型检测（不需要 API Key）
    result = test_model_type_detection()
    results.append(("模型类型检测", result))
    
    # 测试 2: 文本模型（需要 API Key）
    if api_key:
        result = test_qwen_plus_text_model()
        results.append(("qwen-plus 文本模型", result))
    else:
        print("\n跳过 qwen-plus 文本模型测试（未设置 API Key）")
        results.append(("qwen-plus 文本模型", None))
    
    # 测试 3: 多模态模型（需要 API Key）
    if api_key:
        result = test_qwen_vl_plus_multimodal_model()
        results.append(("qwen-vl-plus 多模态模型", result))
    else:
        print("\n跳过 qwen-vl-plus 多模态模型测试（未设置 API Key）")
        results.append(("qwen-vl-plus 多模态模型", None))
    
    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results:
        if result is True:
            print(f"✓ {test_name}: 通过")
            passed += 1
        elif result is False:
            print(f"❌ {test_name}: 失败")
            failed += 1
        else:
            print(f"⏭️  {test_name}: 跳过")
            skipped += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败, {skipped} 跳过")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n❌ 有 {failed} 个测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)