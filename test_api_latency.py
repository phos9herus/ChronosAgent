"""
Qwen API 响应时间测试脚本
用于诊断 API 调用延迟问题
"""
import os
import sys
import time
import dashscope
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_adapters.qwen_native_adapter import QwenNativeAdapter


def test_api_connection():
    """测试 API 连接性和响应时间"""
    print("=" * 80)
    print("Qwen API 连接性和响应时间测试")
    print("=" * 80)
    print(f"测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    
    if not api_key:
        print("❌ 错误：未设置 API Key")
        print("请设置环境变量：set DASHSCOPE_API_KEY=your_api_key")
        return False
    
    print(f"✓ API Key 已配置")
    print()
    
    # 测试不同模型
    test_models = [
        "qwen-plus",
        "qwen-turbo",
        "qwen3.5-plus"
    ]
    
    test_prompt = "你好，请用一句话回答：1+1 等于几？"
    
    results = []
    
    for model_name in test_models:
        print(f"测试模型：{model_name}")
        print("-" * 80)
        
        try:
            adapter = QwenNativeAdapter(api_key=api_key, model=model_name)
            
            # 记录开始时间
            start_time = time.time()
            first_token_time = None
            full_response = ""
            total_tokens = {"input": 0, "output": 0}
            
            # 流式调用
            first_token_time = None
            for chunk_type, content in adapter.stream_chat(test_prompt):
                if chunk_type == "answer":
                    if first_token_time is None:
                        first_token_time = time.time()
                        time_to_first_token = first_token_time - start_time
                        print(f"⏱️  首字延迟：{time_to_first_token:.2f}秒")
                    
                    full_response += content
                    print(f"📝 接收中... ({len(full_response)} 字符)", end='\r')
                
                elif chunk_type == "usage":
                    total_tokens = content
                
                elif chunk_type == "error":
                    print(f"❌ API 错误：{content}")
                    break
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print()  # 换行
            print(f"✓ 完成")
            print(f"  - 总响应时间：{total_time:.2f}秒")
            print(f"  - 首字延迟：{time_to_first_token:.2f}秒 (如果已记录)")
            print(f"  - 响应长度：{len(full_response)} 字符")
            print(f"  - Token 使用：输入={total_tokens.get('input', 0)}, 输出={total_tokens.get('output', 0)}")
            
            results.append({
                "model": model_name,
                "success": True,
                "total_time": total_time,
                "first_token_time": time_to_first_token if first_token_time else None,
                "response_length": len(full_response)
            })
            
        except Exception as e:
            print(f"❌ 测试失败：{str(e)}")
            results.append({
                "model": model_name,
                "success": False,
                "error": str(e)
            })
        
        print()
    
    # 打印汇总报告
    print("=" * 80)
    print("测试汇总报告")
    print("=" * 80)
    
    for result in results:
        status = "✓ 成功" if result["success"] else "❌ 失败"
        print(f"{result['model']}: {status}")
        if result["success"]:
            print(f"  - 总耗时：{result['total_time']:.2f}秒")
            if result.get('first_token_time'):
                print(f"  - 首字延迟：{result['first_token_time']:.2f}秒")
        else:
            print(f"  - 错误：{result.get('error', '未知错误')}")
        print()
    
    # 判断是否存在性能问题
    print("=" * 80)
    print("性能诊断")
    print("=" * 80)
    
    successful_tests = [r for r in results if r["success"]]
    
    if not successful_tests:
        print("❌ 所有测试均失败，请检查网络连接或 API Key 配置")
        return False
    
    avg_total_time = sum(r["total_time"] for r in successful_tests) / len(successful_tests)
    avg_first_token = sum(r.get("first_token_time", 0) for r in successful_tests if r.get("first_token_time")) / len(successful_tests)
    
    print(f"平均总响应时间：{avg_total_time:.2f}秒")
    print(f"平均首字延迟：{avg_first_token:.2f}秒")
    print()
    
    # 性能判断标准
    if avg_first_token > 10:
        print("⚠️  警告：首字延迟超过 10 秒，可能存在以下问题：")
        print("   1. 网络连接不稳定")
        print("   2. API 服务器负载过高")
        print("   3. 本地网络环境限制（防火墙、代理等）")
        return False
    elif avg_first_token > 5:
        print("⚠️  注意：首字延迟略高（5-10 秒），建议检查网络环境")
        return True
    else:
        print("✓ 响应时间正常")
        return True


def test_network_quality():
    """测试网络质量"""
    print("\n" + "=" * 80)
    print("网络质量测试")
    print("=" * 80)
    
    import requests
    
    # 测试到阿里云 API 的网络延迟
    test_urls = [
        ("阿里云 DashScope API", "https://dashscope.aliyuncs.com"),
        ("通义千问网页", "https://chat2.qianwen.com"),
        ("百度（国内参考）", "https://www.baidu.com"),
    ]
    
    for name, url in test_urls:
        try:
            start = time.time()
            response = requests.get(url, timeout=10)
            elapsed = time.time() - start
            print(f"{name}: {elapsed*1000:.0f}ms (HTTP {response.status_code})")
        except Exception as e:
            print(f"{name}: 连接失败 - {str(e)}")
    
    print()


if __name__ == "__main__":
    try:
        # 测试网络质量
        test_network_quality()
        
        # 测试 API 响应
        success = test_api_connection()
        
        if success:
            print("\n✓ API 响应时间测试完成，结果正常")
            sys.exit(0)
        else:
            print("\n❌ API 响应时间测试发现问题，请检查上述诊断建议")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
