# llm_adapters/qwen_native_adapter.py
import os
import json
from typing import Generator, Tuple
import dashscope
from .base_adapter import BaseLLMAdapter


class QwenNativeAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, model: str = "qwen3.5-plus", thinking_budget: int = 81920):
        if not api_key:
            raise ValueError("未检测到合法的 API Key。")
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
        dashscope.api_key = api_key
        self.model = model
        self.thinking_budget = thinking_budget

    def stream_chat(self, prompt: str, images: list = None, **kwargs) -> Generator[Tuple[str, str], None, None]:
        # 1. 提取前端/核心层传来的思考模式与动态预算
        # 默认设为 False，完全由外部控制
        enable_think = kwargs.pop("enable_think", False)
        dynamic_budget = kwargs.pop("thinking_budget", self.thinking_budget)

        # 2. 构造多模态消息体
        current_content = [{"text": prompt}]
        if images:
            for img in images:
                current_content.append({"image": img})

        messages = kwargs.pop("messages", [])
        formatted_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                formatted_messages.append({"role": msg["role"], "content": [{"text": content}]})
            else:
                formatted_messages.append(msg)
        formatted_messages.append({"role": "user", "content": current_content})

        # 3. 构造基础请求体
        request_params = {"model": self.model, "messages": formatted_messages, "stream": True,
                          "incremental_output": True, "result_format": "message", "max_tokens": 65536,
                          "enable_thinking": bool(enable_think)}

        # ==========================================
        # 核心修复：永远显式传递 enable_thinking
        # 这样你在终端无论开关都会看到它的状态
        # ==========================================

        # 如果开启了思考模式，则追加预算参数
        if request_params["enable_thinking"]:
            request_params["thinking_budget"] = int(dynamic_budget)

        # 5. 合并前端可能传来的其他参数（temperature 等）
        request_params.update(kwargs)

        # ==========================================
        # 终端调试打印区域
        # ==========================================
        # print("\n" + "=" * 50)
        # print("发送给 Qwen API 的完整请求体:")
        # try:
        #     print(json.dumps(request_params, ensure_ascii=False, indent=2))
        # except Exception:
        #     print(request_params)
        # print("=" * 50 + "\n")

        try:
            responses = dashscope.MultiModalConversation.call(**request_params)
            for response in responses:
                if response.status_code == 200:
                    choice = response.output.choices[0]
                    message = choice.message

                    # 安全访问机制
                    def safe_get(obj, key):
                        if isinstance(obj, dict):
                            return obj.get(key, None)
                        return getattr(obj, key, None)

                    # 1. 捕获思考过程
                    reasoning = safe_get(message, 'reasoning_content')
                    if reasoning:
                        text = "".join([item.get("text", "") for item in reasoning]) if isinstance(reasoning,
                                                                                                   list) else str(
                            reasoning)
                        if text: yield "thought", text

                    # 2. 捕获正式回复
                    content_data = safe_get(message, 'content')
                    if content_data:
                        text = "".join([item.get("text", "") for item in content_data]) if isinstance(content_data,
                                                                                                      list) else str(
                            content_data)
                        if text: yield "answer", text

                    # 3. 捕获 Token 消耗 (通常在流的最后几个 chunk 中完整返回)
                    usage = getattr(response, 'usage', None)
                    if usage:
                        yield "usage", {
                            "input": getattr(usage, 'input_tokens', 0),
                            "output": getattr(usage, 'output_tokens', 0),
                            "total": getattr(usage, 'total_tokens', 0)
                        }

                else:
                    yield "error", f"API Error: {response.message}"
        except Exception as e:
            yield "error", f"流式解析异常: {str(e)}"