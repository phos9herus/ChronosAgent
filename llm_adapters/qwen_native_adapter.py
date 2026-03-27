# llm_adapters/qwen_native_adapter.py
import os
import json
import time
import logging
from typing import Generator, Tuple
import dashscope
from .base_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class QwenNativeAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, model: str = "qwen3.5-plus", thinking_budget: int = 81920):
        if not api_key:
            raise ValueError("未检测到合法的 API Key。")
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
        dashscope.api_key = api_key
        self.default_model = model
        self.thinking_budget = thinking_budget

    def _is_text_model(self, model: str) -> bool:
        text_models = {"qwen-plus", "qwen-turbo", "qwen-max", "qwen3-plus", "qwen3-max"}
        return model in text_models

    def stream_chat(self, prompt: str, images: list = None, **kwargs) -> Generator[Tuple[str, str], None, None]:
        enable_think = kwargs.pop("enable_think", False)
        dynamic_budget = kwargs.pop("thinking_budget", self.thinking_budget)
        model = kwargs.pop("model", self.default_model)
        is_text_model = self._is_text_model(model)

        messages = kwargs.pop("messages", [])
        formatted_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if is_text_model:
                if isinstance(content, str):
                    formatted_messages.append({"role": msg["role"], "content": content})
                else:
                    formatted_messages.append(msg)
            else:
                if isinstance(content, str):
                    formatted_messages.append({"role": msg["role"], "content": [{"text": content}]})
                else:
                    formatted_messages.append(msg)

        if is_text_model:
            formatted_messages.append({"role": "user", "content": prompt})
            request_params = {
                "model": model, 
                "messages": formatted_messages, 
                "stream": True,
                "result_format": "message", 
                "max_tokens": 32768,
                "timeout": 60  # 添加 60 秒超时控制
            }
            if enable_think:
                request_params["enable_thinking"] = True
                request_params["thinking_budget"] = int(dynamic_budget)
            request_params.update(kwargs)
        else:
            current_content = [{"text": prompt}]
            if images:
                for img in images:
                    current_content.append({"image": img})
            formatted_messages.append({"role": "user", "content": current_content})
            request_params = {
                "model": model, 
                "messages": formatted_messages, 
                "stream": True,
                "incremental_output": True, 
                "result_format": "message", 
                "max_tokens": 8192,
                "enable_thinking": bool(enable_think),
                "timeout": 60  # 添加 60 秒超时控制
            }
            if request_params["enable_thinking"]:
                request_params["thinking_budget"] = int(dynamic_budget)
            request_params.update(kwargs)

        request_start_time = time.time()
        logger.debug(f"[Qwen API] 开始请求 - 模型：{model}")
        
        try:
            if is_text_model:
                responses = dashscope.Generation.call(**request_params)
            else:
                responses = dashscope.MultiModalConversation.call(**request_params)
            
            first_response_time = None
            response_count = 0
            
            for response in responses:
                response_count += 1
                if first_response_time is None:
                    first_response_time = time.time()
                    elapsed = first_response_time - request_start_time
                    logger.info(f"[Qwen API] 首字延迟：{elapsed:.2f}秒")
                
                if response.status_code == 200:
                    choice = response.output.choices[0]
                    message = choice.message

                    def safe_get(obj, key):
                        if isinstance(obj, dict):
                            return obj.get(key, None)
                        return getattr(obj, key, None)

                    reasoning = safe_get(message, 'reasoning_content')
                    if reasoning:
                        text = "".join([item.get("text", "") for item in reasoning]) if isinstance(reasoning,
                                                                                                   list) else str(
                            reasoning)
                        if text: yield "thought", text

                    content_data = safe_get(message, 'content')
                    if content_data:
                        if is_text_model:
                            text = str(content_data)
                        else:
                            text = "".join([item.get("text", "") for item in content_data]) if isinstance(content_data,
                                                                                                          list) else str(
                                content_data)
                        if text: yield "answer", text

                    usage = getattr(response, 'usage', None)
                    if usage:
                        yield "usage", {
                            "input": getattr(usage, 'input_tokens', 0),
                            "output": getattr(usage, 'output_tokens', 0),
                            "total": getattr(usage, 'total_tokens', 0),
                            "model": model
                        }

                else:
                    elapsed = time.time() - request_start_time
                    logger.error(f"[Qwen API] API错误：{response.message}, 耗时：{elapsed:.2f}秒")
                    yield "error", f"API Error: {response.message}"
        except TimeoutError as e:
            elapsed = time.time() - request_start_time
            logger.error(f"[Qwen API] 请求超时：{str(e)}, 耗时：{elapsed:.2f}秒")
            yield "error", f"请求超时：{str(e)}，请检查网络连接或重试"
        except ConnectionError as e:
            elapsed = time.time() - request_start_time
            logger.error(f"[Qwen API] 连接错误：{str(e)}, 耗时：{elapsed:.2f}秒")
            yield "error", f"网络连接失败：{str(e)}"
        except Exception as e:
            elapsed = time.time() - request_start_time
            logger.error(f"[Qwen API] 流式解析异常：{str(e)}, 耗时：{elapsed:.2f}秒")
            logger.debug("异常详情：", exc_info=True)
            yield "error", f"流式解析异常：{str(e)}"
        finally:
            total_duration = time.time() - request_start_time
            logger.debug(f"[Qwen API] 请求完成，总耗时：{total_duration:.2f}秒")