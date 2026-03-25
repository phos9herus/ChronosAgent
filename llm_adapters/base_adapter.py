from abc import ABC, abstractmethod
from typing import Generator, Tuple


class BaseLLMAdapter(ABC):
    """
    大语言模型适配器基类。
    未来无论是对接网页端逆向、还是官方标准 API，都应继承此接口。
    """

    @abstractmethod
    def stream_chat(self,
                    prompt: str,
                    original_input: str = None,
                    image_mode: bool = False,
                    image_size: str = "1:1",
                    **kwargs) -> Generator[Tuple[str, str], None, None]:
        """
        流式对话接口。

        必须通过 yield 返回以下标准格式的元组 (message_type, content):
        - ("answer", "普通的回答文本块")
        - ("thought", "深度思考/推理过程的文本块")
        - ("status", "系统状态通知，如画图已送达")
        - ("error", "错误信息")
        - ("auth_error", "鉴权失败信息")
        - ("debug", "底层原始数据流(用于调试)")
        """
        pass