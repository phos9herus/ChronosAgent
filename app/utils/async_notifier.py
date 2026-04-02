import asyncio
import threading
from typing import Optional

# 全局事件循环引用
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_lock = threading.Lock()


def set_event_loop(loop: asyncio.AbstractEventLoop):
    """设置全局事件循环"""
    global _event_loop
    with _lock:
        _event_loop = loop


def get_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """获取全局事件循环"""
    with _lock:
        return _event_loop


def run_async(coro):
    """在全局事件循环中运行协程"""
    loop = get_event_loop()
    if loop and not loop.is_closed():
        asyncio.run_coroutine_threadsafe(coro, loop)
