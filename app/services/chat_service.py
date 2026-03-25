import os
from typing import Dict
from roleplay_core import RoleplaySession
from llm_adapters.qwen_native_adapter import QwenNativeAdapter
from app.services.data_service import data_service


class ChatService:
    """管理所有活跃的角色扮演会话 (单例管家)"""

    def __init__(self):
        # 字典结构：{ "role_id": RoleplaySession 实例 }
        # 保持实例存活非常重要，这样底层 _monitor_loop 线程才能持续在后台压缩记忆
        self.active_sessions: Dict[str, RoleplaySession] = {}

    def get_session(self, role_id: str) -> RoleplaySession:
        """获取已存在的会话，如果不存在则初始化一个新的会话"""
        if role_id in self.active_sessions:
            return self.active_sessions[role_id]

        # 1. 查询角色信息是否存在
        role_info = data_service.role_registry.get_role_by_id(role_id)
        if not role_info:
            raise ValueError(f"角色 ID '{role_id}' 不存在，请先创建角色。")

        # 2. 获取鉴权 API Key
        # 优先尝试从环境变量获取 (与你 CLI 的高优逻辑保持一致)
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")

        # 如果环境变量没有，再降级从 json 字典中读取
        if not api_key:
            api_key = data_service.auth_manager.credentials.get("qwen_api", {}).get("api_key", "")

        if not api_key:
            raise ValueError("系统未配置 Qwen API Key，请配置环境变量或在 data/credentials.json 中配置。")

        # 3. 初始化底层适配器与会话核心
        adapter = QwenNativeAdapter(api_key=api_key)

        session = RoleplaySession(
            adapter=adapter,
            role_id=role_id,
            role_name=role_info.get("name", "未命名角色")
        )

        # 4. 存入内存管家
        self.active_sessions[role_id] = session
        return session

    def remove_session(self, role_id: str):
        """手动移除并清理某个角色的会话内存"""
        if role_id in self.active_sessions:
            session = self.active_sessions.pop(role_id)
            session.shutdown_and_flush()

    def shutdown_all(self):
        """Web 服务主进程关闭时，安全清理所有会话的未压缩记忆"""
        for role_id, session in self.active_sessions.items():
            session.shutdown_and_flush()
        self.active_sessions.clear()


# 实例化单例，供 API 路由直接调用
chat_service = ChatService()