from pydantic import BaseModel, Field
from typing import Optional

class RoleCreateRequest(BaseModel):
    """创建新角色的请求 (完整版)"""
    name: str = Field(..., description="新角色的名称")
    system_prompt: Optional[str] = Field("", description="角色的底层 System Prompt")
    # 以下为默认参数配置，前端如果不传，后端也有保底
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    thinking_budget: int = 81920
    enable_think: bool = True

class RoleResponse(BaseModel):
    role_id: str
    role_name: str