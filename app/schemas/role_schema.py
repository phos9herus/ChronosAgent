from pydantic import BaseModel, Field
from typing import Optional, List

class RoleCreateRequest(BaseModel):
    """创建新角色的请求"""
    name: str = Field(..., description="新角色的名称")
    system_prompt: Optional[str] = Field(None, description="角色的底层 System Prompt")

class RoleResponse(BaseModel):
    """返回给前端的角色基础信息"""
    role_id: str
    role_name: str