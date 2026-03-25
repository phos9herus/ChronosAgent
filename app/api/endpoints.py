# app/api/endpoints.py
import re
import os
import base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.data_service import data_service
from app.services.chat_service import chat_service

router = APIRouter()

# 【修复】将模型超参数与角色核心指令隔离
class RoleSettingsUpdate(BaseModel):
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    presence_penalty: float
    thinking_budget: int
    enable_think: bool

class RoleMetaUpdate(BaseModel):
    system_prompt: str
    settings: RoleSettingsUpdate

@router.get("/roles")
async def get_roles():
    """获取系统中所有的角色列表供前端侧边栏渲染"""
    # 直接调用底层的 role_registry 获取列表
    roles = data_service.role_registry.get_all_roles()
    return roles

@router.get("/roles/{role_id}/settings")
async def get_role_settings(role_id: str):
    session = chat_service.get_session(role_id)
    # 将分散的数据合并给前端表单使用
    system_prompt = session.memory_manager.meta_data.get("system_prompt", "")
    settings = session.memory_manager.meta_data.get("settings", {})
    return {"system_prompt": system_prompt, **settings}

@router.put("/roles/{role_id}/settings")
async def update_role_settings(role_id: str, data: RoleMetaUpdate):
    # 1. 持久化到磁盘 (保存正确的拓扑层级)
    success = data_service.role_registry.update_role_settings(role_id, data.model_dump())
    if not success:
        raise HTTPException(status_code=500, detail="磁盘写入失败，请检查角色文件夹是否存在。")

    # 2. 同步更新内存中活跃的会话
    if role_id in chat_service.active_sessions:
        session = chat_service.active_sessions[role_id]
        session.memory_manager.update_meta_settings(data.model_dump())

    return {"status": "success"}


@router.get("/roles/{role_id}/history")
async def get_role_history(role_id: str):
    """获取角色的历史聊天上下文，并还原图片占位符为 Base64"""
    try:
        session = chat_service.get_session(role_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 直接读取内存管家中的活跃上下文缓存
    context = session.memory_manager.context_buffer

    # 定位资源目录
    role_dir = session.memory_manager.base_dir
    img_dir = os.path.join(role_dir, "images")

    history = []
    for msg in context:
        content = msg.get("content", "")
        role = msg.get("role", "")
        images = []

        # 1. 寻找所有的图片占位符 [IMAGE: img_1234abcd.jpg]
        matches = re.findall(r'\[IMAGE:\s*(.+?)\]', content)
        for filename in matches:
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                        mime = "image/png" if filename.endswith(".png") else "image/jpeg"
                        # 组装为前端可以直接渲染的 Data URL
                        images.append(f"data:{mime};base64,{encoded}")
                except Exception as e:
                    print(f"读取图片失败 {filename}: {e}")

        # 2. 清理文本中的占位符，防止在前端 UI 中以文字形式暴露
        clean_content = re.sub(r'\n?\[IMAGE:\s*.+?\]', '', content).strip()

        history.append({
            "role": role,
            "content": clean_content,
            "images": images
        })

    return history
