# app/api/endpoints.py
import re
import os
import uuid
import base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.data_service import data_service
from app.services.chat_service import chat_service
from app.schemas.role_schema import RoleCreateRequest
from vdb_tools.hierarchical_memory_db import HierarchicalMemoryManager

router = APIRouter()

class RoleSettingsUpdate(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    thinking_budget: Optional[int] = None
    enable_think: Optional[bool] = None

class RoleMetaUpdate(BaseModel):
    system_prompt: Optional[str] = None
    display_name: Optional[str] = None
    avatar_mode: Optional[str] = None
    settings: Optional[RoleSettingsUpdate] = None

class UserProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    avatar_mode: Optional[str] = None

# 【核心修改】：支持两段式头像的上传接收
class AvatarUploadRequest(BaseModel):
    target_type: str  # "user" 或 "role"
    role_id: Optional[str] = None
    image_circle_base64: str            # 接收 1:1 图
    image_bg_base64: Optional[str] = None # 接收 4:1 图

@router.get("/user")
async def get_user_profile():
    return data_service.user_manager.get_user()

@router.put("/user")
async def update_user_profile(data: UserProfileUpdate):
    return data_service.user_manager.update_user(data.model_dump(exclude_unset=True))

@router.post("/upload_avatar")
async def upload_avatar(data: AvatarUploadRequest):
    try:
        update_dict = {}
        relative_paths = {}

        # 处理 1:1 圆形头像
        if data.image_circle_base64:
            header, encoded = data.image_circle_base64.split(",", 1) if "," in data.image_circle_base64 else ("", data.image_circle_base64)
            filename_circle = f"avatar_circle_{uuid.uuid4().hex[:8]}.png"
            filepath_circle = os.path.join("data", "avatars", filename_circle)
            with open(filepath_circle, "wb") as f:
                f.write(base64.b64decode(encoded))
            update_dict["avatar_circle"] = f"/avatars/{filename_circle}"
            relative_paths["avatar_circle"] = update_dict["avatar_circle"]

        # 处理 4:1 渐变背景图
        if data.image_bg_base64:
            header, encoded = data.image_bg_base64.split(",", 1) if "," in data.image_bg_base64 else ("", data.image_bg_base64)
            filename_bg = f"avatar_bg_{uuid.uuid4().hex[:8]}.png"
            filepath_bg = os.path.join("data", "avatars", filename_bg)
            with open(filepath_bg, "wb") as f:
                f.write(base64.b64decode(encoded))
            update_dict["avatar_bg"] = f"/avatars/{filename_bg}"
            relative_paths["avatar_bg"] = update_dict["avatar_bg"]

        # 路由并更新内存
        # 路由并更新内存 (引入 .copy() 防止字典引用污染)
        if data.target_type == "user":
            data_service.user_manager.update_user(update_dict.copy())
        elif data.target_type == "role" and data.role_id:
            data_service.role_registry.update_role_settings(data.role_id, update_dict.copy())
            if data.role_id in chat_service.active_sessions:
                chat_service.active_sessions[data.role_id].memory_manager.update_meta_settings(update_dict.copy())
        else:
            raise ValueError("缺失归属目标 role_id")

        return {"status": "success", "paths": relative_paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"头像处理失败: {str(e)}")

@router.post("/roles")
async def create_role(data: RoleCreateRequest):
    role_info = data_service.role_registry.create_role(data.name)
    role_id = role_info["role_id"]
    settings = {
        "temperature": data.temperature,
        "top_p": data.top_p,
        "top_k": data.top_k,
        "repetition_penalty": data.repetition_penalty,
        "presence_penalty": data.presence_penalty,
        "thinking_budget": data.thinking_budget,
        "enable_think": data.enable_think
    }
    try:
        HierarchicalMemoryManager(
            role_id=role_id, role_name=data.name, system_prompt=data.system_prompt, initial_api_settings=settings
        )
        return {"status": "success", "role_id": role_id, "name": data.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"底层文件初始化失败: {str(e)}")

@router.get("/roles")
async def get_roles():
    return data_service.role_registry.get_all_roles()

@router.get("/roles/{role_id}/settings")
async def get_role_settings(role_id: str):
    session = chat_service.get_session(role_id)
    return {
        "system_prompt": session.memory_manager.meta_data.get("system_prompt", ""),
        "display_name": session.memory_manager.meta_data.get("display_name", session.role_name),
        "avatar_mode": session.memory_manager.meta_data.get("avatar_mode", "circle"),
        "avatar_circle": session.memory_manager.meta_data.get("avatar_circle", ""),
        "avatar_bg": session.memory_manager.meta_data.get("avatar_bg", ""),
        **session.memory_manager.meta_data.get("settings", {})
    }

@router.put("/roles/{role_id}/settings")
async def update_role_settings(role_id: str, data: RoleMetaUpdate):
    payload = data.model_dump(exclude_unset=True)
    # 【核心修复】：注入 .copy() 阻断双写时的内存污染
    success = data_service.role_registry.update_role_settings(role_id, payload.copy())
    if not success:
        raise HTTPException(status_code=500, detail="磁盘写入失败")

    if role_id in chat_service.active_sessions:
        session = chat_service.active_sessions[role_id]
        session.memory_manager.update_meta_settings(payload.copy())
        if data.display_name: session.role_name = data.display_name
    return {"status": "success"}

@router.get("/roles/{role_id}/history")
async def get_role_history(role_id: str):
    try:
        session = chat_service.get_session(role_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    context = session.memory_manager.context_buffer
    img_dir = os.path.join(session.memory_manager.base_dir, "images")
    history = []
    for msg in context:
        content = msg.get("content", "")
        role = msg.get("role", "")
        images = []
        matches = re.findall(r'\[IMAGE:\s*(.+?)\]', content)
        for filename in matches:
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                        mime = "image/png" if filename.endswith(".png") else "image/jpeg"
                        images.append(f"data:{mime};base64,{encoded}")
                except Exception:
                    pass
        clean_content = re.sub(r'\n?\[IMAGE:\s*.+?\]', '', content).strip()
        history.append({"role": role, "content": clean_content, "images": images})
    return history