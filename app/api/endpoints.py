# app/api/endpoints.py
import re
import os
import uuid
import base64
import copy
import shutil
import sys
import signal
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from app.services.data_service import data_service
from app.services.chat_service import chat_service
from app.services.stats_service import stats_service
from app.schemas.role_schema import RoleCreateRequest
from app.schemas.knowledge_schema import (
    KnowledgeBaseCreate, KnowledgeBaseUpdate,
    KnowledgeBaseResponse, KnowledgeBaseListItem,
    DocumentInfo, DocumentUploadResponse, DocumentDetailResponse
)
from app.services.knowledge_service import knowledge_service
from vdb_tools.hierarchical_memory_db import HierarchicalMemoryManager
from app.config.models import get_all_model_details, is_valid_model

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

class GlobalSettingsUpdate(BaseModel):
    model: Optional[str] = None

class ConversationCreateRequest(BaseModel):
    name: Optional[str] = None

class ConversationUpdateRequest(BaseModel):
    name: str

class DepthRecallModeUpdate(BaseModel):
    depth_recall_mode: str

# 支持两段式头像的上传接收
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

@router.get("/models")
async def get_models_list():
    """
    返回可用的Qwen模型列表及详细信息
    """
    return {"models": get_all_model_details()}

@router.put("/settings")
async def update_global_settings(data: GlobalSettingsUpdate):
    """
    更新全局设置，如模型选择
    """
    if data.model:
        # 验证模型是否在可用列表中
        if not is_valid_model(data.model):
            raise HTTPException(status_code=400, detail=f"无效的模型: {data.model}")
        
        # 保存到用户配置中
        update_dict = {"preferred_model": data.model}
        data_service.user_manager.update_user(update_dict)
    
    return {"status": "success"}

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
        # 使用深拷贝防止字典引用污染
        if data.target_type == "user":
            data_service.user_manager.update_user(copy.deepcopy(update_dict))
        elif data.target_type == "role" and data.role_id:
            data_service.role_registry.update_role_settings(data.role_id, copy.deepcopy(update_dict))
            if data.role_id in chat_service.active_sessions:
                chat_service.active_sessions[data.role_id].memory_manager.update_meta_settings(copy.deepcopy(update_dict))
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

@router.get("/roles/{role_id}/depth_recall_mode")
async def get_depth_recall_mode(role_id: str):
    session = chat_service.get_session(role_id)
    return {
        "depth_recall_mode": session.memory_manager.get_depth_recall_mode()
    }

@router.put("/roles/{role_id}/depth_recall_mode")
async def set_depth_recall_mode(role_id: str, data: DepthRecallModeUpdate):
    session = chat_service.get_session(role_id)
    session.memory_manager.set_depth_recall_mode(data.depth_recall_mode)
    return {"status": "success", "depth_recall_mode": data.depth_recall_mode}

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
    for msg in context[1:]:
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
        msg_data = {"role": role, "content": clean_content, "images": images}
        if "model" in msg:
            msg_data["model"] = msg["model"]
        if "token_usage" in msg:
            msg_data["token_usage"] = msg["token_usage"]
        history.append(msg_data)
    return history


@router.get("/stats/models")
async def get_models_stats():
    return {"models": stats_service.get_all_models_stats()}


@router.get("/stats/models/{model_id}")
async def get_model_stats_detail(model_id: str):
    stats = stats_service.get_model_stats_detail(model_id)
    if stats is None:
        raise HTTPException(status_code=404, detail=f"模型 {model_id} 无统计数据")
    return stats


@router.get("/stats/roles")
async def get_roles_stats():
    return {"roles": stats_service.get_all_roles_stats()}


@router.get("/stats/roles/{role_id}")
async def get_role_stats_detail(role_id: str):
    stats = stats_service.get_role_stats_detail(role_id)
    if stats is None:
        raise HTTPException(status_code=404, detail=f"角色 {role_id} 无统计数据")
    return stats


@router.get("/stats/usage")
async def get_global_usage_stats():
    """
    获取全局用量统计（所有模型的总输入和总输出 Token）
    """
    return stats_service.get_global_usage_stats()


@router.get("/roles/{role_id}/companion_days")
async def get_companion_days(role_id: str):
    """
    获取角色的陪伴天数
    """
    try:
        session = chat_service.get_session(role_id)
        first_timestamp = session.memory_manager.get_first_conversation_timestamp()
        current_timestamp = datetime.now().timestamp()
        companion_days = (current_timestamp - first_timestamp) // (24 * 3600)
        return {"companion_days": max(0, int(companion_days))}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/roles/{role_id}")
async def delete_role(role_id: str):
    """
    删除指定角色
    """
    try:
        role_info = data_service.role_registry.get_role_by_id(role_id)
        if not role_info:
            raise HTTPException(status_code=404, detail=f"角色 {role_id} 不存在")
        
        role_name = role_info["name"]
        companion_days = 0
        
        try:
            session = chat_service.get_session(role_id)
            first_timestamp = session.memory_manager.get_first_conversation_timestamp()
            current_timestamp = datetime.now().timestamp()
            companion_days = (current_timestamp - first_timestamp) // (24 * 3600)
            companion_days = max(0, int(companion_days))
        except Exception:
            companion_days = 0
        
        if role_id in chat_service.active_sessions:
            try:
                session = chat_service.active_sessions[role_id]
                if hasattr(session, 'memory_manager') and hasattr(session.memory_manager, 'close'):
                    session.memory_manager.close()
            except Exception as e:
                print(f"关闭数据库连接时出错: {e}")
            del chat_service.active_sessions[role_id]
        
        role_dir = os.path.join("data", "roles", role_name)
        if os.path.exists(role_dir):
            max_retries = 10
            retry_delay = 0.3
            for attempt in range(max_retries):
                try:
                    import gc
                    gc.collect()
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                    shutil.rmtree(role_dir, ignore_errors=True)
                    if not os.path.exists(role_dir):
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        import tempfile
                        try:
                            temp_name = f"{role_dir}_to_delete_{int(time.time())}"
                            os.rename(role_dir, temp_name)
                            print(f"重命名文件夹以避免锁定: {role_dir} -> {temp_name}")
                        except Exception as rename_error:
                            print(f"重命名文件夹也失败了: {rename_error}")
                            raise
                    print(f"删除文件夹失败 (尝试 {attempt + 1}/{max_retries}): {e}")
        
        data_service.role_registry.roles = [r for r in data_service.role_registry.roles if r["role_id"] != role_id]
        data_service.role_registry.save()
        
        return {"status": "success", "companion_days": companion_days}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除角色失败: {str(e)}")

@router.get("/roles/{role_id}/conversations")
async def get_conversations(role_id: str):
    try:
        session = chat_service.get_session(role_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"conversations": session.memory_manager.get_conversations()}

@router.post("/roles/{role_id}/conversations")
async def create_conversation(role_id: str, data: ConversationCreateRequest):
    try:
        session = chat_service.get_session(role_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    conv_id = session.memory_manager.create_conversation(data.name)
    conversations = session.memory_manager.get_conversations()
    new_conv = next((conv for conv in conversations if conv["conversation_id"] == conv_id), None)
    return {"status": "success", "conversation_id": conv_id, "name": new_conv["name"] if new_conv else data.name}

@router.get("/roles/{role_id}/conversations/{conv_id}")
async def get_conversation(role_id: str, conv_id: str):
    try:
        session = chat_service.get_session(role_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    conversations = session.memory_manager.get_conversations()
    conv = next((c for c in conversations if c["conversation_id"] == conv_id), None)
    if not conv:
        raise HTTPException(status_code=404, detail=f"对话 {conv_id} 不存在")
    return conv

@router.put("/roles/{role_id}/conversations/{conv_id}")
async def update_conversation(role_id: str, conv_id: str, data: ConversationUpdateRequest):
    try:
        session = chat_service.get_session(role_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    conversations = session.memory_manager.get_conversations()
    conv = next((c for c in conversations if c["conversation_id"] == conv_id), None)
    if not conv:
        raise HTTPException(status_code=404, detail=f"对话 {conv_id} 不存在")
    session.memory_manager.update_conversation_name(conv_id, data.name)
    return {"status": "success"}

@router.delete("/roles/{role_id}/conversations/{conv_id}")
async def delete_conversation(role_id: str, conv_id: str):
    try:
        session = chat_service.get_session(role_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    conversations = session.memory_manager.get_conversations()
    conv = next((c for c in conversations if c["conversation_id"] == conv_id), None)
    if not conv:
        raise HTTPException(status_code=404, detail=f"对话 {conv_id} 不存在")
    session.memory_manager.delete_conversation(conv_id)
    return {"status": "success"}

@router.get("/roles/{role_id}/conversations/{conv_id}/history")
async def get_conversation_history(role_id: str, conv_id: str):
    try:
        session = chat_service.get_session(role_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    conversations = session.memory_manager.get_conversations()
    conv = next((c for c in conversations if c["conversation_id"] == conv_id), None)
    if not conv:
        raise HTTPException(status_code=404, detail=f"对话 {conv_id} 不存在")
    
    current_conv_id = session.memory_manager.current_conversation_id
    session.memory_manager.switch_conversation(conv_id)
    
    context = session.memory_manager.context_buffer
    img_dir = os.path.join(session.memory_manager.base_dir, "images")
    history = []
    for msg in context[1:]:
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
        msg_data = {"role": role, "content": clean_content, "images": images}
        if "model" in msg:
            msg_data["model"] = msg["model"]
        if "token_usage" in msg:
            msg_data["token_usage"] = msg["token_usage"]
        history.append(msg_data)
    
    session.memory_manager.switch_conversation(current_conv_id)
    return history


@router.post("/shutdown")
async def shutdown_service():
    """
    安全退出服务，触发所有会话的关闭操作
    """
    try:
        print("\033[93m[系统] 收到退出请求，开始安全关闭...\033[0m")
        chat_service.shutdown_all()
        print("\033[92m[系统] 所有会话已安全关闭\033[0m")
        print("\033[92m[系统] 服务即将退出...\033[0m")
        
        # 在后台线程中延迟退出，给API返回响应的机会
        # 延迟3秒确保所有关闭操作完成
        import threading
        import time
        def delayed_exit():
            time.sleep(3)
            print("\033[92m[系统] 程序自动退出\033[0m")
            # 使用更可靠的方式退出进程，兼容Windows和Linux
            os._exit(0)
        threading.Thread(target=delayed_exit, daemon=True).start()
        
        return {"status": "success", "message": "服务正在安全关闭"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"退出失败: {str(e)}")

@router.get("/knowledge-bases", response_model=List[KnowledgeBaseListItem])
async def list_knowledge_bases():
    """获取所有知识库摘要列表"""
    try:
        kbs = knowledge_service.list_knowledge_bases()
        return kbs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge-bases", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(req: KnowledgeBaseCreate):
    """创建新知识库"""
    try:
        kb = knowledge_service.create_knowledge_base(
            name=req.name,
            description=req.description or ""
        )
        return kb
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(kb_id: str):
    """获取指定知识库详情"""
    try:
        kb = knowledge_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return kb
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(kb_id: str, req: KnowledgeBaseUpdate):
    """更新知识库元数据"""
    try:
        kb = knowledge_service.update_knowledge_base(
            kb_id=kb_id,
            name=req.name,
            description=req.description
        )
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return kb
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/knowledge-bases/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    """删除知识库（含所有文档）"""
    try:
        success = knowledge_service.delete_knowledge_base(kb_id)
        if not success:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return {"message": "知识库已删除", "kb_id": kb_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge-bases/{kb_id}/documents", response_model=DocumentUploadResponse)
async def upload_document_to_kb(
    kb_id: str,
    file: UploadFile = File(...),
):
    """
    上传文档到指定知识库

    支持格式: .docx, .xlsx, .pdf
    最大文件大小: 50MB
    上传后自动触发文档解析
    """
    allowed_extensions = {'.docx', '.xlsx', '.pdf'}
    filename = file.filename or 'unnamed'
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，仅支持 {', '.join(allowed_extensions)}"
        )

    MAX_FILE_SIZE = 50 * 1024 * 1024
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大：{len(content) / 1024 / 1024:.1f}MB，最大允许 50MB"
        )

    try:
        result = knowledge_service.upload_document(
            kb_id=kb_id,
            original_filename=filename,
            file_content=content,
            file_type=ext.replace('.', '')
        )

        return DocumentUploadResponse(
            doc_id=result['doc_id'],
            kb_id=result['kb_id'],
            filename=result['filename'],
            file_type=result['file_type'],
            file_size=result.get('file_size', len(content)),
            uploaded_at=result['uploaded_at'],
            parsed_status=result.get('parsed_status', 'success'),
            parsed_message=result.get('parsed_message')
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@router.get("/knowledge-bases/{kb_id}/documents", response_model=List[DocumentInfo])
async def list_kb_documents(kb_id: str):
    """获取指定知识库的文档列表"""
    try:
        kb = knowledge_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        return kb.get('documents', [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-bases/{kb_id}/documents/{doc_id}", response_model=DocumentDetailResponse)
async def get_document_detail(kb_id: str, doc_id: str):
    """获取文档详情和解析后的结构化数据"""
    try:
        kb = knowledge_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        doc_info = next((d for d in kb['documents'] if d['doc_id'] == doc_id), None)
        if not doc_info:
            raise HTTPException(status_code=404, detail="文档不存在")

        parsed_data = None
        kb_dir = os.path.join(knowledge_service.base_dir, kb_id)
        parsed_path = os.path.join(kb_dir, 'documents', f'{doc_id}_parsed.json')

        if os.path.exists(parsed_path):
            with open(parsed_path, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)

        return DocumentDetailResponse(**doc_info, parsed_data=parsed_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/knowledge-bases/{kb_id}/documents/{doc_id}")
async def delete_document_from_kb(kb_id: str, doc_id: str):
    """删除知识库中的指定文档（包括原始文件和解析数据）"""
    try:
        success = knowledge_service.delete_document(kb_id, doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="文档或知识库不存在")

        return {"message": "文档已删除", "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))