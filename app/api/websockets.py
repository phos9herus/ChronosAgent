# app/api/websockets.py
import json
import asyncio
import os
import base64
import uuid
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.chat_service import chat_service
from app.utils.async_notifier import set_event_loop

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """向所有活跃连接广播消息"""
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()


async def send_summarizing_status(is_summarizing: bool, role_id: str = None):
    """发送总结状态消息"""
    msg_type = "summarizing" if is_summarizing else "summarizing_done"
    message = {
        "msg_type": msg_type,
        "content": "",
        "role_id": role_id
    }
    await manager.broadcast(message)


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                # 优化：将接收超时从 300 秒降低到 60 秒，避免长时间无响应
                raw_data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"msg_type": "error", "content": "连接超时，请刷新页面重试"})
                break

            try:
                data = json.loads(raw_data)
            except Exception as e:
                await websocket.send_json({"msg_type": "error", "content": f"JSON解析失败: {e}"})
                continue

            # 1. 提取关键参数
            role_id = data.get("role_id")
            user_input = data.get("user_input", "")
            images = data.get("images", [])
            enable_think = data.get("enable_think", False)
            enable_search = data.get("enable_search", False)
            force_deep_recall = data.get("force_deep_recall", False)
            model = data.get("model", None)  # 新增：模型参数
            conversation_id = data.get("conversation_id", None)  # 新增：对话ID参数
            depth_recall_mode = data.get("depth_recall_mode", None)  # 新增：深度回忆模式参数

            if not role_id:
                await websocket.send_json({"msg_type": "error", "content": "缺失 role_id"})
                continue

            try:
                # 记录请求开始时间
                request_start_time = time.time()

                
                # 2. 获取会话
                session = chat_service.get_session(role_id)

                # ==========================================
                # 核心新增：图片落盘与占位符植入逻辑
                # ==========================================
                if images:
                    # 确保角色的图片资源文件夹存在
                    img_dir = os.path.join(session.memory_manager.base_dir, "images")
                    os.makedirs(img_dir, exist_ok=True)

                    for img_b64 in images:
                        try:
                            # 拆分 data:image/jpeg;base64, 和实际的 base64 字符串
                            header, encoded = img_b64.split(",", 1)
                            ext = "png" if "image/png" in header else "jpg"
                            filename = f"img_{uuid.uuid4().hex[:8]}.{ext}"
                            filepath = os.path.join(img_dir, filename)

                            # 落盘保存
                            with open(filepath, "wb") as f:
                                f.write(base64.b64decode(encoded))

                            # 在 user_input 末尾隐式追加占位符
                            # 这将伴随文本进入 memory_manager，成为永久上下文标记
                            user_input += f"\n[IMAGE: {filename}]"
                        except Exception as img_err:
                            print(f"图片保存失败: {img_err}")

                # 3. 开始流式生成
                # 注意：原生 images 依然传给 adapter，因为 Qwen API 视觉模型需要真的 base64

                # 发送总结开始消息（示例）
                await websocket.send_json({
                    "msg_type": "summarizing",
                    "content": "",
                    "role_id": role_id,
                    "conversation_id": conversation_id
                })

                generator = session.stream_chat(
                    user_input=user_input,
                    images=images,
                    enable_think=enable_think,
                    enable_search=enable_search,
                    force_deep_recall=force_deep_recall,
                    model=model,  # 新增：传递模型参数
                    conversation_id=conversation_id,  # 新增：传递对话ID参数
                    depth_recall_mode=depth_recall_mode  # 新增：传递深度回忆模式参数
                )

                # 4. 迭代生成器并推送
                message_count = 0

                for msg_type, content in generator:
                    message_count += 1

                    try:
                        await asyncio.wait_for(
                            websocket.send_json({
                                "msg_type": msg_type,
                                "content": content,
                                "role_id": role_id,
                                "conversation_id": conversation_id or session.memory_manager.current_conversation_id
                            }),
                            timeout=10.0
                        )

                    except asyncio.TimeoutError:

                        await websocket.send_json({"msg_type": "error", "content": "发送超时"})
                        break

                # 5. 发送完成信号
                request_duration = time.time() - request_start_time
                
                # 发送总结完成消息
                await websocket.send_json({
                    "msg_type": "summarizing_done",
                    "content": "",
                    "role_id": role_id,
                    "conversation_id": conversation_id
                })
                
                await websocket.send_json({"msg_type": "status", "content": "[DONE]"})

            except Exception as e:
                request_duration = time.time() - request_start_time if 'request_start_time' in locals() else 0
                import traceback
                traceback.print_exc()
                await websocket.send_json({"msg_type": "error", "content": f"生成异常：{str(e)}"})

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        await manager.disconnect(websocket)