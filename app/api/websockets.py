# app/api/websockets.py
import json
import asyncio
import os
import base64
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.chat_service import chat_service

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw_data = await websocket.receive_text()
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
            force_deep_recall = data.get("force_deep_recall", False)

            if not role_id:
                await websocket.send_json({"msg_type": "error", "content": "缺失 role_id"})
                continue

            try:
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
                generator = session.stream_chat(
                    user_input=user_input,
                    images=images,
                    enable_think=enable_think,
                    force_deep_recall=force_deep_recall
                )

                # 4. 迭代生成器并推送
                for msg_type, content in generator:
                    await websocket.send_json({
                        "msg_type": msg_type,
                        "content": content,
                        "role_id": role_id
                    })

                # 5. 发送完成信号
                await websocket.send_json({"msg_type": "status", "content": "[DONE]"})

            except Exception as e:
                await websocket.send_json({"msg_type": "error", "content": f"生成异常: {str(e)}"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket 客户端已正常断开")
    except Exception as e:
        manager.disconnect(websocket)
        print(f"WebSocket 运行异常: {e}")