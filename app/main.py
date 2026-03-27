# app/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.websockets import router as websocket_router
from app.api.endpoints import router as api_router
from app.services.chat_service import chat_service
from app.config.settings import settings
from app.middleware.error_handler import (
    base_app_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler
)
from app.exceptions import BaseAppException

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from vdb_tools.hierarchical_memory_db import get_embedding_function

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理器
    """
    print("Web 服务正在启动...")
    print(">>> 正在预加载核心嵌入模型与向量空间，请稍候...")
    get_embedding_function() # 【核心触发】：强行唤醒并加载 1.5GB 的大模型
    print(">>> 模型预加载完成，神经引擎已就绪！")
    yield
    print("Web 服务正在关闭，执行安全清理...")

# 初始化 FastAPI 实例
app = FastAPI(
    title="Roleplay System Web API",
    description="基于 FastAPI 与 WebSocket 的多角色对话引擎",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# 注册全局异常处理器
app.add_exception_handler(BaseAppException, base_app_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# ==========================================
# 前端页面挂载 (Phase 2 新增)
# ==========================================
# 挂载静态资源目录 (CSS, JS, 图片)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
os.makedirs("data/avatars", exist_ok=True)
app.mount("/avatars", StaticFiles(directory="data/avatars"), name="avatars")
# 初始化 Jinja2 模板引擎
templates = Jinja2Templates(directory="app/templates")

@app.get("/", tags=["WebUI"])
async def get_web_ui(request: Request):
    """
    返回主 Web UI 界面
    """
    return templates.TemplateResponse(request=request, name="index.html")
# ==========================================

# 挂载路由
app.include_router(websocket_router, tags=["WebSockets"])
app.include_router(api_router, prefix="/api", tags=["API"])

# 提供一个简单的 HTTP 健康检查接口
@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "message": "Roleplay Web API is running smoothly."}