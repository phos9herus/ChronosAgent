# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.websockets import router as websocket_router
from app.api.endpoints import router as api_router
from app.services.chat_service import chat_service

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理器
    """
    print("Web 服务正在启动...")
    yield
    print("Web 服务正在关闭，执行安全清理...")
    # 确保在按 Ctrl+C 关闭服务器时，所有未保存的长期记忆都能被凝固保存
    chat_service.shutdown_all()
    print("资源清理完毕。")

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
    allow_origins=["*"],  # 生产环境中请替换为前端实际的域名或 IP，如 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 前端页面挂载 (Phase 2 新增)
# ==========================================
# 挂载静态资源目录 (CSS, JS, 图片)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

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