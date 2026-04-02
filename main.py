import argparse
import uvicorn


def start_web_server(host: str = "0.0.0.0", port: int = 8000):
    """启动 FastAPI Web 服务"""
    print(f"正在启动 Web 服务: http://{host}:{port}")
    # 注意：这里的 app.main:app 指向我们新建的 app/main.py 中的 FastAPI 实例
    uvicorn.run("app.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Roleplay System 启动器")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web 服务的端口号 (默认: 8000)"
    )

    args = parser.parse_args()
    start_web_server(port=args.port)
