import argparse
import uvicorn
import sys


def start_web_server(host: str = "0.0.0.0", port: int = 8000):
    """启动 FastAPI Web 服务"""
    print(f"正在启动 Web 服务: http://{host}:{port}")
    # 注意：这里的 app.main:app 指向我们新建的 app/main.py 中的 FastAPI 实例
    uvicorn.run("app.main:app", host=host, port=port, reload=True)


def start_cli_mode():
    """启动原有的终端交互模式"""
    print("正在启动 CLI 终端调试模式...")
    try:
        # 将原先 main.py 里的终端交互逻辑封装到 cli.roleplay_cli 模块中调用
        from cli.roleplay_cli import run_cli
        run_cli()
    except ImportError as e:
        print(f"启动 CLI 模式失败，请确保已将原有终端逻辑迁移至 cli/roleplay_cli.py。详细错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Roleplay System 启动器")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="以纯终端命令行模式启动 (用于调试)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web 服务的端口号 (默认: 8000)"
    )

    args = parser.parse_args()

    if args.cli:
        start_cli_mode()
    else:
        start_web_server(port=args.port)