import os
import sys
import time
import threading
import random
import json

# 将项目根目录加入系统路径，确保能导入其他模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from role_manager import RoleRegistry
from auth_manager import AuthManager
from roleplay_core import RoleplaySession
from llm_adapters.qwen_native_adapter import QwenNativeAdapter


def typewriter(text, delay=(0.01, 0.04)):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(random.uniform(*delay))


class CursorGuardSpinner:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._animate)

    def _animate(self):
        chars = ["|", "/", "-", "\\"]
        idx = 0
        while not self.stop_event.is_set():
            sys.stdout.write(chars[idx % 4])
            sys.stdout.flush()
            time.sleep(0.1)
            if not self.stop_event.is_set():
                sys.stdout.write("\b")
                sys.stdout.flush()
            idx += 1

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.thread.is_alive(): self.thread.join()


def _multiline_input(prompt_message):
    print(prompt_message)
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()


class RoleplayCLIApp:
    def __init__(self):
        self.registry = RoleRegistry()
        self.auth = AuthManager()

        qwen_web_creds = self.auth.credentials.get("qwen_web", {})
        qwen_api_creds = self.auth.credentials.get("qwen_api", {})

        self.qwen_cookie = qwen_web_creds.get("cookie", "")
        env_api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")

        if env_api_key:
            self.qwen_api_key = env_api_key
            print("\033[92m[系统] 检测到系统环境变量，已自动挂载 API Key。\033[0m")
        else:
            self.qwen_api_key = qwen_api_creds.get("api_key", "")

        if not self.qwen_cookie and not self.qwen_api_key:
            print("\n[系统提示] 未检测到 API Key (环境变量与本地文件均未配置)。")
            choice = input("请选择接入模式：\n1. 官方原生 API 调用 (推荐)\n2. 网页端 Cookie 逆向\n>>> ").strip()
            if choice == "1":
                self.qwen_api_key = input("请输入您的 Qwen API Key:\n>>> ").strip()
                self.auth.credentials.setdefault("qwen_api", {})["api_key"] = self.qwen_api_key
                self.auth.save()
            else:
                self.qwen_cookie = input("请输入您的 Qwen Cookie:\n>>> ").strip()
                self.auth.credentials.setdefault("qwen_web", {})["cookie"] = self.qwen_cookie
                self.auth.save()
            print("[系统] 凭证已安全保存至本地。")

    def run(self):
        while True:
            print("\n" + "=" * 15 + " AI 角色扮演终端 " + "=" * 15)
            roles = self.registry.get_all_roles()

            if not roles:
                print("当前没有角色。请先创建一个！")
                self._create_role_ui()
                continue

            for i, r in enumerate(roles):
                role_name = r['name']
                # 从独立的 role_meta 文件夹中解析严格模式标识
                meta_path = os.path.join("data", "roles", role_name, "role_meta.json")
                strict_mode = False
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            strict_mode = meta.get("strict_mode", False)
                    except Exception:
                        pass

                strict_tag = "\033[31m[严格]\033[0m" if strict_mode else "\033[32m[自由]\033[0m"
                print(f"{i + 1}. {strict_tag} {r['name']}")

            print(f"{len(roles) + 1}. [+] 创建新角色")
            print("0. [x] 退出系统")

            choice = input("\n请选择要互动的角色序号: ").strip()

            if choice == "0":
                break
            elif choice == str(len(roles) + 1):
                self._create_role_ui()
            elif choice.isdigit() and 1 <= int(choice) <= len(roles):
                role_data = roles[int(choice) - 1]
                self._enter_chat(role_data)
            else:
                print("[系统] 输入无效。")

    def _create_role_ui(self):
        print("\n--- 创建新角色 ---")
        name = input("请输入角色名称: ").strip() or "未命名角色"

        print("\n\033[33m[系统] 已自动进入多行/长文本输入模式。\033[0m")
        print("       请输入角色的 System Prompt (人设/世界观/对话规则等)。")
        print("       完成后在新起的一行单独输入 \033[31mEND\033[0m 并按回车确认。")
        prompt = _multiline_input(">>> [人设录入区]")

        print("")
        strict = input("是否开启严格模式? (开启后在对话中无法修改人设) [y/N]: ").strip().lower() == 'y'

        print("\n--- 配置大模型 API 参数 (直接按回车使用默认值) ---")

        def get_val(prompt_text, default, cast_type):
            val = input(f"{prompt_text} [默认 {default}]: ").strip()
            if not val: return default
            try:
                return cast_type(val)
            except ValueError:
                return default

        api_settings = {
            "temperature": get_val("温度 (temperature, 决定发散度)", 1.0, float),
            "top_p": get_val("核采样阈值 (top_p)", 0.8, float),
            "top_k": get_val("Top-K (top_k)", 20, int),
            "presence_penalty": get_val("话题惩罚 (presence_penalty)", 0.0, float),
            "repetition_penalty": get_val("重复惩罚 (repetition_penalty)", 1.1, float),
            "enable_think": input("是否默认开启思考引擎? [Y/n]: ").strip().lower() != 'n',
            "display_think": input("是否默认显示思考过程? [Y/n]: ").strip().lower() != 'n'
        }

        # 仅在 registry 注册花名册
        role = self.registry.create_role(name)

        # 传递完整参数，让 MemoryManager 固化生成专属配置
        from vdb_tools.hierarchical_memory_db import HierarchicalMemoryManager
        _db = HierarchicalMemoryManager(role["role_id"], role["name"],
                                        initial_api_settings=api_settings,
                                        system_prompt=prompt,
                                        strict_mode=strict)

        print(f"\n[系统] 成功创建角色【{role['name']}】！专属数据存储卷与配置已生成。")

    def _handle_cli_commands(self, user_input: str, session: RoleplaySession, role_data: dict) -> bool:
        parts = user_input.strip().split(" ", 1)
        base_cmd = parts[0].lower()

        if base_cmd in ["/h", "/help"]:
            print("\n" + "-" * 12 + " 角色交互指令菜单 " + "-" * 12)
            print("  /image <路径>          : 挂载本地图片至下一轮对话")
            print("  /enable_think          : 切换思考模式开关")
            print("  /display_think_process : 切换是否在屏幕上输出思考过程")
            print("  /set_api               : 动态修改大模型 API 请求参数")
            print("  /config                : 显示当前角色配置与状态信息")
            print("  /debug                 : 开发者模式，打印底层原生数据包")
            print("  /paste                 : 进入多行/超长文本安全粘贴模式")
            print("  /role                  : 查看或修改当前角色的 System Prompt")
            print("  /check_mem             : 透视四级记忆引擎存储状态")
            print("  /force_sum             : 强制触发后台时钟与容量引擎巡检")
            print("  /recall <词>           : 强制深层检索冷库查找细节")
            print("  /back                  : 记忆凝固并返回主菜单")
            print("  exit                   : 记忆凝固并彻底退出程序\n" + "-" * 40)
            return True

        elif base_cmd == "/set_api":
            print("\n--- 修改当前角色的 API 参数 (直接回车保持当前值) ---")
            current = session.memory_manager.meta_data.get("settings", {})

            def get_val(prompt_text, current_val, cast_type):
                val = input(f"{prompt_text} (当前 {current_val}): ").strip()
                if not val: return current_val
                try:
                    return cast_type(val)
                except ValueError:
                    print("格式错误，保持原值。")
                    return current_val

            new_settings = {
                "temperature": get_val("温度 (temperature)", current.get("temperature", 1.0), float),
                "top_p": get_val("核采样阈值 (top_p)", current.get("top_p", 0.8), float),
                "top_k": get_val("Top-K (top_k)", current.get("top_k", 20), int),
                "presence_penalty": get_val("话题惩罚 (presence_penalty)", current.get("presence_penalty", 0.0),
                                            float),
                "repetition_penalty": get_val("重复惩罚 (repetition_penalty)",
                                              current.get("repetition_penalty", 1.1),
                                              float)
            }
            session.memory_manager.update_meta_settings(new_settings)
            print("\n\033[32m[系统] API 参数已更新，并持久化到该角色的 role_meta.json 中。\033[0m")
            return True

        elif base_cmd == "/role":
            # 通过底层 DB 的 meta 取出当前设定验证
            current_prompt = session.memory_manager.meta_data.get("system_prompt", "")
            print(f"\n\033[36m[当前角色人设]:\n{current_prompt}\033[0m")

            if session.memory_manager.meta_data.get("strict_mode", False):
                print("\n\033[31m[系统拦截] 严格模式已开启，禁止动态篡改人设！\033[0m")
                return True

            print("\n\033[33m[系统] 已自动进入多行/长文本输入模式。\033[0m")
            print("       请输入新的人设词 (输入 clear 清空，什么都不写直接输入 END 取消修改)。")
            print("       完成后在新起的一行单独输入 \033[31mEND\033[0m 并按回车保存。")
            new_prompt = _multiline_input(">>> [人设编辑区]")

            if not new_prompt:
                print("[系统] 操作取消，人设未更改。")
            else:
                final_prompt = "" if new_prompt.lower() == "clear" else new_prompt
                session.memory_manager.update_meta(system_prompt=final_prompt)
                session.system_prompt = final_prompt
                print("[系统] 角色人设已更新并持久化至独立配置。")
            return True

        elif base_cmd == "/check_mem":
            print("\n" + "=" * 20 + " 四层记忆架构透视仪 " + "=" * 20)
            manager = session.memory_manager

            print("\n\033[36m[层级 0: L1 瞬时工作记忆区 (滑动窗口)]\033[0m")
            buffer = manager.context_buffer
            if not buffer:
                print("  (空)")
            else:
                for item in buffer:
                    r_name = "你" if item["role"] == "user" else role_data["name"]
                    tag = "[\033[31m已打包\033[0m]" if item.get("daily_summarized") else "[\033[32m活跃态\033[0m]"
                    print(f"  {tag} [{r_name}]: {item['content']}")

            print("\n\033[35m[宏观剧情总结区 (ChromaDB 向量簇)]\033[0m")
            for tier, name in [("daily", "L1 日级"), ("weekly", "L2 周级"), ("monthly", "L3 月级"),
                               ("yearly", "L4 年级")]:
                db_data = manager.summary_dbs[tier].get()
                docs = db_data.get("documents", [])
                print(f"  - {name}总结数量: {len(docs)} 篇 (冗余上限: {manager.retention_limits[tier]})")

            print("=" * 60 + "\n")
            return True

        elif base_cmd == "/force_sum":
            if getattr(session.memory_manager, '_compressing', False):
                print("\n\033[33m[系统] 后台已在执行记忆巡检任务中，请稍后。\033[0m")
            else:
                print("\n\033[33m[系统] 强行唤醒后台巡检，立即检查...\033[0m")
                session.memory_manager._compressing = True
                threading.Thread(target=self._force_maintenance_wrapper, args=(session,)).start()
            return True

        return False

    def _force_maintenance_wrapper(self, session):
        try:
            session._maintenance_task()
            print("\n\033[32m[系统异步通知] 强制记忆巡检与凝固已完成。\033[0m")
        except Exception as e:
            print(f"\n\033[31m[系统异步错误] 巡检异常: {e}\033[0m")
        finally:
            session.memory_manager._compressing = False
            session.time_boundary_hit_time = None
            session.capacity_boundary_hit_time = None

    def _enter_chat(self, role_data):
        print(f"\n[系统] 正在连接神经网络，唤醒角色【{role_data['name']}】的多级记忆矩阵...")

        if getattr(self, "qwen_api_key", None):
            adapter = QwenNativeAdapter(api_key=self.qwen_api_key, model="qwen3.5-plus")
        else:
            print("no API key")
            sys.exit(0)

        # 让其内部自动调用 memory_manager 的 system_prompt
        session = RoleplaySession(
            adapter=adapter,
            role_id=role_data["role_id"],
            role_name=role_data["name"]
        )

        print(f"================ 与 {role_data['name']} 的对话已开始 ================")
        print("输入 /h 查看可用指令。")

        pending_images = []
        debug_mode = False

        # 从 MemoryManager 解构已沉淀的思考开关状态
        settings = session.memory_manager.meta_data.get("settings", {})
        enable_think = settings.get("enable_think", True)
        display_think = settings.get("display_think", True)

        while True:
            try:
                user_input = input(f"\n\033[92m[你]:\033[0m ").strip()
                if not user_input: continue

                if user_input.lower() == "/back":
                    if hasattr(session, 'shutdown_and_flush'): session.shutdown_and_flush()
                    print(f"\n[系统] 已安全挂起角色【{role_data['name']}】，返回主菜单。")
                    break
                elif user_input.lower() == "exit":
                    if hasattr(session, 'shutdown_and_flush'): session.shutdown_and_flush()
                    print("\n[系统] 程序安全退出。")
                    sys.exit(0)

                if user_input.lower() == "/paste":
                    print("\n\033[33m[系统] 已进入多行/超长文本安全粘贴模式。\033[0m")
                    user_input = _multiline_input(">>> [防溢出粘贴区]")
                    if not user_input: continue

                if user_input.lower().startswith("/image"):
                    parts = user_input.split(" ", 1)
                    if len(parts) < 2:
                        print("\033[33m[用法错误] 请提供图片路径\033[0m")
                        continue
                    img_path = parts[1].strip()
                    if not os.path.exists(img_path):
                        print(f"\033[31m[系统错误] 找不到物理文件: {img_path}\033[0m")
                        continue
                    if os.path.getsize(img_path) > 4 * 1024 * 1024:
                        print(f"\033[33m[系统拦截] 图片大小超过 4MB 限制，请压缩后重试。\033[0m")
                        continue
                    pending_images.append(img_path)
                    print(f"\033[94m[视觉模块] 图像已挂载 ({len(pending_images)}): {img_path}\033[0m")
                    continue

                if user_input.lower() == "/debug":
                    debug_mode = not debug_mode
                    print(
                        f"\n[系统] 开发者 Debug 模式已{'\033[32m开启\033[0m' if debug_mode else '\033[31m关闭\033[0m'}。")
                    continue

                if user_input.lower() == "/enable_think":
                    enable_think = not enable_think
                    session.memory_manager.update_meta_settings({"enable_think": enable_think})
                    print(
                        f"\n[系统] 思考模式已{'\033[32m开启\033[0m' if enable_think else '\033[31m关闭\033[0m'}，并已永久保存至该角色设定。")
                    continue

                if user_input.lower() == "/display_think_process":
                    display_think = not display_think
                    session.memory_manager.update_meta_settings({"display_think": display_think})
                    print(
                        f"\n[系统] 屏幕思考过程输出已{'\033[32m显示\033[0m' if display_think else '\033[31m隐藏\033[0m'}，并已永久保存至该角色设定。")
                    continue

                if user_input.lower() == "/config":
                    api_settings = session.memory_manager.meta_data.get("settings", {})

                    print("\n" + "=" * 15 + " 当前角色与系统配置 " + "=" * 15)
                    print(f"角色名称: {role_data['name']}")
                    print(f"底层模型: {getattr(session.adapter, 'model', '未知')}")
                    print(f"思考模式引擎: {'开启' if enable_think else '关闭'}")
                    print(f"思考内容显示: {'是' if display_think else '否'}")
                    print(f"Debug状态: {'开启' if debug_mode else '关闭'}")
                    print("\n--- 大模型请求参数 (role_meta.json) ---")
                    print(f"温度 (temperature): {api_settings.get('temperature', 1.0)}")
                    print(f"核采样阈值 (top_p): {api_settings.get('top_p', 0.8)}")
                    print(f"Top-K (top_k): {api_settings.get('top_k', 20)}")
                    print(f"话题惩罚 (presence_penalty): {api_settings.get('presence_penalty', 0.0)}")
                    print(f"重复惩罚 (repetition_penalty): {api_settings.get('repetition_penalty', 1.1)}")
                    print(f"\n[System Prompt]:\n{session.system_prompt}")
                    print("=" * 48 + "\n")
                    continue

                if self._handle_cli_commands(user_input, session, role_data):
                    continue

                force_recall = False
                chat_input = user_input
                if user_input.lower().startswith("/recall"):
                    force_recall = True
                    parts = user_input.split(" ", 1)
                    chat_input = parts[1] if len(parts) > 1 else ""
                    print(f"\033[35m(系统强制切入冷库溯源模式，检索词: {chat_input})\033[0m")

                print(f"\033[96m[{role_data['name']}]:\033[0m ", end="", flush=True)

                generator = session.stream_chat(
                    chat_input,
                    image_paths=pending_images,
                    force_deep_recall=force_recall,
                    debug_mode=debug_mode,
                    enable_thinking=enable_think
                )

                self._render_stream(generator, display_think)
                pending_images.clear()

            except KeyboardInterrupt:
                print("\n[系统] 检测到中断信号，输入 /back 返回菜单，或输入 exit 退出程序。")
            except Exception as e:
                print(f"\n\033[31m[外层循环崩溃] 未捕获的系统异常: {str(e)}\033[0m")

    def _render_stream(self, generator, display_think=True):
        is_waiting = True
        last_status = None
        spinner = CursorGuardSpinner()
        spinner.__enter__()

        try:
            for m_type, content in generator:
                if is_waiting:
                    spinner.stop_event.set()
                    spinner.thread.join()
                    is_waiting = False
                    sys.stdout.write("\b \b\n")
                    sys.stdout.flush()

                if m_type == "raw":
                    if last_status == "thought": print("\033[0m", end="", flush=True)
                    print(f"\n\033[35m[底层流式数据包]\n{content}\033[0m")
                    last_status = "raw"

                elif m_type == "thought":
                    if not display_think:
                        continue
                    if last_status not in ["thought", "raw"]:
                        print("\n\033[35m[think]\033[0m\n", end="", flush=True)
                        last_status = "thought"
                    typewriter(content, delay=(0.001, 0.005))

                elif m_type == "answer":
                    if last_status == "thought":
                        print("\n\n\033[36m[answer]\033[0m\n", end="", flush=True)
                    elif last_status != "answer":
                        print("\n\033[36m[answer]\033[0m\n", end="", flush=True)
                    last_status = "answer"
                    typewriter(content)

                elif m_type == "debug":
                    print(f"\n\033[90m{content}\033[0m")

                elif m_type in ["error", "auth_error"]:
                    if last_status == "thought": print("\033[0m", end="", flush=True)
                    print(f"\n\033[31m[API底层异常]\n{content}\033[0m")
                    last_status = "error"
        finally:
            if is_waiting:
                spinner.stop_event.set()
                spinner.thread.join()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            print("\033[0m", end="", flush=True)
        print("\n")


def run_cli():
    """CLI 终端交互大循环主逻辑"""
    print("初始化系统中...")
    app = RoleplayCLIApp()
    app.run()
