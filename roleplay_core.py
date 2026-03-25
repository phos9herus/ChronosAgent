# roleplay_core.py
import os
import re
import time
import threading
from datetime import datetime
from typing import Generator, Tuple

from llm_adapters.base_adapter import BaseLLMAdapter
from vdb_tools.hierarchical_memory_db import HierarchicalMemoryManager


class RoleplaySession:
    """
    角色扮演核心调度器。
    集成【惰性求值与强刷机制】的延迟压缩调度引擎。
    """

    def __init__(self,
                 adapter: BaseLLMAdapter,
                 role_id: str,
                 role_name: str = "未命名角色"):

        self.adapter = adapter
        self.role_id = role_id
        self.role_name = role_name
        self.memory_manager = HierarchicalMemoryManager(role_id=self.role_id, role_name=self.role_name)


        self.last_interaction_time = time.time()
        self.capacity_boundary_hit_time = None
        self.time_boundary_hit_time = None

        self._stop_monitor = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _check_deep_recall_intent(self, user_input: str) -> bool:
        trigger_keywords = [
            r"你还记得.*吗", r"仔细回忆", r"很久以前", r"上次.*说",
            r"最初的", r"咱们第一次", r"详细想想"
        ]
        for pattern in trigger_keywords:
            if re.search(pattern, user_input):
                return True
        return False

    def stream_chat(self, user_input: str, images: list = None, original_input: str = None,
                    force_deep_recall: bool = False, **kwargs) -> \
            Generator[Tuple[str, str], None, None]:
        """
        核心会话流：处理动态参数提取、记忆检索、消息构建及多模态图片透传
        """
        self.last_interaction_time = time.time()

        # ==========================================
        # 1. 实时读取角色配置参数并注入 kwargs
        # ==========================================
        if hasattr(self, "memory_manager") and hasattr(self.memory_manager, "meta_data"):
            api_settings = self.memory_manager.meta_data.get("settings", {})

            # 定义允许透传给大模型的参数白名单（严格包含 thinking_budget）
            param_whitelist = [
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "repetition_penalty",
                "thinking_budget"
            ]

            # 如果 kwargs 中没有该参数，则使用角色设定中的最新参数覆盖
            for param in param_whitelist:
                if param in api_settings and param not in kwargs:
                    kwargs[param] = api_settings[param]

        # ==========================================
        # 2. 意图识别与记忆检索
        # ==========================================
        is_deep_recall = force_deep_recall or self._check_deep_recall_intent(user_input)
        long_term_memories = []
        if is_deep_recall:
            # 检索深层知识库中的关联记忆
            long_term_memories = self.memory_manager.retrieve_with_time_routing(query=user_input, top_k=5)

        buffer = self.memory_manager.context_buffer

        # ==========================================
        # 3. 动态组装最新的 System Prompt 与上下文
        # ==========================================
        current_sys_prompt = self.memory_manager.meta_data.get("system_prompt", "")
        if not current_sys_prompt:
            current_sys_prompt = f"你现在扮演【{self.role_name}】。请完全沉浸在角色中，保持你的语气、性格和设定，不要暴露自己是AI。"

        # 如果提取到了深层记忆，将其作为外挂知识库通过系统提示词注入
        if long_term_memories:
            memory_text = "\n".join(long_term_memories)
            current_sys_prompt += f"\n\n[系统提示：以下是关于过去的深层回忆片段，请将其作为背景参考，自然地融入对话中：\n{memory_text}]"

        structured_messages = [
            {"role": "system", "content": current_sys_prompt}
        ]

        # 依次推入滑动窗口中的历史对话（短期记忆）
        for msg in buffer:
            role = "user" if msg["role"] == "user" else "assistant"
            structured_messages.append({
                "role": role,
                "content": msg["content"]
            })

        # ==========================================
        # 4. 调用底层适配器生成内容并捕获结果
        # ==========================================
        full_answer = ""
        full_thought = ""

        # 【核心修复 1】：正式将用户的输入写入记忆库，否则 AI 会失去用户的上下文
        if user_input.strip():
            self.memory_manager.add_memory(role="user", text=user_input)

        # 将组装好的 messages 传给底层适配器，kwargs 中已经包含了最新的思考预算等参数
        for msg_type, content in self.adapter.stream_chat(
                prompt=user_input,
                images=images,
                messages=structured_messages,
                **kwargs
        ):
            if msg_type == "answer":
                full_answer += content
            elif msg_type == "thought":
                full_thought += content

            yield msg_type, content

        # ==========================================
        # 5. 会话结束，持久化本次问答到记忆数据库
        # ==========================================
        # 【核心修复 2】：取消缩进！等 for 循环彻底结束（流式接收完毕），再执行一次性存库
        if full_answer:
            self.memory_manager.add_memory(role="assistant", text=full_answer)

            # 提取刚刚存入的 AI 回复记录，将思维过程追加进去
            if full_thought:
                self.memory_manager.context_buffer[-1]["reasoning_content"] = full_thought

            # 手动刷新一次上下文落地
            self.memory_manager.save_context()



    def _monitor_loop(self):
        while not self._stop_monitor:
            time.sleep(10)
            if getattr(self.memory_manager, '_compressing', False):
                continue

            current_time = time.time()
            idle_time = current_time - self.last_interaction_time

            needs_time_comp = self._detect_time_boundary()
            needs_cap_comp = self._detect_capacity_boundary()

            if not needs_time_comp and not needs_cap_comp:
                self.time_boundary_hit_time = None
                self.capacity_boundary_hit_time = None
                continue

            if needs_time_comp and self.time_boundary_hit_time is None:
                self.time_boundary_hit_time = current_time
            if needs_cap_comp and self.capacity_boundary_hit_time is None:
                self.capacity_boundary_hit_time = current_time

            should_execute = False
            if idle_time >= 900:
                should_execute = True
            elif (self.time_boundary_hit_time and current_time - self.time_boundary_hit_time >= 1800) or \
                    (self.capacity_boundary_hit_time and current_time - self.capacity_boundary_hit_time >= 1800):
                should_execute = True

            if should_execute:
                self.memory_manager._compressing = True
                try:
                    self._maintenance_task()
                finally:
                    self.memory_manager._compressing = False
                    self.time_boundary_hit_time = None
                    self.capacity_boundary_hit_time = None

    def shutdown_and_flush(self):
        self._stop_monitor = True

        needs_time_comp = self._detect_time_boundary()
        needs_cap_comp = self._detect_capacity_boundary()

        if needs_time_comp or needs_cap_comp:
            print("\n\033[90m[系统] 检测到未归档的记忆，正在执行退出前的记忆凝固整理，请稍候...\033[0m")
            self.memory_manager._compressing = True
            try:
                self._maintenance_task()
                print("\033[90m[系统] 记忆凝固完成。\033[0m")
            finally:
                self.memory_manager._compressing = False

    def _detect_time_boundary(self) -> bool:
        buffer = self.memory_manager.context_buffer
        if not buffer: return False
        current_date = datetime.now().date()
        for m in buffer:
            if datetime.fromtimestamp(m["timestamp"]).date() < current_date and not m.get("daily_summarized", False):
                return True
        return False

    def _detect_capacity_boundary(self) -> bool:
        buffer = self.memory_manager.context_buffer
        current_len = sum(len(m["content"]) for m in buffer)
        return current_len > self.memory_manager.max_context_length

    def _maintenance_task(self):
        buffer = self.memory_manager.context_buffer
        current_date = datetime.now().date()
        to_compress_time = [m for m in buffer if
                            datetime.fromtimestamp(m["timestamp"]).date() < current_date and not m.get(
                                "daily_summarized", False)]

        if to_compress_time:
            dates = set(datetime.fromtimestamp(m["timestamp"]).date() for m in to_compress_time)
            for d in sorted(dates):
                day_msgs = [m for m in to_compress_time if datetime.fromtimestamp(m["timestamp"]).date() == d]
                self._execute_summarization(day_msgs, tier="daily")
                for m in day_msgs:
                    m["daily_summarized"] = True
            self.memory_manager.save_context()

        buffer = self.memory_manager.context_buffer
        current_len = sum(len(m["content"]) for m in buffer)

        if current_len > self.memory_manager.max_context_length:
            split_idx = max(1, int(len(buffer) * 0.6))
            evicted_msgs = buffer[:split_idx]

            unsummarized_msgs = [m for m in evicted_msgs if not m.get("daily_summarized", False)]
            if unsummarized_msgs:
                self._execute_summarization(unsummarized_msgs, tier="daily")

            new_buffer = buffer[split_idx:]
            self.memory_manager.context_buffer = new_buffer
            self.memory_manager.save_context()

    def _execute_summarization(self, messages: list, tier: str):
        history_text = "\n".join([f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in messages])
        if len(history_text) < 500:
            prompt = f"请将以下发生在两位角色之间的历史对话进行总结。因为原文篇幅极短，请你【巨细无遗】地保留所有对话细节、动作描写和情感微变化，切勿过度精简：\n\n{history_text}"
        else:
            prompt = f"请以旁观者的客观视角，高度概括以下发生在两位角色之间的长篇历史对话。因为原文篇幅较长，请你【优先提取核心剧情骨架、重大事件和关键情感转折】：\n\n{history_text}"

        summary_result = ""
        system_p = "你是一个无情的记忆压缩机器，专注于提取客观的剧情摘要，不回答任何问题。"

        # 核心改造：显式构造总结任务专用的 messages 数组
        summary_messages = [
            {"role": "system", "content": system_p}
        ]

        # 显式传递 messages 数组给适配器
        for msg_type, content in self.adapter.stream_chat(prompt=prompt, messages=summary_messages):
            if msg_type in ["answer", "thought"]:
                summary_result += content

        if summary_result.strip():
            target_timestamp = messages[-1]["timestamp"]
            self.memory_manager.save_summary(tier, summary_result.strip(), target_timestamp)
