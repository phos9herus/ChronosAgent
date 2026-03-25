import os
import time
import json
import uuid
import threading
from datetime import datetime
from typing import List, Dict, Any
import chromadb

from chromadb.utils import embedding_functions

# 全局单例嵌入模型缓存
_GLOBAL_EMBEDDING_FN = None

def get_embedding_function():
    global _GLOBAL_EMBEDDING_FN
    if _GLOBAL_EMBEDDING_FN is None:
        print(">>> 正在初始化全局嵌入模型 BAAI/bge-large-zh-v1.5 ...")
        _GLOBAL_EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-zh-v1.5"
        )
        print(">>> 嵌入模型加载完成！")
    return _GLOBAL_EMBEDDING_FN

# 定义 HF 缓存路径防 Windows 权限报错
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "hf_models")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR


class HierarchicalMemoryManager:
    """
    四级时间退化 + 滚动遗忘 + 冷库自动分表(Sharding) 的角色专属记忆引擎。
    带有 role_meta.json 物理硬映射机制，保证角色记忆文件夹 100% 可移植。
    """

    def __init__(self, role_id: str, role_name: str, max_context_length: int = 6000,
                 initial_api_settings: dict = None, system_prompt: str = "", strict_mode: bool = False):
        self.role_name = role_name
        self.max_context_length = max_context_length

        # 使用中文名作为文件夹，保证极高可读性
        self.base_dir = os.path.join(PROJECT_ROOT, "data", "roles", self.role_name)

        self.paths = {
            "meta": os.path.join(self.base_dir, "role_meta.json"),
            "context": os.path.join(self.base_dir, "current_context.json"),
            "raw_base": os.path.join(self.base_dir, "raw_records"),
            "daily": os.path.join(self.base_dir, "summary_L1_daily"),
            "weekly": os.path.join(self.base_dir, "summary_L2_weekly"),
            "monthly": os.path.join(self.base_dir, "summary_L3_monthly"),
            "yearly": os.path.join(self.base_dir, "summary_L4_yearly"),
        }

        # 创建目录
        for k, path in self.paths.items():
            if k not in ["context", "meta"]:
                os.makedirs(path, exist_ok=True)

        # ==========================================
        # 本地硬映射与元数据加载/初始化
        # ==========================================
        if os.path.exists(self.paths["meta"]):
            with open(self.paths["meta"], "r", encoding="utf-8") as f:
                self.meta_data = json.load(f)
                self.role_id = self.meta_data.get("role_id", role_id)
        else:
            self.role_id = role_id

            # 默认的基础参数 (包含思考开关下放)
            default_settings = {
                "temperature": 1.0,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.1,
                "enable_think": True,
                "display_think": True,
                "voice_id": ""
            }

            if initial_api_settings:
                default_settings.update(initial_api_settings)

            self.meta_data = {
                "role_id": self.role_id,
                "role_name": self.role_name,
                "system_prompt": system_prompt,
                "strict_mode": strict_mode,
                "description": "角色内部硬映射与独立配置文件",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "settings": default_settings
            }
            with open(self.paths["meta"], "w", encoding="utf-8") as f:
                json.dump(self.meta_data, f, ensure_ascii=False, indent=4)

        # ==========================================
        # 挂载底层 ChromaDB 数据库
        # ==========================================
        self.embedding_fn = get_embedding_function()

        self.summary_dbs = {}
        for tier in ["daily", "weekly", "monthly", "yearly"]:
            client = chromadb.PersistentClient(path=self.paths[tier])
            self.summary_dbs[tier] = client.get_or_create_collection(
                name=f"{self.role_id}_{tier}".lower(),
                embedding_function=self.embedding_fn
            )

        self.raw_clients = {}

        self.retention_limits = {
            "daily": 14,
            "weekly": 8,
            "monthly": 12,
            "yearly": 100
        }

        self.context_buffer = self._load_context()
        self._compressing = False

    def update_meta(self, **kwargs):
        """动态更新 meta_data 根级别的属性 (如 system_prompt) 并持久化"""
        self.meta_data.update(kwargs)
        with open(self.paths["meta"], "w", encoding="utf-8") as f:
            json.dump(self.meta_data, f, ensure_ascii=False, indent=4)

    # 【修复】重构此方法，严格遵循同级拓扑契约
    def update_meta_settings(self, payload: dict):
        """动态更新角色 API 参数及核心系统设定并持久化"""

        # 1. 如果 payload 中包含 system_prompt，必须将其提升到根节点更新
        if "system_prompt" in payload:
            self.meta_data["system_prompt"] = payload["system_prompt"]

        # 2. 如果 payload 中包含 settings 嵌套域，则更新 settings 域
        if "settings" in payload:
            if "settings" not in self.meta_data:
                self.meta_data["settings"] = {}
            self.meta_data["settings"].update(payload["settings"])
        # 兼容旧逻辑：如果 payload 是平铺的参数字典
        else:
            settings_payload = {k: v for k, v in payload.items() if k != "system_prompt"}
            if settings_payload:
                if "settings" not in self.meta_data:
                    self.meta_data["settings"] = {}
                self.meta_data["settings"].update(settings_payload)

        # 3. 持久化到磁盘
        with open(self.paths["meta"], "w", encoding="utf-8") as f:
            json.dump(self.meta_data, f, ensure_ascii=False, indent=4)

    # ==========================================
    # 模块 A: 上下文读写
    # ==========================================
    def _load_context(self) -> list:
        if os.path.exists(self.paths["context"]):
            with open(self.paths["context"], "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_context(self):
        with open(self.paths["context"], "w", encoding="utf-8") as f:
            json.dump(self.context_buffer, f, ensure_ascii=False, indent=4)

    # ==========================================
    # 模块 B: Raw冷库半年期自动分表 (Sharding)
    # ==========================================
    def _get_raw_db_folder_name(self, timestamp: float) -> str:
        dt = datetime.fromtimestamp(timestamp)
        half_year_tag = "H1_Jan_to_Jun" if dt.month <= 6 else "H2_Jul_to_Dec"
        return f"{dt.year}_{half_year_tag}"

    def get_raw_collection(self, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        folder_name = self._get_raw_db_folder_name(timestamp)
        db_path = os.path.join(self.paths["raw_base"], folder_name)

        if folder_name not in self.raw_clients:
            os.makedirs(db_path, exist_ok=True)
            client = chromadb.PersistentClient(path=db_path)
            self.raw_clients[folder_name] = client.get_or_create_collection(
                name=f"{self.role_id}_raw_{folder_name}".lower(),
                embedding_function=self.embedding_fn
            )
        return self.raw_clients[folder_name]

    # ==========================================
    # 模块 C: 滚动遗忘算法 (Pruning)
    # ==========================================
    def _enforce_retention_policy(self, tier: str):
        limit = self.retention_limits.get(tier)
        if not limit: return

        collection = self.summary_dbs[tier]
        all_data = collection.get()
        ids = all_data.get("ids", [])
        metas = all_data.get("metadatas", [])

        if len(ids) > limit:
            records = list(zip(ids, [m["timestamp"] for m in metas]))
            records.sort(key=lambda x: x[1])

            overflow_count = len(ids) - limit
            ids_to_delete = [r[0] for r in records[:overflow_count]]
            collection.delete(ids=ids_to_delete)

    # ==========================================
    # 模块 D: 动态时间切片路由检索
    # ==========================================
    def retrieve_with_time_routing(self, query: str, top_k: int = 3) -> List[str]:
        current_time = time.time()
        results = []

        day_14 = 14 * 24 * 3600
        week_8 = 8 * 7 * 24 * 3600
        month_12 = 365 * 24 * 3600

        res_daily = self.summary_dbs["daily"].query(query_texts=[query], n_results=top_k)
        if res_daily.get("documents") and res_daily["documents"][0]:
            for doc, meta in zip(res_daily["documents"][0], res_daily["metadatas"][0]):
                if current_time - meta["timestamp"] <= day_14:
                    results.append(f"[近期回忆]: {doc}")

        res_weekly = self.summary_dbs["weekly"].query(query_texts=[query], n_results=top_k)
        if res_weekly.get("documents") and res_weekly["documents"][0]:
            for doc, meta in zip(res_weekly["documents"][0], res_weekly["metadatas"][0]):
                if day_14 < (current_time - meta["timestamp"]) <= week_8:
                    results.append(f"[数周前回忆]: {doc}")

        res_monthly = self.summary_dbs["monthly"].query(query_texts=[query], n_results=top_k)
        if res_monthly.get("documents") and res_monthly["documents"][0]:
            for doc, meta in zip(res_monthly["documents"][0], res_monthly["metadatas"][0]):
                if week_8 < (current_time - meta["timestamp"]) <= month_12:
                    results.append(f"[数月前回忆]: {doc}")

        res_yearly = self.summary_dbs["yearly"].query(query_texts=[query], n_results=top_k)
        if res_yearly.get("documents") and res_yearly["documents"][0]:
            for doc, meta in zip(res_yearly["documents"][0], res_yearly["metadatas"][0]):
                if (current_time - meta["timestamp"]) > month_12:
                    results.append(f"[深层年份回忆]: {doc}")

        return results

    # ==========================================
    # 数据写入主接口
    # ==========================================
    def add_memory(self, role: str, text: str):
        if not text.strip(): return
        current_time = time.time()
        msg_id = f"msg_{uuid.uuid4().hex[:8]}"

        self.context_buffer.append({
            "role": role, "content": text, "timestamp": current_time, "msg_id": msg_id
        })
        self.save_context()

        raw_col = self.get_raw_collection(current_time)
        raw_col.add(
            documents=[text],
            metadatas=[{"role": role, "timestamp": current_time, "msg_id": msg_id}],
            ids=[msg_id]
        )

    def save_summary(self, tier: str, summary_content: str, source_timestamp: float):
        sum_id = f"sum_{tier}_{uuid.uuid4().hex[:8]}"
        collection = self.summary_dbs[tier]

        collection.add(
            documents=[summary_content],
            metadatas=[{
                "timestamp": source_timestamp,
                "summary_tier": tier
            }],
            ids=[sum_id]
        )
        self._enforce_retention_policy(tier)