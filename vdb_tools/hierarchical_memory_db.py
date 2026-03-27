import os
import sys
import time
import json
import uuid
import threading
import copy
import logging
import warnings
from datetime import datetime
from typing import List, Dict, Any
import chromadb
from contextlib import contextmanager

from chromadb.utils import embedding_functions
from llm_adapters.qwen_native_adapter import QwenNativeAdapter

logger = logging.getLogger(__name__)

SUMMARY_PROMPT_DAILY = """请将以下对话内容进行日级别总结。要求：
1. 突出重要事件和情感变化
2. 保留关键细节和人物关系
3. 保持自然流畅的语言风格
4. 字数控制在200-400字之间

对话内容：
{texts}
"""

SUMMARY_PROMPT_WEEKLY = """请将以下日总结内容进行周级别总结。要求：
1. 提炼本周核心主题和重要发展
2. 保留情感脉络和关键转折点
3. 保持连贯性和逻辑性
4. 字数控制在300-500字之间

日总结内容：
{texts}
"""

SUMMARY_PROMPT_MONTHLY = """请将以下周总结内容进行月级别总结。要求：
1. 提炼本月核心事件和成长变化
2. 保留长期记忆和重要关系发展
3. 保持历史感和深度
4. 字数控制在400-600字之间

周总结内容：
{texts}
"""

SUMMARY_PROMPT_YEARLY = """请将以下月总结内容进行年级别总结。要求：
1. 提炼本年核心主题和重大变化
2. 保留人生重要节点和长期发展
3. 保持历史深度和反思视角
4. 字数控制在500-800字之间

月总结内容：
{texts}
"""

_GLOBAL_EMBEDDING_FN = None
_GLOBAL_DB_LOCK = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

def _get_hf_cache_dir():
    """从环境变量或配置文件获取 Hugging Face 缓存目录"""
    try:
        from app.config.settings import settings
        cache_dir = getattr(settings, 'HF_CACHE_DIR', None)
        if cache_dir:
            if os.path.isabs(cache_dir):
                return cache_dir
            else:
                return os.path.join(PROJECT_ROOT, cache_dir)
    except Exception:
        pass
    
    default_cache_dir = os.path.join(PROJECT_ROOT, "data", "hf_models")
    os.makedirs(default_cache_dir, exist_ok=True)
    return default_cache_dir

HF_CACHE_DIR = _get_hf_cache_dir()
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

# 抑制 ChromaDB 的 "Could not reconstruct embedding function" 警告
warnings.filterwarnings("ignore", message="Could not reconstruct embedding function")


def _check_local_model_exists(model_name: str) -> bool:
    """
    检查本地是否已存在指定的模型文件
    
    Hugging Face 缓存目录结构：
    HF_CACHE_DIR/
      hub/
        models--{model_name}/
          snapshots/
            {commit_hash}/
              {model_files}
    
    Args:
        model_name: 模型名称，如 "BAAI/bge-large-zh-v1.5"
    
    Returns:
        bool: 模型是否存在
    """
    hub_dir = os.path.join(HF_CACHE_DIR, "hub")
    model_path = os.path.join(hub_dir, f"models--{model_name.replace('/', '--')}")
    
    if not os.path.exists(model_path):
        return False
    
    required_files = [
        "config.json",
        "sentence_bert_config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "vocab.txt"
    ]
    
    snapshots_dir = os.path.join(model_path, "snapshots")
    if not os.path.exists(snapshots_dir):
        return False
    
    for snapshot in os.listdir(snapshots_dir):
        snapshot_path = os.path.join(snapshots_dir, snapshot)
        if os.path.isdir(snapshot_path):
            all_files_exist = all(
                os.path.exists(os.path.join(snapshot_path, f))
                for f in required_files
            )
            if all_files_exist:
                return True
    
    return False


def _get_local_model_path(model_name: str) -> str:
    """
    获取本地模型快照路径
    
    Args:
        model_name: 模型名称
    
    Returns:
        str: 本地模型快照路径，如果不存在则返回 None
    """
    hub_dir = os.path.join(HF_CACHE_DIR, "hub")
    model_path = os.path.join(hub_dir, f"models--{model_name.replace('/', '--')}")
    
    if not os.path.exists(model_path):
        return None
    
    snapshots_dir = os.path.join(model_path, "snapshots")
    if not os.path.exists(snapshots_dir):
        return None
    
    # 获取最新的快照（按修改时间排序）
    snapshots = sorted(
        [s for s in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, s))],
        key=lambda x: os.path.getmtime(os.path.join(snapshots_dir, x)),
        reverse=True
    )
    
    if snapshots:
        return os.path.join(snapshots_dir, snapshots[0])
    
    return None


def get_embedding_function():
    """
    获取全局嵌入模型函数
    
    实现逻辑：
    1. 首先检查本地是否已存在模型文件
    2. 若存在，直接使用本地路径加载模型，完全避免网络请求
    3. 若不存在，尝试从 Hugging Face 下载
    4. 若网络不可用，抛出友好错误提示而非崩溃
    
    Returns:
        embedding_functions.SentenceTransformerEmbeddingFunction: 嵌入模型函数
    """
    global _GLOBAL_EMBEDDING_FN
    
    model_name = "BAAI/bge-large-zh-v1.5"
    
    with _GLOBAL_DB_LOCK:
        if _GLOBAL_EMBEDDING_FN is None:
            logger.info(">>> 正在初始化全局嵌入模型 BAAI/bge-large-zh-v1.5 ...")
            
            local_model_path = _get_local_model_path(model_name)
            
            if local_model_path:
                logger.info(f">>> 检测到本地已存在模型文件，直接加载本地模型")
                logger.debug(f">>> 模型路径：{local_model_path}")
                try:
                    _GLOBAL_EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=local_model_path,
                        cache_folder=HF_CACHE_DIR,
                        trust_remote_code=True
                    )
                    logger.info(">>> 本地模型加载完成！")
                except Exception as e:
                    logger.warning(f">>> 本地模型加载失败：{str(e)}")
                    logger.info(">>> 尝试使用模型名称重新加载...")
                    _download_model_with_fallback(model_name)
            else:
                logger.info(f">>> 本地未找到模型文件，开始下载模型到：{HF_CACHE_DIR}")
                _download_model_with_fallback(model_name)
            
            logger.info(">>> 嵌入模型加载完成！")
    
    return _GLOBAL_EMBEDDING_FN


def _download_model_with_fallback(model_name: str):
    """
    下载模型并实现优雅的错误处理
    
    Args:
        model_name: 模型名称
    """
    global _GLOBAL_EMBEDDING_FN
    
    try:
        _GLOBAL_EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            cache_folder=HF_CACHE_DIR
        )
        logger.info(f">>> 模型下载/加载成功！")
    except Exception as e:
        error_msg = str(e)
        
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            logger.error("="*70)
            logger.error(">>>警告：无法连接到 Hugging Face 服务器")
            logger.error("="*70)
            logger.error(f">>> 错误详情：{error_msg}")
            logger.error(">>> 可能的原因:")
            logger.error("    1. 网络连接不稳定或无法访问 Hugging Face")
            logger.error("    2. 本地模型文件不存在且无法下载")
            logger.error(">>> 建议解决方案:")
            logger.error(f"    1. 检查网络连接状态")
            logger.error(f"    2. 手动下载模型到：{HF_CACHE_DIR}")
            logger.error(f"    3. 使用镜像源：设置环境变量 HF_ENDPOINT=https://hf-mirror.com")
            logger.error("="*70)
            
            logger.info(">>> 尝试使用离线模式加载...")
            try:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                _GLOBAL_EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name,
                    cache_folder=HF_CACHE_DIR
                )
                logger.info(">>> 离线模式加载成功！")
                return
            except Exception:
                pass
            
            logger.error("="*70)
            logger.error(">>>错误：无法加载嵌入模型")
            logger.error("="*70)
            logger.error(">>> 程序将无法进行向量检索功能，但基础对话功能仍可使用")
            logger.error(">>> 请按照上述建议修复后重新启动服务")
            logger.error("="*70)
            
            sys.exit(1)
        else:
            logger.error(f">>>错误：模型加载失败 - {error_msg}")
            sys.exit(1)


class HierarchicalMemoryManager:
    def __init__(self, role_id: str, role_name: str, max_context_length: int = 6000,
                 initial_api_settings: dict = None, system_prompt: str = "", strict_mode: bool = False):
        self.role_name = role_name
        self.max_context_length = max_context_length
        self.base_dir = os.path.join(PROJECT_ROOT, "data", "roles", self.role_name)

        self.paths = {
            "meta": os.path.join(self.base_dir, "role_meta.json"),
            "conversations": os.path.join(self.base_dir, "conversations.json"),
            "conversations_dir": os.path.join(self.base_dir, "conversations"),
            "old_context": os.path.join(self.base_dir, "current_context.json"),
            "raw_base": os.path.join(self.base_dir, "raw_records"),
            "daily": os.path.join(self.base_dir, "summary_L1_daily"),
            "weekly": os.path.join(self.base_dir, "summary_L2_weekly"),
            "monthly": os.path.join(self.base_dir, "summary_L3_monthly"),
            "yearly": os.path.join(self.base_dir, "summary_L4_yearly"),
        }

        for k, path in self.paths.items():
            if k not in ["conversations", "old_context", "meta"]:
                os.makedirs(path, exist_ok=True)

        if os.path.exists(self.paths["meta"]):
            with open(self.paths["meta"], "r", encoding="utf-8") as f:
                self.meta_data = json.load(f)
                self.role_id = self.meta_data.get("role_id", role_id)
        else:
            self.role_id = role_id
            default_settings = {
                "temperature": 1.0,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.1,
                "enable_think": True,
                "display_think": True,
                "voice_id": "",
                "depth_recall_mode": "off"
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

        self.embedding_fn = get_embedding_function()
        self.summary_dbs = {}
        for tier in ["daily", "weekly", "monthly", "yearly"]:
            client = chromadb.PersistentClient(path=self.paths[tier])
            self.summary_dbs[tier] = client.get_or_create_collection(
                name=f"{self.role_id}_{tier}".lower(),
                embedding_function=self.embedding_fn
            )
            # 显式重新设置 embedding_function，确保使用当前路径
            self.summary_dbs[tier]._embedding_function = self.embedding_fn

        self.raw_clients = {}
        self.retention_limits = { "daily": 14, "weekly": 8, "monthly": 12, "yearly": 100 }
        self.time_limits = { 
            "daily": 14 * 24 * 3600, 
            "weekly": 8 * 7 * 24 * 3600, 
            "monthly": 12 * 30 * 24 * 3600, 
            "yearly": 100 * 365 * 24 * 3600 
        }
        
        self.current_conversation_id = None
        self._ensure_conversations()
        self.context_buffer = self._load_context()
        self._buffer_lock = threading.Lock()
        self._db_lock = threading.Lock()
        
        self.depth_recall_mode = self.meta_data.get("settings", {}).get("depth_recall_mode", "off")
        
        self.llm_adapter = None
        try:
            api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
            if not api_key:
                try:
                    from app.services.data_service import data_service
                    api_key = data_service.auth_manager.credentials.get("qwen_api", {}).get("api_key", "")
                except Exception:
                    pass
            
            if api_key:
                self.llm_adapter = QwenNativeAdapter(api_key=api_key, model="qwen3.5-flash")
                logger.info("[MemoryDB] LLM 适配器初始化成功")
            else:
                logger.warning("[MemoryDB] 未找到 API Key，将使用回退算法生成总结")
        except Exception as e:
            logger.warning(f"[MemoryDB] LLM 适配器初始化失败: {str(e)}，将使用回退算法生成总结")

    def update_meta(self, **kwargs):
        self.meta_data.update(kwargs)
        with open(self.paths["meta"], "w", encoding="utf-8") as f:
            json.dump(self.meta_data, f, ensure_ascii=False, indent=4)

    def get_depth_recall_mode(self) -> str:
        return self.depth_recall_mode

    def set_depth_recall_mode(self, mode: str) -> None:
        valid_modes = ["off", "normal", "enhanced"]
        if mode not in valid_modes:
            logger.warning(f"[MemoryDB] 无效的深度回忆模式: {mode}，使用默认值 'off'")
            mode = "off"
        self.depth_recall_mode = mode
        if "settings" not in self.meta_data:
            self.meta_data["settings"] = {}
        self.meta_data["settings"]["depth_recall_mode"] = mode
        with open(self.paths["meta"], "w", encoding="utf-8") as f:
            json.dump(self.meta_data, f, ensure_ascii=False, indent=4)

    def update_meta_settings(self, payload: dict):
        """动态更新角色 API 参数及核心系统设定并持久化"""
        # 【核心修复 1】：深拷贝阻断字典突变，防止上游传入的字典被 pop 破坏
        payload_copy = copy.deepcopy(payload)

        # 【核心修复 2】：一次性提取所有根节点属性
        root_keys = ["system_prompt", "display_name", "avatar_mode", "avatar_circle", "avatar_bg"]
        for key in root_keys:
            if key in payload_copy:
                self.meta_data[key] = payload_copy.pop(key)

        # 2. 如果 payload 中包含 settings 嵌套域，则更新 settings 域
        if "settings" in payload_copy:
            if "settings" not in self.meta_data:
                self.meta_data["settings"] = {}
            self.meta_data["settings"].update(payload_copy["settings"])
        # 兼容旧逻辑：如果 payload 是平铺的参数字典
        else:
            if payload_copy:
                if "settings" not in self.meta_data:
                    self.meta_data["settings"] = {}
                self.meta_data["settings"].update(payload_copy)

        # 3. 持久化到磁盘
        with open(self.paths["meta"], "w", encoding="utf-8") as f:
            json.dump(self.meta_data, f, ensure_ascii=False, indent=4)

    def _ensure_conversations(self):
        os.makedirs(self.paths["conversations_dir"], exist_ok=True)
        
        if os.path.exists(self.paths["conversations"]):
            with open(self.paths["conversations"], "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations = data.get("conversations", [])
        else:
            conversations = []
        
        if os.path.exists(self.paths["old_context"]) and not conversations:
            conv_id = f"conv_{uuid.uuid4().hex[:8]}"
            created_at = time.strftime("%Y-%m-%d %H:%M:%S")
            
            conv_dir = os.path.join(self.paths["conversations_dir"], conv_id)
            os.makedirs(conv_dir, exist_ok=True)
            
            old_context_path = self.paths["old_context"]
            new_context_path = os.path.join(conv_dir, "context.json")
            if os.path.exists(old_context_path):
                import shutil
                shutil.move(old_context_path, new_context_path)
            
            conversations = [{
                "conversation_id": conv_id,
                "name": "对话1",
                "created_at": created_at,
                "last_updated": created_at
            }]
            
            with open(self.paths["conversations"], "w", encoding="utf-8") as f:
                json.dump({"conversations": conversations}, f, ensure_ascii=False, indent=4)
        
        if not conversations:
            conv_id = f"conv_{uuid.uuid4().hex[:8]}"
            created_at = time.strftime("%Y-%m-%d %H:%M:%S")
            
            conv_dir = os.path.join(self.paths["conversations_dir"], conv_id)
            os.makedirs(conv_dir, exist_ok=True)
            
            conversations = [{
                "conversation_id": conv_id,
                "name": "对话1",
                "created_at": created_at,
                "last_updated": created_at
            }]
            
            with open(self.paths["conversations"], "w", encoding="utf-8") as f:
                json.dump({"conversations": conversations}, f, ensure_ascii=False, indent=4)
        
        self.current_conversation_id = conversations[0]["conversation_id"]
    
    def _get_conversation_dir(self, conversation_id: str) -> str:
        return os.path.join(self.paths["conversations_dir"], conversation_id)
    
    def _get_context_path(self, conversation_id: str) -> str:
        return os.path.join(self._get_conversation_dir(conversation_id), "context.json")
    
    def _load_context(self) -> list:
        context_path = self._get_context_path(self.current_conversation_id)
        if os.path.exists(context_path):
            with open(context_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def save_context(self):
        with self._buffer_lock:
            context_path = self._get_context_path(self.current_conversation_id)
            with open(context_path, "w", encoding="utf-8") as f:
                json.dump(self.context_buffer, f, ensure_ascii=False, indent=4)
            self._update_conversation_last_updated(self.current_conversation_id)
    
    def _update_conversation_last_updated(self, conversation_id: str):
        if os.path.exists(self.paths["conversations"]):
            with open(self.paths["conversations"], "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations = data.get("conversations", [])
            
            for conv in conversations:
                if conv["conversation_id"] == conversation_id:
                    conv["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    break
            
            with open(self.paths["conversations"], "w", encoding="utf-8") as f:
                json.dump({"conversations": conversations}, f, ensure_ascii=False, indent=4)
    
    def create_conversation(self, name: str = None) -> str:
        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if name is None:
            name = f"对话{len(self.get_conversations()) + 1}"
        
        conv_dir = self._get_conversation_dir(conv_id)
        os.makedirs(conv_dir, exist_ok=True)
        
        if os.path.exists(self.paths["conversations"]):
            with open(self.paths["conversations"], "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations = data.get("conversations", [])
        else:
            conversations = []
        
        conversations.append({
            "conversation_id": conv_id,
            "name": name,
            "created_at": created_at,
            "last_updated": created_at
        })
        
        with open(self.paths["conversations"], "w", encoding="utf-8") as f:
            json.dump({"conversations": conversations}, f, ensure_ascii=False, indent=4)
        
        return conv_id
    
    def delete_conversation(self, conversation_id: str):
        if os.path.exists(self.paths["conversations"]):
            with open(self.paths["conversations"], "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations = data.get("conversations", [])
            
            conversations = [conv for conv in conversations if conv["conversation_id"] != conversation_id]
            
            with open(self.paths["conversations"], "w", encoding="utf-8") as f:
                json.dump({"conversations": conversations}, f, ensure_ascii=False, indent=4)
        
        conv_dir = self._get_conversation_dir(conversation_id)
        if os.path.exists(conv_dir):
            import shutil
            shutil.rmtree(conv_dir)
        
        if not conversations:
            self._ensure_conversations()
        elif self.current_conversation_id == conversation_id:
            self.current_conversation_id = conversations[0]["conversation_id"]
            self.context_buffer = self._load_context()
    
    def switch_conversation(self, conversation_id: str):
        if os.path.exists(self.paths["conversations"]):
            with open(self.paths["conversations"], "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations = data.get("conversations", [])
            
            found = any(conv["conversation_id"] == conversation_id for conv in conversations)
            if found:
                self.current_conversation_id = conversation_id
                self.context_buffer = self._load_context()
    
    def get_conversations(self) -> list:
        if os.path.exists(self.paths["conversations"]):
            with open(self.paths["conversations"], "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("conversations", [])
        return []
    
    def update_conversation_name(self, conversation_id: str, new_name: str):
        if os.path.exists(self.paths["conversations"]):
            with open(self.paths["conversations"], "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations = data.get("conversations", [])
            
            for conv in conversations:
                if conv["conversation_id"] == conversation_id:
                    conv["name"] = new_name
                    break
            
            with open(self.paths["conversations"], "w", encoding="utf-8") as f:
                json.dump({"conversations": conversations}, f, ensure_ascii=False, indent=4)
    
    def get_current_conversation(self) -> dict:
        conversations = self.get_conversations()
        for conv in conversations:
            if conv["conversation_id"] == self.current_conversation_id:
                return conv
        return None
    
    def get_first_conversation_timestamp(self) -> float:
        conversations = self.get_conversations()
        if conversations:
            first_conv = min(conversations, key=lambda x: x["created_at"])
            try:
                dt = datetime.strptime(first_conv["created_at"], "%Y-%m-%d %H:%M:%S")
                return dt.timestamp()
            except Exception:
                pass
        return time.time()

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
            # 显式重新设置 embedding_function，确保使用当前路径
            self.raw_clients[folder_name]._embedding_function = self.embedding_fn
        return self.raw_clients[folder_name]

    def _enforce_retention_policy(self, tier: str):
        count_limit = self.retention_limits.get(tier)
        time_limit = self.time_limits.get(tier)
        if not count_limit or not time_limit: return

        collection = self.summary_dbs[tier]
        all_data = collection.get()
        ids = all_data.get("ids", [])
        metas = all_data.get("metadatas", [])
        current_time = time.time()

        if not ids: return

        records = list(zip(ids, [m["timestamp"] for m in metas]))
        
        # 第一步：按时间过滤，只保留时间限制内的记录
        valid_records = []
        ids_to_delete_time = []
        for record in records:
            record_id, record_timestamp = record
            if current_time - record_timestamp <= time_limit:
                valid_records.append(record)
            else:
                ids_to_delete_time.append(record_id)
        
        # 删除超过时间限制的记录
        if ids_to_delete_time:
            collection.delete(ids=ids_to_delete_time)
            logger.debug(f"[MemoryDB] {tier}层删除 {len(ids_to_delete_time)} 条超过时间限制的记录")
        
        # 第二步：如果剩下的记录仍然超过数量限制，删除最旧的
        if len(valid_records) > count_limit:
            valid_records.sort(key=lambda x: x[1])
            overflow_count = len(valid_records) - count_limit
            ids_to_delete_count = [r[0] for r in valid_records[:overflow_count]]
            collection.delete(ids=ids_to_delete_count)
            logger.debug(f"[MemoryDB] {tier}层删除 {len(ids_to_delete_count)} 条超过数量限制的记录")

    def retrieve_with_time_routing(self, query: str, top_k: int = 3) -> List[str]:
        current_time = time.time()
        results = []
        
        day_7 = 7 * 24 * 3600
        day_14 = 14 * 24 * 3600
        month_2 = 60 * 24 * 3600
        year_1 = 365 * 24 * 3600
        week_8 = 8 * 7 * 24 * 3600
        
        all_candidates = []
        
        def add_candidates(tier: str, format_str: str):
            res = self.summary_dbs[tier].query(query_texts=[query], n_results=top_k * 2)
            if res.get("documents") and res["documents"][0]:
                for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                    time_diff = current_time - meta["timestamp"]
                    all_candidates.append((-time_diff, time_diff, doc, format_str))
        
        add_candidates("daily", "[数日前的模糊回忆]")
        add_candidates("weekly", "[数周前的模糊回忆]")
        add_candidates("monthly", "[数月前的模糊回忆]")
        add_candidates("yearly", "[数年前的模糊回忆]")
        
        all_candidates.sort(key=lambda x: x[0])
        
        selected = []
        for _, time_diff, doc, format_str in all_candidates:
            if time_diff <= day_7:
                if "数日前" in format_str:
                    selected.append((time_diff, f"{format_str}: {doc}"))
            elif day_7 < time_diff <= day_14:
                if "数日前" in format_str or "数周前" in format_str:
                    selected.append((time_diff, f"{format_str}: {doc}"))
            elif day_14 < time_diff <= month_2:
                if "数周前" in format_str or "数月前" in format_str:
                    selected.append((time_diff, f"{format_str}: {doc}"))
            elif month_2 < time_diff <= year_1:
                if "数月前" in format_str or "数年前" in format_str:
                    selected.append((time_diff, f"{format_str}: {doc}"))
            else:
                if "数年前" in format_str:
                    selected.append((time_diff, f"{format_str}: {doc}"))
        
        selected.sort(key=lambda x: x[0])
        
        final_results = []
        week_8_candidates = []
        other_candidates = []
        
        for time_diff, result_str in selected:
            if time_diff <= week_8:
                week_8_candidates.append(result_str)
            else:
                other_candidates.append(result_str)
        
        final_results = week_8_candidates + other_candidates
        
        return final_results[:top_k * 2]

    def retrieve_from_raw_records(self, query: str, top_k: int = 3) -> List[str]:
        """
        从原始记录库中检索记忆（增强模式）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[str]: 带格式标记的记忆列表
        """
        current_time = time.time()
        half_year_seconds = 180 * 24 * 3600
        start_time = current_time - half_year_seconds
        
        all_candidates = []
        
        # 遍历所有可能的原始记录分区（最近半年可能涉及1-2个分区）
        for i in range(0, 2):
            check_time = current_time - (i * 180 * 24 * 3600)
            try:
                collection = self.get_raw_collection(check_time)
                res = collection.query(query_texts=[query], n_results=top_k * 3)
                
                if res.get("documents") and res["documents"][0]:
                    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                        timestamp = meta.get("timestamp", 0)
                        if timestamp >= start_time:
                            time_diff = current_time - timestamp
                            all_candidates.append((-time_diff, doc))
            except Exception as e:
                logger.warning(f"[MemoryDB] 检索原始记录分区失败：{str(e)}")
                continue
        
        # 按时间倒序排序（最近的优先）
        all_candidates.sort(key=lambda x: x[0])
        
        # 格式化结果
        results = []
        for _, doc in all_candidates[:top_k]:
            results.append(f"[仔细回想起来的清晰回忆]: {doc}")
        
        return results

    def retrieve_with_depth_mode(self, query: str, top_k: int = 3) -> List[str]:
        """
        根据深度回忆模式选择不同的检索策略
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[str]: 检索到的记忆列表
        """
        import time
        start_time = time.time()
        mode = self.depth_recall_mode
        
        print("\n" + "=" * 80)
        print(f"[深度回忆调试] 开始检索")
        print(f"[深度回忆调试] 查询文本: {query}")
        print(f"[深度回忆调试] 当前深度回忆模式: {mode}")
        print(f"[深度回忆调试] 检索数量: top_k={top_k}")
        
        logger.debug(f"[MemoryDB] 深度回忆模式: {mode}")
        
        results = []
        if mode == "off":
            logger.debug("[MemoryDB] 深度回忆已关闭，不检索任何记忆")
            print(f"[深度回忆调试] 模式为 'off'，不检索任何记忆")
            results = []
        elif mode == "normal":
            logger.debug("[MemoryDB] 正常模式：使用时间路由检索")
            print(f"[深度回忆调试] 正常模式：使用时间路由检索")
            results = self.retrieve_with_time_routing(query, top_k)
        elif mode == "enhanced":
            logger.debug("[MemoryDB] 增强模式：从原始记录库检索")
            print(f"[深度回忆调试] 增强模式：从原始记录库检索")
            results = self.retrieve_from_raw_records(query, top_k)
        else:
            logger.warning(f"[MemoryDB] 未知的深度回忆模式: {mode}，使用默认值 'off'")
            print(f"[深度回忆调试] 未知模式: {mode}，使用默认值 'off'")
            results = []
        
        elapsed_time = time.time() - start_time
        print(f"[深度回忆调试] 检索耗时: {elapsed_time:.4f} 秒")
        print(f"[深度回忆调试] 检索到的回忆数量: {len(results)}")
        if results:
            print(f"[深度回忆调试] 成功检索到的回忆内容")
            # 不再打印内容作为调试信息
            # for i, memory in enumerate(results, 1):
            #     print(f"  [{i}] {memory}")
        else:
            print(f"[深度回忆调试] 没有检索到任何回忆")
        print("=" * 80 + "\n")
        
        return results

    def add_memory(self, role: str, text: str, model: str = None, token_usage: dict = None):
        if not text.strip(): return
        current_time = time.time()
        msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        
        logger.debug(f"[MemoryDB] 添加记忆：role={role}, text长度={len(text)}, model={model}")

        memory_data = {
            "role": role, "content": text, "timestamp": current_time, "msg_id": msg_id,
            "conversation_id": self.current_conversation_id
        }
        if model:
            memory_data["model"] = model
        if token_usage:
            memory_data["token_usage"] = token_usage

        self.context_buffer.append(memory_data)
        self.save_context()
        
        raw_col = self.get_raw_collection(current_time)
        
        try:
            start = time.time()
            metadata = {"role": role, "timestamp": current_time, "msg_id": msg_id,
                       "conversation_id": self.current_conversation_id}
            if model:
                metadata["model"] = model
            if token_usage:
                metadata["token_usage"] = json.dumps(token_usage) if isinstance(token_usage, dict) else str(token_usage)
            raw_col.add(
                documents=[text],
                metadatas=[metadata],
                ids=[msg_id]
            )
            elapsed = time.time() - start
            logger.debug(f"[MemoryDB] 向量数据库写入完成，耗时：{elapsed:.2f}秒")
        except Exception as e:
            logger.warning(f"[MemoryDB] 向量数据库写入失败：{str(e)}")
            logger.debug("向量数据库错误详情：", exc_info=True)

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

    def _get_summaries_in_time_range(self, tier: str, start_time: float, end_time: float):
        """
        从指定层级的总结库中获取时间范围内的所有总结
        
        Args:
            tier: 层级名称 ("daily", "weekly", "monthly", "yearly")
            start_time: 起始时间戳
            end_time: 结束时间戳
            
        Returns:
            tuple: (documents, metadatas) 按时间排序的文档列表和元数据列表
        """
        if tier not in self.summary_dbs:
            logger.warning(f"[MemoryDB] 找不到 {tier} 层的总结数据库")
            return [], []
        
        collection = self.summary_dbs[tier]
        all_data = collection.get()
        
        if not all_data.get("documents"):
            return [], []
        
        documents = all_data["documents"]
        metadatas = all_data.get("metadatas", [])
        
        records = []
        for doc, meta in zip(documents, metadatas):
            timestamp = meta.get("timestamp", 0)
            if start_time <= timestamp <= end_time:
                records.append((timestamp, doc, meta))
        
        records.sort(key=lambda x: x[0])
        sorted_docs = [r[1] for r in records]
        sorted_metas = [r[2] for r in records]
        
        logger.debug(f"[MemoryDB] 从 {tier} 层获取了 {len(sorted_docs)} 条时间范围内的总结")
        return sorted_docs, sorted_metas

    def _fallback_summary(self, texts: List[str], target_tier: str) -> str:
        """
        基于关键词提取的非LLM总结算法（回退方法）
        
        Args:
            texts: 文本列表
            target_tier: 目标层级
            
        Returns:
            str: 生成的总结字符串
        """
        if not texts:
            return ""
        
        logger.debug(f"[MemoryDB] 使用回退算法从 {len(texts)} 条文本生成 {target_tier} 级总结")
        
        all_text = "\n".join(texts)
        
        sentences = []
        for line in all_text.split("\n"):
            line = line.strip()
            if line:
                sentences.extend(line.split("。"))
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 5:
            summary = "。".join(sentences) + "。"
        else:
            key_sentences = sentences[:3] + sentences[-2:]
            summary = "。".join(key_sentences) + "。"
        
        logger.debug(f"[MemoryDB] 回退算法生成总结完成，长度: {len(summary)}")
        return summary
    
    def _generate_summary_from_texts(self, texts: List[str], target_tier: str) -> str:
        """
        生成总结，优先使用 LLM，失败时回退到非 LLM 算法
        
        Args:
            texts: 文本列表
            target_tier: 目标层级
            
        Returns:
            str: 生成的总结字符串
        """
        if not texts:
            return ""
        
        start_time = time.time()
        use_llm = False
        
        try:
            if self.llm_adapter:
                prompt_map = {
                    "daily": SUMMARY_PROMPT_DAILY,
                    "weekly": SUMMARY_PROMPT_WEEKLY,
                    "monthly": SUMMARY_PROMPT_MONTHLY,
                    "yearly": SUMMARY_PROMPT_YEARLY
                }
                
                prompt_template = prompt_map.get(target_tier, SUMMARY_PROMPT_DAILY)
                prompt = prompt_template.format(texts="\n\n".join(texts))
                
                logger.debug(f"[MemoryDB] 使用 LLM 生成 {target_tier} 级总结")
                
                answer_parts = []
                for chunk_type, chunk_content in self.llm_adapter.stream_chat(
                    prompt,
                    enable_think=True,
                    thinking_budget=4096,
                    model="qwen3.5-flash"
                ):
                    if chunk_type == "answer":
                        answer_parts.append(chunk_content)
                
                summary = "".join(answer_parts)
                
                if summary and len(summary.strip()) > 0:
                    use_llm = True
                    elapsed = time.time() - start_time
                    logger.info(f"[MemoryDB] LLM 生成 {target_tier} 级总结成功，耗时: {elapsed:.2f}秒，长度: {len(summary)}")
                    return summary
                else:
                    logger.warning(f"[MemoryDB] LLM 生成的总结为空，使用回退算法")
            else:
                logger.debug(f"[MemoryDB] LLM 适配器未初始化，使用回退算法")
        except Exception as e:
            logger.warning(f"[MemoryDB] LLM 生成总结失败: {str(e)}，使用回退算法")
            logger.debug("异常详情：", exc_info=True)
        
        summary = self._fallback_summary(texts, target_tier)
        elapsed = time.time() - start_time
        logger.info(f"[MemoryDB] 使用回退算法生成 {target_tier} 级总结，耗时: {elapsed:.2f}秒，长度: {len(summary)}")
        return summary

    def _should_compress_summaries(self, source_tier: str, target_tier: str) -> bool:
        """
        检测是否应该触发总结压缩
        
        Args:
            source_tier: 源层级
            target_tier: 目标层级
            
        Returns:
            bool: 是否应该压缩
        """
        if source_tier not in self.summary_dbs:
            return False
        
        collection = self.summary_dbs[source_tier]
        all_data = collection.get()
        count = len(all_data.get("ids", []))
        
        limit = self.retention_limits.get(source_tier, 100)
        threshold = limit // 2
        
        should_compress = count > threshold
        
        logger.debug(f"[MemoryDB] 检查 {source_tier} -> {target_tier} 压缩条件: 当前={count}, 阈值={threshold}, 结果={should_compress}")
        return should_compress

    def _generate_weekly_summary(self):
        """
        生成周级总结
        
        功能：
        - 获取最近的日总结
        - 使用已有的 _generate_summary_from_texts() 方法生成周总结
        - 使用已有的 save_summary() 方法保存周总结
        - 保存时使用该周第一天的时间戳作为 source_timestamp
        """
        if "daily" not in self.summary_dbs:
            logger.warning("[MemoryDB] 找不到日级总结数据库，无法生成周总结")
            return
        
        collection = self.summary_dbs["daily"]
        all_data = collection.get()
        
        if not all_data.get("documents"):
            logger.debug("[MemoryDB] 没有可用的日总结，跳过周总结生成")
            return
        
        documents = all_data["documents"]
        metadatas = all_data.get("metadatas", [])
        
        records = []
        for doc, meta in zip(documents, metadatas):
            timestamp = meta.get("timestamp", 0)
            records.append((timestamp, doc, meta))
        
        records.sort(key=lambda x: x[0], reverse=True)
        
        if not records:
            logger.debug("[MemoryDB] 没有有效的日总结记录，跳过周总结生成")
            return
        
        limit = self.retention_limits.get("daily", 14)
        threshold = limit // 2
        
        if len(records) < threshold:
            logger.debug(f"[MemoryDB] 日总结数量不足，当前={len(records)}, 阈值={threshold}")
            return
        
        selected_records = records[:limit]
        selected_records.sort(key=lambda x: x[0])
        
        texts = [r[1] for r in selected_records]
        first_timestamp = selected_records[0][0]
        
        dt = datetime.fromtimestamp(first_timestamp)
        days_to_monday = dt.weekday()
        week_start = datetime(dt.year, dt.month, dt.day) - datetime.timedelta(days=days_to_monday)
        week_start_timestamp = week_start.timestamp()
        
        weekly_summary = self._generate_summary_from_texts(texts, "weekly")
        
        if weekly_summary:
            self.save_summary("weekly", weekly_summary, week_start_timestamp)
            logger.info(f"[MemoryDB] 成功生成并保存周总结，涵盖 {len(texts)} 天的日总结")

    def compress_to_weekly(self):
        """
        压缩到周级
        
        功能：
        - 检测是否应该生成周总结（使用 _should_compress_summaries()）
        - 如果应该，则调用 _generate_weekly_summary()
        - 生成后调用 _enforce_retention_policy("weekly") 执行保留策略
        """
        logger.info("[MemoryDB] 开始执行周级压缩检查")
        
        if self._should_compress_summaries("daily", "weekly"):
            logger.info("[MemoryDB] 触发周级总结生成")
            self._generate_weekly_summary()
            self._enforce_retention_policy("weekly")
            logger.info("[MemoryDB] 周级压缩完成")
        else:
            logger.debug("[MemoryDB] 暂不需要进行周级压缩")

    def _generate_monthly_summary(self):
        """
        生成月级总结
        
        功能：
        - 获取最近的周总结
        - 使用已有的 _generate_summary_from_texts() 方法生成月总结
        - 使用已有的 save_summary() 方法保存月总结
        - 保存时使用该月第一天的时间戳作为 source_timestamp
        """
        if "weekly" not in self.summary_dbs:
            logger.warning("[MemoryDB] 找不到周级总结数据库，无法生成月总结")
            return
        
        collection = self.summary_dbs["weekly"]
        all_data = collection.get()
        
        if not all_data.get("documents"):
            logger.debug("[MemoryDB] 没有可用的周总结，跳过月总结生成")
            return
        
        documents = all_data["documents"]
        metadatas = all_data.get("metadatas", [])
        
        records = []
        for doc, meta in zip(documents, metadatas):
            timestamp = meta.get("timestamp", 0)
            records.append((timestamp, doc, meta))
        
        records.sort(key=lambda x: x[0], reverse=True)
        
        if not records:
            logger.debug("[MemoryDB] 没有有效的周总结记录，跳过月总结生成")
            return
        
        limit = self.retention_limits.get("weekly", 8)
        threshold = limit // 2
        
        if len(records) < threshold:
            logger.debug(f"[MemoryDB] 周总结数量不足，当前={len(records)}, 阈值={threshold}")
            return
        
        selected_records = records[:limit]
        selected_records.sort(key=lambda x: x[0])
        
        texts = [r[1] for r in selected_records]
        first_timestamp = selected_records[0][0]
        
        dt = datetime.fromtimestamp(first_timestamp)
        month_start = datetime(dt.year, dt.month, 1)
        month_start_timestamp = month_start.timestamp()
        
        monthly_summary = self._generate_summary_from_texts(texts, "monthly")
        
        if monthly_summary:
            self.save_summary("monthly", monthly_summary, month_start_timestamp)
            logger.info(f"[MemoryDB] 成功生成并保存月总结，涵盖 {len(texts)} 周的周总结")

    def compress_to_monthly(self):
        """
        压缩到月级
        
        功能：
        - 检测是否应该生成月总结（使用 _should_compress_summaries()）
        - 如果应该，则调用 _generate_monthly_summary()
        - 生成后调用 _enforce_retention_policy("monthly") 执行保留策略
        """
        logger.info("[MemoryDB] 开始执行月级压缩检查")
        
        if self._should_compress_summaries("weekly", "monthly"):
            logger.info("[MemoryDB] 触发月级总结生成")
            self._generate_monthly_summary()
            self._enforce_retention_policy("monthly")
            logger.info("[MemoryDB] 月级压缩完成")
        else:
            logger.debug("[MemoryDB] 暂不需要进行月级压缩")

    def _generate_yearly_summary(self):
        """
        生成年级总结
        
        功能：
        - 获取最近的月总结
        - 使用已有的 _generate_summary_from_texts() 方法生成年总结
        - 使用已有的 save_summary() 方法保存年总结
        - 保存时使用该年第一天的时间戳作为 source_timestamp
        """
        if "monthly" not in self.summary_dbs:
            logger.warning("[MemoryDB] 找不到月级总结数据库，无法生成年总结")
            return
        
        collection = self.summary_dbs["monthly"]
        all_data = collection.get()
        
        if not all_data.get("documents"):
            logger.debug("[MemoryDB] 没有可用的月总结，跳过年总结生成")
            return
        
        documents = all_data["documents"]
        metadatas = all_data.get("metadatas", [])
        
        records = []
        for doc, meta in zip(documents, metadatas):
            timestamp = meta.get("timestamp", 0)
            records.append((timestamp, doc, meta))
        
        records.sort(key=lambda x: x[0], reverse=True)
        
        if not records:
            logger.debug("[MemoryDB] 没有有效的月总结记录，跳过年总结生成")
            return
        
        limit = self.retention_limits.get("monthly", 12)
        threshold = limit // 2
        
        if len(records) < threshold:
            logger.debug(f"[MemoryDB] 月总结数量不足，当前={len(records)}, 阈值={threshold}")
            return
        
        selected_records = records[:limit]
        selected_records.sort(key=lambda x: x[0])
        
        texts = [r[1] for r in selected_records]
        first_timestamp = selected_records[0][0]
        
        dt = datetime.fromtimestamp(first_timestamp)
        year_start = datetime(dt.year, 1, 1)
        year_start_timestamp = year_start.timestamp()
        
        yearly_summary = self._generate_summary_from_texts(texts, "yearly")
        
        if yearly_summary:
            self.save_summary("yearly", yearly_summary, year_start_timestamp)
            logger.info(f"[MemoryDB] 成功生成并保存年总结，涵盖 {len(texts)} 个月的月总结")

    def compress_to_yearly(self):
        """
        压缩到年级
        
        功能：
        - 检测是否应该生成年总结（使用 _should_compress_summaries()）
        - 如果应该，则调用 _generate_yearly_summary()
        - 生成后调用 _enforce_retention_policy("yearly") 执行保留策略
        """
        logger.info("[MemoryDB] 开始执行年级压缩检查")
        
        if self._should_compress_summaries("monthly", "yearly"):
            logger.info("[MemoryDB] 触发年级总结生成")
            self._generate_yearly_summary()
            self._enforce_retention_policy("yearly")
            logger.info("[MemoryDB] 年级压缩完成")
        else:
            logger.debug("[MemoryDB] 暂不需要进行年级压缩")

    def close(self):
        """
        关闭所有数据库连接，释放文件锁定
        """
        try:
            for tier in ["daily", "weekly", "monthly", "yearly"]:
                if tier in self.summary_dbs:
                    try:
                        if hasattr(self.summary_dbs[tier], '_client') and hasattr(self.summary_dbs[tier]._client, 'stop'):
                            self.summary_dbs[tier]._client.stop()
                    except Exception:
                        pass
                    self.summary_dbs[tier] = None
            self.summary_dbs = {}
        except Exception as e:
            logger.warning(f"关闭数据库连接失败: {e}")
        
        try:
            for folder_name in list(self.raw_clients.keys()):
                try:
                    if hasattr(self.raw_clients[folder_name], '_client') and hasattr(self.raw_clients[folder_name]._client, 'stop'):
                        self.raw_clients[folder_name]._client.stop()
                except Exception:
                    pass
                self.raw_clients[folder_name] = None
            self.raw_clients = {}
        except Exception as e:
            logger.warning(f"关闭原始记录数据库连接失败: {e}")