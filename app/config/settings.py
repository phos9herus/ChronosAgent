from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """应用配置类,支持环境变量和默认值"""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Roleplay System"
    VERSION: str = "1.0.0"

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    DATA_DIR: str = "data"
    ROLES_DIR: str = "data/roles"
    AVATARS_DIR: str = "data/avatars"
    CREDENTIALS_FILE: str = "data/credentials.json"
    USER_META_FILE: str = "data/user_meta.json"
    ROLES_REGISTRY_FILE: str = "data/roles_registry.json"

    HF_CACHE_DIR: str = "data/hf_models"
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    
    @property
    def hf_cache_dir_absolute(self) -> str:
        """获取 HF 缓存目录的绝对路径"""
        if os.path.isabs(self.HF_CACHE_DIR):
            return self.HF_CACHE_DIR
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            return os.path.join(base_dir, self.HF_CACHE_DIR)

    MAX_CONTEXT_LENGTH: int = 6000
    IDLE_TIMEOUT: int = 900
    BOUNDARY_TIMEOUT: int = 1800

    RETENTION_LIMITS: dict = {
        "daily": 14,
        "weekly": 8,
        "monthly": 12,
        "yearly": 100
    }

    DEFAULT_API_SETTINGS: dict = {
        "temperature": 1.0,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.1,
        "enable_think": True,
        "display_think": True,
        "thinking_budget": 81920,
        "voice_id": ""
    }

    QWEN_API_KEY: Optional[str] = None
    DASHSCOPE_API_KEY: Optional[str] = None
    QWEN_API_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_WEB_API_URL: str = "https://chat2.qianwen.com/api/v2/chat"

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/app.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 5

    MAX_IMAGE_SIZE: int = 4 * 1024 * 1024
    ALLOWED_IMAGE_TYPES: List[str] = ["image/png", "image/jpeg", "image/jpg"]

    MONITOR_INTERVAL: int = 10
    COMPRESS_THRESHOLD: float = 0.6

    @property
    def cors_origins_list(self) -> List[str]:
        """获取CORS允许的来源列表"""
        if self.DEBUG:
            return ["*"]
        return self.CORS_ORIGINS


settings = Settings()
