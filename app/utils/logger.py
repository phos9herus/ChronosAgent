import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional
from app.config.settings import settings


class Logger:
    """统一的日志管理器"""

    _loggers = {}

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """获取或创建日志记录器"""
        if name is None:
            name = "app"

        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))

        if not logger.handlers:
            formatter = logging.Formatter(settings.LOG_FORMAT)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            log_dir = os.path.dirname(settings.LOG_FILE)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = RotatingFileHandler(
                settings.LOG_FILE,
                maxBytes=settings.LOG_MAX_BYTES,
                backupCount=settings.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return Logger.get_logger(name)
