from typing import Any, Optional


class BaseAppException(Exception):
    """应用基础异常类"""

    def __init__(self, message: str, code: int = 500, details: Optional[dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(BaseAppException):
    """认证错误"""

    def __init__(self, message: str = "认证失败", details: Optional[dict] = None):
        super().__init__(message, code=401, details=details)


class AuthorizationError(BaseAppException):
    """授权错误"""

    def __init__(self, message: str = "权限不足", details: Optional[dict] = None):
        super().__init__(message, code=403, details=details)


class ValidationError(BaseAppException):
    """验证错误"""

    def __init__(self, message: str = "数据验证失败", details: Optional[dict] = None):
        super().__init__(message, code=422, details=details)


class NotFoundError(BaseAppException):
    """资源未找到错误"""

    def __init__(self, message: str = "资源未找到", details: Optional[dict] = None):
        super().__init__(message, code=404, details=details)


class ConflictError(BaseAppException):
    """冲突错误"""

    def __init__(self, message: str = "资源冲突", details: Optional[dict] = None):
        super().__init__(message, code=409, details=details)


class InternalServerError(BaseAppException):
    """内部服务器错误"""

    def __init__(self, message: str = "内部服务器错误", details: Optional[dict] = None):
        super().__init__(message, code=500, details=details)


class ServiceUnavailableError(BaseAppException):
    """服务不可用错误"""

    def __init__(self, message: str = "服务暂时不可用", details: Optional[dict] = None):
        super().__init__(message, code=503, details=details)


class RoleNotFoundError(NotFoundError):
    """角色未找到错误"""

    def __init__(self, role_id: str):
        super().__init__(message=f"角色 '{role_id}' 不存在", details={"role_id": role_id})


class SessionNotFoundError(NotFoundError):
    """会话未找到错误"""

    def __init__(self, session_id: str):
        super().__init__(message=f"会话 '{session_id}' 不存在", details={"session_id": session_id})


class InvalidAPIKeyError(AuthenticationError):
    """无效的API密钥错误"""

    def __init__(self):
        super().__init__(message="无效的API密钥")


class ConfigurationError(InternalServerError):
    """配置错误"""

    def __init__(self, message: str = "配置错误", details: Optional[dict] = None):
        super().__init__(message, details=details)


class FileOperationError(InternalServerError):
    """文件操作错误"""

    def __init__(self, message: str = "文件操作失败", details: Optional[dict] = None):
        super().__init__(message, details=details)


class DatabaseError(InternalServerError):
    """数据库错误"""

    def __init__(self, message: str = "数据库操作失败", details: Optional[dict] = None):
        super().__init__(message, details=details)


class ConcurrencyError(InternalServerError):
    """并发错误"""

    def __init__(self, message: str = "并发操作冲突", details: Optional[dict] = None):
        super().__init__(message, details=details)
