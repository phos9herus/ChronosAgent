from app.services.role_manager import RoleRegistry
from app.services.auth_manager import AuthManager
from app.services.user_manager import UserManager # 新增

class DataService:
    """全局数据服务单例，管理角色注册表、API凭证与用户个人中心"""
    def __init__(self):
        self.role_registry = RoleRegistry()
        self.auth_manager = AuthManager()
        self.user_manager = UserManager() # 新增

# 实例化单例，供 chat_service 和 API 路由直接导入使用
data_service = DataService()