from role_manager import RoleRegistry
from auth_manager import AuthManager

class DataService:
    """全局数据服务单例，管理角色注册表与API凭证"""
    def __init__(self):
        # 实例化时会默认读取根目录下 data/ 文件夹中的 json 配置
        self.role_registry = RoleRegistry()
        self.auth_manager = AuthManager()



# 实例化单例，供 chat_service 和 API 路由直接导入使用
data_service = DataService()