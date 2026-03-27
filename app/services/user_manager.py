import os
import json

class UserManager:
    """
    纯后端接口：管理全局用户的个人中心信息 (data/user_meta.json)
    """
    def __init__(self, file_path: str = "data/user_meta.json"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        self.user_data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 默认用户初始化配置
        default_user = {
            "display_name": "User",
            "avatar_mode": "circle", # 可选: "circle" 或 "gradient"
            "avatar_circle": "",
            "avatar_bg": "",
            "preferred_model": "qwen3.5-plus"
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(default_user, f, ensure_ascii=False, indent=4)

        return default_user

    def get_user(self) -> dict:
        return self.user_data

    def update_user(self, data: dict):
        self.user_data.update(data)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.user_data, f, ensure_ascii=False, indent=4)
        return self.user_data