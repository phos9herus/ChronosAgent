import os
import json
import uuid
import time

class RoleRegistry:
    """
    纯后端接口：管理 roles_registry.json（退化为纯角色索引列表）。
    """

    def __init__(self, file_path: str = "data/roles_registry.json"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        self.roles = self._load()

    def update_role_settings(self, role_id: str, new_settings: dict) -> bool:
        role_info = self.get_role_by_id(role_id)
        if not role_info: return False

        meta_path = os.path.join("data", "roles", role_info.get("name"), "role_meta.json")
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                role_data = json.load(f)

            # 分离顶层参数和 settings 参数
            if "system_prompt" in new_settings:
                role_data["system_prompt"] = new_settings.pop("system_prompt")

            if "settings" not in role_data: role_data["settings"] = {}
            role_data["settings"].update(new_settings)

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(role_data, f, ensure_ascii=False, indent=4)
            return True
        except Exception:
            return False

    def _load(self) -> list:
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.roles, f, ensure_ascii=False, indent=4)

    def get_all_roles(self) -> list:
        return self.roles

    def get_role_by_id(self, role_id: str) -> dict:
        for role in self.roles:
            if role["role_id"] == role_id:
                return role
        return None

    def create_role(self, name: str) -> dict:
        """接口：仅在全局索引中注册新角色，具体属性交由独立 meta 管理"""
        role_data = {
            "role_id": f"role_{uuid.uuid4().hex[:8]}",
            "name": name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.roles.append(role_data)
        self.save()
        return role_data