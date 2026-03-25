import os
import json
import uuid
import time


class RoleRegistry:
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

            # 【核心修改】：拦截根节点字段，防止被塞进 settings 域
            root_keys = ["system_prompt", "display_name", "avatar_mode", "avatar_circle", "avatar_bg"]
            for key in root_keys:
                if key in new_settings:
                    role_data[key] = new_settings.pop(key)

            if "settings" not in role_data: role_data["settings"] = {}
            if new_settings:  # 剩下的才是大模型超参数
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
        roles_with_meta = []
        for r in self.roles:
            meta_path = os.path.join("data", "roles", r["name"], "role_meta.json")
            display_name = r["name"]
            avatar_mode = "circle"
            avatar_circle = ""
            avatar_bg = ""

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        display_name = meta.get("display_name", r["name"])
                        avatar_mode = meta.get("avatar_mode", "circle")
                        avatar_circle = meta.get("avatar_circle", "")
                        avatar_bg = meta.get("avatar_bg", "")
                except Exception:
                    pass

            roles_with_meta.append({
                "role_id": r["role_id"],
                "name": r["name"],
                "display_name": display_name,
                "avatar_mode": avatar_mode,
                "avatar_circle": avatar_circle,
                "avatar_bg": avatar_bg
            })
        return roles_with_meta

    def get_role_by_id(self, role_id: str) -> dict:
        for role in self.roles:
            if role["role_id"] == role_id:
                return role
        return None

    def create_role(self, name: str) -> dict:
        role_data = {
            "role_id": f"role_{uuid.uuid4().hex[:8]}",
            "name": name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.roles.append(role_data)
        self.save()
        return role_data