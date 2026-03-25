import os
import json


class AuthManager:
    """
    纯后端接口：统一管理各平台大模型的鉴权凭证（Cookie / API Keys）。
    基于 credentials.json 实现中心化存储。
    """

    def __init__(self, file_path: str = "data/credentials.json"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        self.credentials = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 默认凭证模板 (全部换为 Qwen 体系)
        default_creds = {
            "qwen_web": {
                "cookie": ""
            },
            "qwen_api": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            }
        }

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(default_creds, f, ensure_ascii=False, indent=4)

        return default_creds

    def save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.credentials, f, ensure_ascii=False, indent=4)

    def get_credential(self, provider: str, key: str) -> str:
        """获取指定提供商的指定凭证"""
        return self.credentials.get(provider, {}).get(key, "")

    def set_credential(self, provider: str, key: str, value: str):
        """设置并持久化凭证"""
        if provider not in self.credentials:
            self.credentials[provider] = {}

        self.credentials[provider][key] = value
        self.save()