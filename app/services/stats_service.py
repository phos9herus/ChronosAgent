import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


class StatsService:
    def __init__(self, data_dir: str = "data/stats"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._model_stats_file = self.data_dir / "model_stats.json"
        self._role_stats_file = self.data_dir / "role_stats.json"
        
        self._model_stats = self._load_json(self._model_stats_file, {})
        self._role_stats = self._load_json(self._role_stats_file, {})
    
    def _load_json(self, file_path: Path, default: Any) -> Any:
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default
    
    def _save_json(self, file_path: Path, data: Any) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def record_conversation(self, model_id: str, token_total: int, role_id: str, timestamp: float = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        
        date_key = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        
        self._record_model_usage(model_id, token_total, date_key, timestamp)
        self._record_role_usage(role_id, date_key, timestamp)
        
        self._save_json(self._model_stats_file, self._model_stats)
        self._save_json(self._role_stats_file, self._role_stats)
    
    def _record_model_usage(self, model_id: str, token_total: int, date_key: str, timestamp: float) -> None:
        if model_id not in self._model_stats:
            self._model_stats[model_id] = {
                "total_conversations": 0,
                "total_tokens": 0,
                "daily": {},
                "timeline": []
            }
        
        model = self._model_stats[model_id]
        model["total_conversations"] += 1
        model["total_tokens"] += token_total
        
        if date_key not in model["daily"]:
            model["daily"][date_key] = {"conversations": 0, "tokens": 0}
        model["daily"][date_key]["conversations"] += 1
        model["daily"][date_key]["tokens"] += token_total
        
        model["timeline"].append({
            "timestamp": timestamp,
            "date": date_key,
            "tokens": token_total
        })
    
    def _record_role_usage(self, role_id: str, date_key: str, timestamp: float) -> None:
        if role_id not in self._role_stats:
            self._role_stats[role_id] = {
                "total_conversations": 0,
                "daily": {},
                "timeline": [],
                "created_at": timestamp
            }
        
        role = self._role_stats[role_id]
        role["total_conversations"] += 1
        
        if date_key not in role["daily"]:
            role["daily"][date_key] = {"conversations": 0}
        role["daily"][date_key]["conversations"] += 1
        
        role["timeline"].append({
            "timestamp": timestamp,
            "date": date_key
        })
    
    def get_all_models_stats(self) -> List[Dict[str, Any]]:
        result = []
        for model_id, stats in self._model_stats.items():
            result.append({
                "model_id": model_id,
                "total_conversations": stats["total_conversations"],
                "total_tokens": stats["total_tokens"]
            })
        return result
    
    def get_model_stats_detail(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._model_stats:
            return None
        
        stats = self._model_stats[model_id]
        
        conversations_timeline = []
        tokens_timeline = []
        
        sorted_dates = sorted(stats["daily"].keys())
        for date in sorted_dates:
            daily = stats["daily"][date]
            conversations_timeline.append({
                "date": date,
                "count": daily["conversations"]
            })
            tokens_timeline.append({
                "date": date,
                "count": daily["tokens"]
            })
        
        return {
            "model_id": model_id,
            "total_conversations": stats["total_conversations"],
            "total_tokens": stats["total_tokens"],
            "conversations_timeline": conversations_timeline,
            "tokens_timeline": tokens_timeline
        }
    
    def get_all_roles_stats(self) -> List[Dict[str, Any]]:
        result = []
        for role_id, stats in self._role_stats.items():
            result.append({
                "role_id": role_id,
                "total_conversations": stats["total_conversations"]
            })
        return result
    
    def get_role_stats_detail(self, role_id: str) -> Dict[str, Any]:
        if role_id not in self._role_stats:
            return None
        
        stats = self._role_stats[role_id]
        
        conversations_timeline = []
        sorted_dates = sorted(stats["daily"].keys())
        for date in sorted_dates:
            daily = stats["daily"][date]
            conversations_timeline.append({
                "date": date,
                "count": daily["conversations"]
            })
        
        created_at = stats.get("created_at", time.time())
        created_seconds = int(time.time() - created_at)
        created_duration = int(created_seconds / 86400)
        
        return {
            "role_id": role_id,
            "total_conversations": stats["total_conversations"],
            "conversations_timeline": conversations_timeline,
            "created_days": created_duration,
            "created_seconds": created_seconds,
            "memory_overview": "记忆总览占位"
        }


stats_service = StatsService()
