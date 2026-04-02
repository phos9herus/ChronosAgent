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
        
        self._migrate_model_stats()
    
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
    
    def _migrate_model_stats(self) -> None:
        needs_save = False
        for model_id, stats in self._model_stats.items():
            if "total_input_tokens" not in stats:
                stats["total_input_tokens"] = 0
                needs_save = True
            if "total_output_tokens" not in stats:
                stats["total_output_tokens"] = 0
                needs_save = True
            if "total_cached_tokens" not in stats:
                stats["total_cached_tokens"] = 0
                needs_save = True
            
            for date_key, daily in stats["daily"].items():
                if "input_tokens" not in daily:
                    daily["input_tokens"] = 0
                    needs_save = True
                if "output_tokens" not in daily:
                    daily["output_tokens"] = 0
                    needs_save = True
                if "cached_tokens" not in daily:
                    daily["cached_tokens"] = 0
                    needs_save = True
            
            for item in stats["timeline"]:
                if "input_tokens" not in item:
                    item["input_tokens"] = 0
                    needs_save = True
                if "output_tokens" not in item:
                    item["output_tokens"] = 0
                    needs_save = True
                if "cached_tokens" not in item:
                    item["cached_tokens"] = 0
                    needs_save = True
        
        if needs_save:
            self._save_json(self._model_stats_file, self._model_stats)
    
    def record_conversation(self, model_id: str, token_usage: Dict[str, int], role_id: str, timestamp: float = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        
        date_key = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        
        self._record_model_usage(model_id, token_usage, date_key, timestamp)
        self._record_role_usage(role_id, date_key, timestamp)
        
        self._save_json(self._model_stats_file, self._model_stats)
        self._save_json(self._role_stats_file, self._role_stats)
    
    def _record_model_usage(self, model_id: str, token_usage: Dict[str, int], date_key: str, timestamp: float) -> None:
        if model_id not in self._model_stats:
            self._model_stats[model_id] = {
                "total_conversations": 0,
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cached_tokens": 0,
                "daily": {},
                "timeline": []
            }
        
        model = self._model_stats[model_id]
        model["total_conversations"] += 1
        model["total_tokens"] += token_usage.get("total", 0)
        model["total_input_tokens"] += token_usage.get("input", 0)
        model["total_output_tokens"] += token_usage.get("output", 0)
        
        cached_value = token_usage.get("cached", 0)
        if cached_value != "不可用":
            model["total_cached_tokens"] += cached_value
        
        if date_key not in model["daily"]:
            model["daily"][date_key] = {
                "conversations": 0, 
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0
            }
        model["daily"][date_key]["conversations"] += 1
        model["daily"][date_key]["tokens"] += token_usage.get("total", 0)
        model["daily"][date_key]["input_tokens"] += token_usage.get("input", 0)
        model["daily"][date_key]["output_tokens"] += token_usage.get("output", 0)
        
        if cached_value != "不可用":
            model["daily"][date_key]["cached_tokens"] += cached_value
        
        timeline_entry = {
            "timestamp": timestamp,
            "date": date_key,
            "tokens": token_usage.get("total", 0),
            "input_tokens": token_usage.get("input", 0),
            "output_tokens": token_usage.get("output", 0)
        }
        
        if cached_value != "不可用":
            timeline_entry["cached_tokens"] = cached_value
        
        model["timeline"].append(timeline_entry)
    
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
                "total_tokens": stats["total_tokens"],
                "total_input_tokens": stats["total_input_tokens"],
                "total_output_tokens": stats["total_output_tokens"],
                "total_cached_tokens": stats["total_cached_tokens"]
            })
        return result
    
    def get_model_stats_detail(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._model_stats:
            return None
        
        stats = self._model_stats[model_id]
        
        conversations_timeline = []
        tokens_timeline = []
        input_tokens_timeline = []
        output_tokens_timeline = []
        cached_tokens_timeline = []
        
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
            input_tokens_timeline.append({
                "date": date,
                "count": daily["input_tokens"]
            })
            output_tokens_timeline.append({
                "date": date,
                "count": daily["output_tokens"]
            })
            cached_tokens_timeline.append({
                "date": date,
                "count": daily["cached_tokens"]
            })
        
        return {
            "model_id": model_id,
            "total_conversations": stats["total_conversations"],
            "total_tokens": stats["total_tokens"],
            "total_input_tokens": stats["total_input_tokens"],
            "total_output_tokens": stats["total_output_tokens"],
            "total_cached_tokens": stats["total_cached_tokens"],
            "conversations_timeline": conversations_timeline,
            "tokens_timeline": tokens_timeline,
            "input_tokens_timeline": input_tokens_timeline,
            "output_tokens_timeline": output_tokens_timeline,
            "cached_tokens_timeline": cached_tokens_timeline
        }
    
    def get_global_usage_stats(self) -> Dict[str, int]:
        total_input = 0
        total_output = 0
        for stats in self._model_stats.values():
            total_input += stats.get("total_input_tokens", 0)
            total_output += stats.get("total_output_tokens", 0)
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output
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
