# app/config/models.py

"""
模型配置管理模块
统一管理所有可用的 Qwen 模型
"""

# 完整的模型详细信息
MODEL_DETAILS = {
    "qwen3-max": {
        "name": "Qwen3-Max",
        "description": "通义千问3旗舰版，集成Web搜索、网页信息提取和代码解释器三项工具，通过在思考过程中引入外部工具，在复杂问题上实现更高的准确率。",
        "multimodal": True,
        "contextWindow": 131072,
        "supportsThinking": True,
        "maxThinkingBudget": 131072,
        "features": ["工具集成", "Web搜索", "代码解释器", "深度思考", "长上下文"],
        "isToolEnabled": True,
        "tools": ["Web搜索", "网页信息提取", "代码解释器"]
    },
    "qwen3.5-plus": {
        "name": "Qwen3.5-Plus",
        "description": "通义千问3.5增强版，支持深度思考和多模态理解，平衡性能与效果。",
        "multimodal": True,
        "contextWindow": 131072,
        "supportsThinking": True,
        "maxThinkingBudget": 131072,
        "features": ["深度思考", "多模态", "长上下文", "代码生成"]
    },
    "qwen3.5-flash": {
        "name": "Qwen3.5-Flash",
        "description": "通义千问3.5轻量版，快速响应，适合实时对话和简单任务。",
        "multimodal": True,
        "contextWindow": 65536,
        "supportsThinking": True,
        "maxThinkingBudget": 65536,
        "features": ["快速响应", "多模态", "轻量级", "高性价比"]
    },
    "qwq-plus": {
        "name": "QwQ-Plus",
        "description": "QwQ原生推理模型，专为深度推理任务设计，深度思考参数对其无效，模型本身即具备强大的推理能力。",
        "multimodal": False,
        "contextWindow": 131072,
        "supportsThinking": False,
        "maxThinkingBudget": 0,
        "features": ["原生推理", "深度思考", "数学推理", "逻辑分析"],
        "isReasoningModel": True,
        "note": "原生推理模型，深度思考参数对其无效"
    },
    "qwen3-vl-plus": {
        "name": "Qwen3-VL-Plus",
        "description": "通义千问3视觉语言增强版，专注于图像理解与生成，支持多模态对话。",
        "multimodal": True,
        "contextWindow": 65536,
        "supportsThinking": True,
        "maxThinkingBudget": 65536,
        "features": ["图像理解", "图像生成", "多模态", "视觉问答", "OCR识别"]
    },
    "qwen3-vl-flash": {
        "name": "Qwen3-VL-Flash",
        "description": "通义千问3视觉语言轻量版，快速图像处理，适合实时视觉任务。",
        "multimodal": True,
        "contextWindow": 32768,
        "supportsThinking": False,
        "maxThinkingBudget": 0,
        "features": ["快速图像处理", "多模态", "轻量级", "实时视觉"]
    },
    "qwen3.5-397b-a17b": {
        "name": "Qwen3.5-397B-A17B",
        "description": "通义千问3.5超大规模模型，397B总参数，A17B激活参数，超强性能，适合最复杂的任务。",
        "multimodal": True,
        "contextWindow": 131072,
        "supportsThinking": True,
        "maxThinkingBudget": 131072,
        "features": ["超大规模", "MoE架构", "最强性能", "复杂推理", "多模态"]
    }
}

# 用户指定的模型列表（保持向后兼容）
AVAILABLE_MODELS = list(MODEL_DETAILS.keys())

def get_available_models() -> list:
    """
    获取所有可用的模型列表
    
    Returns:
        list: 模型 ID 列表
    """
    return AVAILABLE_MODELS

def get_all_model_details() -> dict:
    """
    获取所有模型的详细信息
    
    Returns:
        dict: 完整的模型详细信息字典
    """
    return MODEL_DETAILS

def get_model_details(model_id: str) -> dict:
    """
    获取指定模型的详细信息
    
    Args:
        model_id: 模型 ID
        
    Returns:
        dict: 模型详细信息
    """
    return MODEL_DETAILS.get(model_id, {})

def is_valid_model(model_id: str) -> bool:
    """
    验证模型是否有效
    
    Args:
        model_id: 模型 ID
        
    Returns:
        bool: 是否为有效模型
    """
    return model_id in MODEL_DETAILS
