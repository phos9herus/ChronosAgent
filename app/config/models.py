# app/config/models.py

"""
模型配置管理模块
统一管理所有可用的 Qwen 模型
"""

# 完整的模型详细信息
MODEL_DETAILS = {
    "deepseek-v4-pro": {
        "name": "deepseek-v4-pro",
        "description": "旗舰级 MoE 大模型，总参1.6T、激活 49B，原生支持百万级超长上下文。依托海量高质量训练数据，具备顶尖数学逻辑、复杂推理、专业代码与长文本深度解析能力，适配高阶科研、复杂办公、深度智能代理等高难度场景。",
        "multimodal": False,
        "contextWindow": 1000000,
        "supportsThinking": True,
        "maxThinkingBudget": 256000,
        "supportsWebSearch": True,
        "features": ["工具集成", "Web搜索", "代码解释器", "深度思考", "长上下文"],
        "isToolEnabled": True,
        "tools": ["Web搜索", "网页信息提取", "代码解释器"]
    },
    "deepseek-v4-flash": {
        "name": "deepseek-v4-flash",
        "description": "高效轻量化MoE模型，总参284B，激活13B，原生支持百万超长上下文能力。推理速度快、延迟低、调用成本低廉，综合能力均衡，主打高并发、轻量化任务，适合日常对话、内容创作、基础 RAG、批量文案处理等普惠刚需场景。",
        "multimodal": False,
        "contextWindow": 1000000,
        "supportsThinking": True,
        "maxThinkingBudget": 256000,
        "supportsWebSearch": True,
        "features": ["工具集成", "Web搜索", "代码解释器", "深度思考", "长上下文"],
        "isToolEnabled": True,
        "tools": ["Web搜索", "网页信息提取", "代码解释器"]
    },
    "qwen3.7-max": {
        "name": "Qwen3.7-Max",
        "description": "Qwen3.7系列中规模最大、综合能力最强的Max模型，当前开放纯文本模型能力供体验。Qwen3.7是面向智能体时代的新一代旗舰模型，核心优势在于智能体能力的广度与深度：在编程、办公与生产力、长周期自主执行方面均能出色胜任各项任务。",
        "multimodal": False,
        "contextWindow": 1000000,
        "supportsThinking": True,
        "maxThinkingBudget": 256000,
        "supportsWebSearch": True,
        "features": ["工具集成", "Web搜索", "代码解释器", "深度思考", "长上下文"],
        "isToolEnabled": True,
        "tools": ["Web搜索", "网页信息提取", "代码解释器"]
    },
    "qwen3-max": {
        "name": "Qwen3-Max",
        "description": "通义千问3旗舰版，集成Web搜索、网页信息提取和代码解释器三项工具，通过在思考过程中引入外部工具，在复杂问题上实现更高的准确率。",
        "multimodal": False,
        "contextWindow": 262144,
        "supportsThinking": True,
        "maxThinkingBudget": 81920,
        "supportsWebSearch": True,
        "features": ["工具集成", "Web搜索", "代码解释器", "深度思考", "长上下文"],
        "isToolEnabled": True,
        "tools": ["Web搜索", "网页信息提取", "代码解释器"]
    },
    "qwen3.6-plus": {
        "name": "Qwen3.6-Plus",
        "description": "通义千问plus系列最新模型，支持深度思考和多模态理解，平衡性能与效果。",
        "multimodal": True,
        "contextWindow": 1000000,
        "supportsThinking": True,
        "maxThinkingBudget": 81920,
        "supportsWebSearch": True,
        "features": ["深度思考", "多模态", "长上下文", "代码生成"]
    },
    "qwen3.5-plus": {
        "name": "Qwen3.5-Plus",
        "description": "通义千问3.5增强版，支持深度思考和多模态理解，平衡性能与效果。",
        "multimodal": True,
        "contextWindow": 1000000,
        "supportsThinking": True,
        "maxThinkingBudget": 81920,
        "supportsWebSearch": True,
        "features": ["深度思考", "多模态", "长上下文", "代码生成"]
    },
    "qwen3.5-flash": {
        "name": "Qwen3.5-Flash",
        "description": "通义千问3.5轻量版，快速响应，适合实时对话和简单任务。",
        "multimodal": True,
        "contextWindow": 1000000,
        "supportsThinking": True,
        "maxThinkingBudget": 81920,
        "supportsWebSearch": True,
        "features": ["快速响应", "多模态", "轻量级", "高性价比"]
    },
    "qwen3-vl-plus": {
        "name": "Qwen3-VL-Plus",
        "description": "通义千问3视觉语言增强版，专注于图像理解与生成，支持多模态对话。",
        "multimodal": True,
        "contextWindow": 262144,
        "supportsThinking": True,
        "maxThinkingBudget": 81920,
        "supportsWebSearch": False,
        "features": ["图像理解", "图像生成", "多模态", "视觉问答", "OCR识别"]
    },
    "qwen3-vl-flash": {
        "name": "Qwen3-VL-Flash",
        "description": "通义千问3视觉语言轻量版，快速图像处理，适合实时视觉任务。",
        "multimodal": True,
        "contextWindow": 262144,
        "supportsThinking": False,
        "maxThinkingBudget": 81920,
        "supportsWebSearch": False,
        "features": ["快速图像处理", "多模态", "轻量级", "实时视觉"]
    },
    "qwen3.5-397b-a17b": {
        "name": "Qwen3.5-397B-A17B",
        "description": "通义千问3.5超大规模模型，397B总参数，A17B激活参数，超强性能，适合最复杂的任务。",
        "multimodal": True,
        "contextWindow": 262144,
        "supportsThinking": True,
        "maxThinkingBudget": 81920,
        "supportsWebSearch": False,
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
