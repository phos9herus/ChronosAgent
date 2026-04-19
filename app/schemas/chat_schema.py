from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class KnowledgeBaseRef(BaseModel):
    """知识库引用"""
    kb_id: str
    kb_name: str
    doc_id: str
    doc_name: str
    selected_text: str
    selected_length: int

class CitationMeta(BaseModel):
    """引用元数据"""
    kb_id: str
    kb_name: str
    doc_id: str
    doc_name: str
    char_start: int
    char_end: int

class ChatRequest(BaseModel):
    user_input: str
    role_id: str
    images: Optional[List[str]] = Field(default_factory=list)
    enable_think: bool = False
    force_deep_recall: bool = False
    knowledge_base_ref: Optional[KnowledgeBaseRef] = None
    knowledge_citations: Optional[List[CitationMeta]] = Field(default_factory=list)

class ChatStreamResponse(BaseModel):
    msg_type: str  # "status", "thought", "answer", "error"
    content: str
    is_finished: bool = False