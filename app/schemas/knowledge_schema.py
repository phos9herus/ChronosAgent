from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: Optional[str] = Field(None, max_length=500, description="知识库描述")

class KnowledgeBaseUpdate(BaseModel):
    """更新知识库请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class DocumentInfo(BaseModel):
    """文档信息"""
    doc_id: str
    filename: str
    file_type: str  # docx | xlsx | pdf
    file_size: int
    uploaded_at: str
    parsed_status: str  # success | pending | error

class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    doc_id: str
    kb_id: str
    filename: str
    file_type: str
    file_size: int
    uploaded_at: str
    parsed_status: str  # success | pending | error
    parsed_message: Optional[str] = None

class DocumentDetailResponse(DocumentInfo):
    """文档详情响应（包含解析数据）"""
    parsed_data: Optional[Dict] = None

class KnowledgeBaseResponse(BaseModel):
    """知识库响应"""
    kb_id: str
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str
    document_count: int
    documents: List[DocumentInfo] = []

class KnowledgeBaseListItem(BaseModel):
    """知识库列表项（摘要）"""
    kb_id: str
    name: str
    document_count: int
    created_at: str
