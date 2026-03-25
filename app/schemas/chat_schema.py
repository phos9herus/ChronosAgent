from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    user_input: str
    role_id: str
    images: Optional[List[str]] = Field(default_factory=list)
    enable_think: bool = False
    force_deep_recall: bool = False

class ChatStreamResponse(BaseModel):
    msg_type: str  # "status", "thought", "answer", "error"
    content: str
    is_finished: bool = False