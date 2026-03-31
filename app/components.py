from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    session_id: str
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    source: str
    session_id: str