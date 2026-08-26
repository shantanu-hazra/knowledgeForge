from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    user_id:str
    conversation_id:int | None = None  # optional, if you want conversation tracking

class ChatResponse(BaseModel):
    reply: str
    conversation_id:int | None = None  # optional, if you want conversation tracking

class RetrieveRequest(BaseModel):
    query: str