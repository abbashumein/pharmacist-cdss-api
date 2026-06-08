from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    detected_emotions: list[str]
    severity: str | None = None
    recommended_action: str | None = None
    follow_up_question: str | None = None
