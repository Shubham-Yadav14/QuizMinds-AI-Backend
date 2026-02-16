from pydantic import BaseModel

class QuizRequest(BaseModel):
    subject: str
    question: str

class ModelResponse(BaseModel):
    model: str
    answer: str
    error: str | None = None
