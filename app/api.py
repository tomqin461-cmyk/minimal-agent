from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.deepseek_chat import ask_agent
from typing import Literal

app = FastAPI(
    title="Energy RAG Agent",
    version="0.1.0",
)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="用户的问题",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=8,
    )

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    history = [
        message.model_dump()
        for message in request.history
    ]

    result = ask_agent(request.question, history)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "references": result["references"],
        "status": "success",
    }