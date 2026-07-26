from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.deepseek_chat import ask_agent
app = FastAPI(
    title="Energy RAG Agent",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="用户的问题",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    result = ask_agent(request.question)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "status": "success",
    }