from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from .config import Settings, get_settings
from .llm import LLMError, chat
from .schemas import ChatRequest, ChatResponse, Message

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # 启动即强制校验，配置有问题立刻崩，不要等到第一次请求
    yield

app = FastAPI(title="chat-service", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/config")
def show_config(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return {
        "model_name": settings.model_name,
        "base_url": settings.deepseek_base_url,
        "request_timeout": settings.request_timeout,
        "max_retries": settings.max_retries,
        "api_key_configured": bool(settings.deepseek_api_key),
        "input_price_per_1m": settings.input_price_per_1m,
        "output_price_per_1m": settings.output_price_per_1m,
    }


@app.post("/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest) -> ChatResponse:
    messages: list[Message] = []
    if req.system_prompt:
        messages.append(Message(role="system", content=req.system_prompt))
    messages.extend(req.history)
    messages.append(Message(role="user", content=req.message))

    try:
        return chat(messages)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc