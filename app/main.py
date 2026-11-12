import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .errors import AppError, new_ref, redact
from .llm import chat
from .logs import setup_logging
from .schemas import ChatRequest, ChatResponse, Message

setup_logging()
logger = logging.getLogger("chat-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # 启动即强制校验，配置有问题立刻崩，不要等到第一次请求
    yield


app = FastAPI(title="chat-service", lifespan=lifespan)


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    logger.error(
        "app_error ref=%s path=%s status=%s detail=%s",
        exc.ref, request.url.path, exc.status_code, exc.detail,
        extra={"ref": exc.ref, "path": request.url.path, "status": exc.status_code},
        exc_info=exc,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.user_message, "ref": exc.ref},
        headers={"X-Ref": exc.ref},
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    ref = new_ref()
    logger.error(
        "unexpected_error ref=%s path=%s detail=%s",
        ref, request.url.path, redact(str(exc)),
        extra={"ref": ref, "path": request.url.path, "status": 500},
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "服务内部错误", "ref": ref},
        headers={"X-Ref": ref},
    )


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
    return chat(messages)