import json
import logging
from contextlib import asynccontextmanager
from .agent import run_agent

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings, get_settings
from .errors import AppError, new_ref, redact
from .llm import chat, stream_chat, chat_round
from .logs import setup_logging
from .schemas import (
    AgentResponse,
    ChatRequest,
    ChatResponse,
    ChatToolsResponse,
    Message,
    ToolCallRecord,
    Usage,
)
from .tools import run_tool, tool_specs

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


def _build_messages(req: ChatRequest) -> list[Message]:
    messages: list[Message] = []
    if req.system_prompt:
        messages.append(Message(role="system", content=req.system_prompt))
    messages.extend(req.history)
    messages.append(Message(role="user", content=req.message))
    return messages


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

@app.post("/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest) -> ChatResponse:
    return chat(_build_messages(req))

@app.post("/chat/stream")
async def post_chat_stream(req: ChatRequest):
    messages = _build_messages(req)

    async def event_stream():
        yield _sse({"type": "start"})
        try:
            async for item in stream_chat(messages):
                yield _sse(item)
        except AppError as exc:
            logger.error(
                "stream_app_error ref=%s detail=%s",
                exc.ref, exc.detail,
                extra={"ref": exc.ref, "path": "/chat/stream", "status": exc.status_code},
                exc_info=exc,
            )
            yield _sse({"type": "error", "message": exc.user_message, "ref": exc.ref})
        except Exception as exc:
            ref = new_ref()
            logger.error(
                "stream_unexpected ref=%s detail=%s",
                ref, redact(str(exc)),
                extra={"ref": ref, "path": "/chat/stream", "status": 500},
                exc_info=exc,
            )
            yield _sse({"type": "error", "message": "服务内部错误", "ref": ref})

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/chat/tools", response_model=ChatToolsResponse)
def post_chat_tools(req: ChatRequest) -> ChatToolsResponse:
    messages: list[dict] = [m.model_dump() for m in _build_messages(req)]
    rounds: list[dict] = []

    first = chat_round(messages, tools=tool_specs())
    rounds.append(first)
    records: list[ToolCallRecord] = []

    if first["tool_calls"]:
        messages.append(
            {
                "role": "assistant",
                "content": first["content"],
                "tool_calls": first["tool_calls"],
            }
        )

        for call in first["tool_calls"]:
            fn = call["function"]
            try:
                args = json.loads(fn["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            result = run_tool(fn["name"], args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )
            records.append(ToolCallRecord(name=fn["name"], arguments=args, result=result))

        rounds.append(chat_round(messages, tools=None))

    final = rounds[-1]
    return ChatToolsResponse(
        content=final["content"],
        tool_calls=records,
        model=final["model"],
        usage=Usage(
            prompt_tokens=sum(r["prompt_tokens"] for r in rounds),
            completion_tokens=sum(r["completion_tokens"] for r in rounds),
            total_tokens=sum(r["prompt_tokens"] + r["completion_tokens"] for r in rounds),
        ),
        cost_yuan=round(sum(r["cost_yuan"] for r in rounds), 6),
        latency_ms=sum(r["latency_ms"] for r in rounds),
    )

@app.post("/agent", response_model=AgentResponse)
def post_agent(req: ChatRequest) -> AgentResponse:
    messages = [m.model_dump() for m in _build_messages(req)]
    result = run_agent(messages)

    return AgentResponse(
        content=result["content"],
        steps=result["steps"],
        stop_reason=result["stop_reason"],
        model=result["model"],
        rounds=result["rounds"],
        usage=Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
        ),
        cost_yuan=result["cost_yuan"],
        latency_ms=result["latency_ms"],
    )        