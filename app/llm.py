import time
import logging
from functools import lru_cache

from openai import AsyncOpenAI, OpenAI

from .config import get_settings
from .errors import UpstreamError
from .schemas import ChatResponse, Message, Usage

logger = logging.getLogger("chat-service.llm")

@lru_cache
def _get_client() -> OpenAI:
    s = get_settings()
    return OpenAI(
        api_key=s.deepseek_api_key,
        base_url=s.deepseek_base_url,
        timeout=s.request_timeout,
        max_retries=s.max_retries,
    )

@lru_cache
def _get_async_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.deepseek_api_key,
        base_url=s.deepseek_base_url,
        timeout=s.request_timeout,
        max_retries=s.max_retries,
    )


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    s = get_settings()
    return (
        prompt_tokens * s.input_price_per_1m
        + completion_tokens * s.output_price_per_1m
    ) / 1_000_000


def chat(messages: list[Message]) -> ChatResponse:
    s = get_settings()
    started = time.perf_counter()

    try:
        resp = _get_client().chat.completions.create(
            model=s.model_name,
            messages=[m.model_dump() for m in messages],
        )
    except Exception as exc:
        raise UpstreamError(
            detail=f"{type(exc).__name__}: {exc}",
            cause=exc,
        ) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = prompt_tokens + completion_tokens
    cost_yuan = round(estimate_cost(prompt_tokens, completion_tokens), 6)
    logger.info(
        "chat_ok model=%s prompt=%d completion=%d total=%d cost=%.6f latency_ms=%d",
        resp.model, prompt_tokens, completion_tokens, total_tokens, cost_yuan, latency_ms,
        extra={"model": resp.model, "latency_ms": latency_ms,
               "cost_yuan": cost_yuan, "tool_count": len(tool_calls)},
    )

    return ChatResponse(
        content=content,
        model=resp.model,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        cost_yuan=cost_yuan,
        latency_ms=latency_ms,
    )    

async def stream_chat(messages: list[Message]):
    """逐块产出 dict：{"type": "delta"} 若干次，最后一次 {"type": "done"}。"""
    s = get_settings()
    started = time.perf_counter()
    prompt_tokens = 0
    completion_tokens = 0

    try:
        stream = await _get_async_client().chat.completions.create(
            model=s.model_name,
            messages=[m.model_dump() for m in messages],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens

            if not chunk.choices:
                continue
    
            delta = chunk.choices[0].delta.content
            if delta:
                yield {"type": "delta", "text": delta}

    except Exception as exc:
        raise UpstreamError(
            detail=f"{type(exc).__name__}: {exc}",
            cause=exc,
        ) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    cost_yuan = round(estimate_cost(prompt_tokens, completion_tokens), 6)
    logger.info(
        "stream_ok model=%s prompt=%d completion=%d cost=%.6f latency_ms=%d",
        s.model_name, prompt_tokens, completion_tokens, cost_yuan, latency_ms,
        extra={"model": s.model_name, "latency_ms": latency_ms, "cost_yuan": cost_yuan},
    )
    yield {
        "type": "done",
        "model": s.model_name,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cost_yuan": cost_yuan,
        "latency_ms": latency_ms,
    }

def chat_round(messages: list[dict], tools: list[dict] | None = None) -> dict:
    s = get_settings()
    kwargs: dict = {"model": s.model_name, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    started = time.perf_counter()
    try:
        resp = _get_client().chat.completions.create(**kwargs)
    except Exception as exc:
        raise UpstreamError(detail=f"{type(exc).__name__}: {exc}", cause=exc) from exc

    choice = resp.choices[0].message
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    tool_calls = [tc.model_dump() for tc in (choice.tool_calls or [])]
    latency_ms = int((time.perf_counter() - started) * 1000)
    cost_yuan = round(estimate_cost(prompt_tokens, completion_tokens), 6)

    logger.info(
        "chat_round model=%s prompt_tokens=%d completion_tokens=%d tool_count=%d "
        "cost=%.6f latency_ms=%d",
        resp.model, prompt_tokens, completion_tokens, len(tool_calls),
        cost_yuan, latency_ms,
        extra={"model": resp.model, "latency_ms": latency_ms, "cost_yuan": cost_yuan},
    )

    return {
        "role": "assistant",
        "content": choice.content or "",
        "tool_calls": tool_calls,
        "model": resp.model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_yuan": cost_yuan,
        "latency_ms": latency_ms,
    }    

