import time
from functools import lru_cache

from openai import OpenAI

from .config import get_settings
from .schemas import ChatResponse, Message, Usage


class LLMError(Exception):
    """模型调用失败时对外抛出的统一错误类型。"""


@lru_cache
def _get_client() -> OpenAI:
    s = get_settings()
    return OpenAI(
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
        raise LLMError(f"模型调用失败: {type(exc).__name__}: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    return ChatResponse(
        content=content,
        model=resp.model,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        cost_yuan=round(estimate_cost(prompt_tokens, completion_tokens), 6),
        latency_ms=latency_ms,
    )