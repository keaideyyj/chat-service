"""agent 循环：模型连续决策 → 执行工具 → 观察结果 → 直到给出答案。"""

import json
import logging
import time

from .llm import chat_round
from .schemas import AgentStep
from .tools import run_tool, tool_specs

logger = logging.getLogger("chat-service.agent")

MAX_STEPS = 6

def _call_key(name: str, args: dict) -> str:
    """把「工具名 + 参数」压成一个可比较的字符串。"""
    return f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

def run_agent(messages: list[dict], max_steps: int = MAX_STEPS) -> dict:
    started = time.perf_counter()
    steps: list[AgentStep] = []
    rounds: list[dict] = []
    final_text = ""
    stop_reason = "max_steps"
    last_key: str | None = None
    loop_detected = False

    for step_index in range(1, max_steps + 1):
        round_result = chat_round(messages, tools=tool_specs())
        rounds.append(round_result)

        if not round_result["tool_calls"]:
            final_text = round_result["content"]
            stop_reason = "answer"
            steps.append(
                AgentStep(
                    step=step_index,
                    type="answer",
                    content=final_text,
                    llm_ms=round_result["latency_ms"],
                )
            )
            break

        messages.append(
            {
                "role": "assistant",
                "content": round_result["content"],
                "tool_calls": round_result["tool_calls"],
            }
        )

        for call in round_result["tool_calls"]:
            fn = call["function"]
            try:
                args = json.loads(fn["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            tool_started = time.perf_counter()
            key = _call_key(fn["name"], args)

            if key == last_key:
                loop_detected = True
                result = (
                    "拒绝执行：你刚刚用完全相同的参数调用过这个工具，结果不会改变。"
                    "请换一种思路，或直接基于已有信息给出回答。"
                )
                tool_ms = 0
            else:
                result = run_tool(fn["name"], args)
                tool_ms = int((time.perf_counter() - tool_started) * 1000)
            last_key = key

            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )
            steps.append(
                AgentStep(
                    step=step_index,
                    type="tool",
                    content=round_result["content"],
                    tool=fn["name"],
                    arguments=args,
                    result=result,
                    llm_ms=round_result["latency_ms"],
                    tool_ms=tool_ms,
                )
            )

    else:
        messages.append(
            {
                "role": "user",
                "content": "请基于以上已经拿到的信息直接给出最终回答，不要再调用任何工具。",
            }
        )
        forced = chat_round(messages, tools=None)
        rounds.append(forced)
        final_text = forced["content"]
        stop_reason = "forced_final"
        steps.append(AgentStep(step=max_steps + 1, type="answer", content=final_text))

    if loop_detected and stop_reason != "answer":
        stop_reason = "loop_detected"

    prompt_tokens = sum(r["prompt_tokens"] for r in rounds)
    completion_tokens = sum(r["completion_tokens"] for r in rounds)
    cost_yuan = round(sum(r["cost_yuan"] for r in rounds), 6)
    latency_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "agent_done stop=%s steps=%d rounds=%d prompt=%d completion=%d cost=%.6f latency_ms=%d",
        stop_reason, len(steps), len(rounds),
        prompt_tokens, completion_tokens, cost_yuan, latency_ms,
        extra={"latency_ms": latency_ms, "cost_yuan": cost_yuan},
    )

    return {
        "content": final_text,
        "steps": steps,
        "stop_reason": stop_reason,
        "model": rounds[-1]["model"] if rounds else "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_yuan": cost_yuan,
        "latency_ms": latency_ms,
        "rounds": len(rounds),
    }