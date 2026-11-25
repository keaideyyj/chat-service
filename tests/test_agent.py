import json

import pytest

from app import agent


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _round(content: str, tool_calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "model": "mock-model",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_yuan": 0.0001,
        "latency_ms": 100,
    }


def test_agent_stops_when_model_answers(monkeypatch):
    script = [
        _round("我先查一下", [_tool_call("search_kb", {"query": "年假"}, "c1")]),
        _round("", [_tool_call("calculate", {"expression": "10*8"}, "c2")]),
        _round("一共 80 小时", []),
    ]
    calls = {"n": 0}

    def fake_chat_round(messages, tools=None):
        item = script[calls["n"]]
        calls["n"] += 1
        return item

    monkeypatch.setattr(agent, "chat_round", fake_chat_round)
    monkeypatch.setattr(agent, "run_tool", lambda name, args: "假结果")

    result = agent.run_agent([{"role": "user", "content": "问题"}])

    assert result["stop_reason"] == "answer"
    assert result["content"] == "一共 80 小时"
    assert [s.type for s in result["steps"]] == ["tool", "tool", "answer"]
    assert result["steps"][0].tool == "search_kb"
    assert result["rounds"] == 3
    assert result["prompt_tokens"] == 30     # 10 × 3 轮

def test_agent_forces_final_answer_at_max_steps(monkeypatch):
    seen_tools: list[bool] = []
    counter = {"n": 0}

    def fake_chat_round(messages, tools=None):
        seen_tools.append(tools is not None)

        if tools is None:                    # ← 兜底那一轮：工具被撤掉了
            return _round("兜底答案", [])

        counter["n"] += 1
        expr = f"{counter['n']}+1"           # ← 每轮参数都不同，避开重复检测
        return _round("", [_tool_call("calculate", {"expression": expr}, f"c{counter['n']}")])

    monkeypatch.setattr(agent, "chat_round", fake_chat_round)
    monkeypatch.setattr(agent, "run_tool", lambda name, args: "3")

    result = agent.run_agent([{"role": "user", "content": "问题"}], max_steps=2)

    assert result["stop_reason"] == "forced_final"
    assert result["content"] == "兜底答案"
    assert seen_tools == [True, True, False]
    assert result["rounds"] == 3
    assert [s.type for s in result["steps"]] == ["tool", "tool", "answer"]

def test_agent_detects_repeated_tool_call(monkeypatch):
    def fake_chat_round(messages, tools=None):
        if tools is None:
            return _round("兜底答案", [])
        return _round("", [_tool_call("get_current_time", {"timezone_offset": 8}, "c-same")])

    monkeypatch.setattr(agent, "chat_round", fake_chat_round)
    monkeypatch.setattr(agent, "run_tool", lambda name, args: "12:00")

    result = agent.run_agent([{"role": "user", "content": "问题"}], max_steps=3)

    assert result["stop_reason"] == "loop_detected"

    rejected = [s for s in result["steps"] if s.result and s.result.startswith("拒绝执行")]
    assert rejected, "应该至少有一次工具调用被拒绝"
    assert result["steps"][0].tool == "get_current_time" 