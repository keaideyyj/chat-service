import pytest
from fastapi.testclient import TestClient

from app import main
from app.errors import UpstreamError
from app.schemas import ChatResponse, Usage


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def _fake_response() -> ChatResponse:
    return ChatResponse(
        content="pong",
        model="deepseek-flash",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost_yuan=0.00006,
        latency_ms=120,
    )


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_chat_ok(client, monkeypatch):
    monkeypatch.setattr(main, "chat", lambda messages: _fake_response())

    r = client.post("/chat", json={"message": "ping"})
    assert r.status_code == 200
    body = r.json()
    assert body["content"] == "pong"
    assert body["usage"]["total_tokens"] == 15


def test_chat_rejects_empty_message(client):
    r = client.post("/chat", json={"message": ""})
    assert r.status_code == 422


def test_chat_upstream_error_is_redacted(client, monkeypatch):
    def boom(messages):
        raise UpstreamError(detail="AuthenticationError: sk-abcdef123456 invalid")

    monkeypatch.setattr(main, "chat", boom)

    r = client.post("/chat", json={"message": "ping"})
    assert r.status_code == 502
    body = r.json()
    assert body["detail"] == "模型服务暂时不可用，请稍后重试"
    assert body["ref"]
    assert "sk-" not in r.text
    assert "Authentication" not in r.text


def test_stream_wire_format(client, monkeypatch):
    async def fake_stream(messages):
        yield {"type": "delta", "text": "你"}
        yield {"type": "delta", "text": "好"}
        yield {"type": "done", "usage": {}, "cost_yuan": 0.0, "latency_ms": 5}

    monkeypatch.setattr(main, "stream_chat", fake_stream)

    r = client.post("/chat/stream", json={"message": "hi"})
    assert r.status_code == 200
    assert "[DONE]" in r.text
    assert r.text.count("data:") >= 4