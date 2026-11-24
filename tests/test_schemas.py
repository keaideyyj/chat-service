"""接口契约测试：不联网、不花钱、几毫秒跑完。

测的是 pydantic 模型层面的规则，也就是你对调用方的第一层承诺。
以后改 schemas.py 时，这些测试会立刻告诉你有没有把契约改坏。
"""

import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest, Message


def test_chat_request_accepts_minimal_payload():
    """只传 message 应该能构造成功，且可选字段有正确的默认值。

    为什么重要：默认值写错是静默 bug —— 比如漏了 default_factory，
    所有请求会共享同一个列表对象，一个请求往里加东西会影响其他请求。
    """
    req = ChatRequest(message="你好")
    assert req.message == "你好"       # 必填字段原样保留，没被改写
    assert req.history == []           # 缺省是空列表，不是 None
    assert req.system_prompt is None   # 可选字段缺省为 None


def test_chat_request_rejects_empty_message():
    """空字符串必须被拒绝（约定见 schemas.py 的 min_length=1）。

    为什么重要：空消息会白白消耗一次模型调用，等于让调用方免费烧你的钱。
    """
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_rejects_overlong_message():
    """超长文本必须被拒绝（max_length=4000）。

    为什么重要：这是成本防线。没有它，一次请求就能塞进几十万 token，
    既烧钱又可能超出模型上下文限制，报一个难懂的 400。
    """
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 4001)


def test_message_role_must_be_known():
    """role 只能是 system / user / assistant（见 Literal 声明）。

    为什么重要：把 role 拼错（比如 "asistant"）不会报错，
    但模型会静默忽略那条消息 —— 排查起来非常费时间。
    """
    with pytest.raises(ValidationError):
        Message(role="robot", content="hi")