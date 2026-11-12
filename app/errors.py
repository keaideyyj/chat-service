import re
import uuid

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{4,}"),
    re.compile(r"(?i)(api[_\- ]?key\"?\s*[:=]\s*)[^\s,}\"']+"),
    re.compile(r"\*{2,}[A-Za-z0-9]{2,}"),
]


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def new_ref() -> str:
    return uuid.uuid4().hex[:8]


class AppError(Exception):
    """所有对外业务异常的基类：同时携带状态码、用户文案、内部详情。"""

    status_code: int = 500
    user_message: str = "服务内部错误"

    def __init__(self, detail: str = "", cause: BaseException | None = None) -> None:
        self.detail = redact(detail or (str(cause) if cause else ""))
        self.ref = new_ref()
        super().__init__(self.detail)


class UpstreamError(AppError):
    """上游模型服务故障：超时、限流、5xx、认证失败。"""

    status_code = 502
    user_message = "模型服务暂时不可用，请稍后重试"