import json
import logging
import logging.handlers
import sys
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """把日志打成一行 JSON，方便将来被 ELK / Loki 直接按字段查询。"""

    EXTRA_FIELDS = ("ref", "path", "status", "model", "latency_ms", "cost_yuan")

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for field in self.EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO, log_dir: str = "logs") -> None:
    formatter = JsonFormatter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)
    root.setLevel(level)