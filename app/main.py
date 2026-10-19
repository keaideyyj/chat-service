"""HTTP 服务入口。"""

from fastapi import FastAPI

app = FastAPI(title="chat-service")

@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}