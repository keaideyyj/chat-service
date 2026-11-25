# 运行时镜像：只装运行需要的依赖，不带测试工具
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE：不生成 .pyc，镜像更干净
# PYTHONUNBUFFERED：日志不缓冲，否则容器里看不到实时输出
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 先只拷依赖清单并安装：利用镜像层缓存，
# 只要 requirements.txt 没变，之后改代码都不会重装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 容器内不用 root 跑业务进程
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# slim 镜像没有 curl，用 python 自带的 urllib 做健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

# 必须绑 0.0.0.0：127.0.0.1 只在容器内部可达，宿主机 port 映射会失效
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
