# syntax=docker/dockerfile:1
#
# 冷却盘管选型核算服务
# 多阶段构建：测试阶段跑全套自动化测试，失败则镜像不产出；
# 最终镜像只带运行依赖，python:3.12-slim 基础镜像。

# ---------- 测试阶段：全套 pytest 不过则构建失败 ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY app ./app
COPY tests ./tests
RUN python -m pytest -q


# ---------- 最终运行镜像：只带运行依赖 ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 非 root 运行
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache

COPY --chown=appuser:appuser app ./app

USER appuser

EXPOSE 8000

# 容器内健康检查直接打健康接口
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/', timeout=2).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
