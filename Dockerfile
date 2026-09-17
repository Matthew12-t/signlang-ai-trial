# syntax=docker/dockerfile:1.7

FROM python:3.10.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime libraries required by OpenCV, MediaPipe, FFmpeg-backed decoding,
# and the CUDA-enabled PyTorch wheel. No compiler or build tool is kept in the
# runtime image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer is intentionally before application source for build cache
# reuse. The checkpoint, upstream SignBart tree, and secrets are never copied.
COPY requirements-docker.txt ./requirements-docker.txt
RUN python -m pip install --no-cache-dir --upgrade pip==25.0.1 \
    && python -m pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY config ./config

RUN groupadd --system --gid 10001 signservice \
    && useradd --system --uid 10001 --gid signservice --home-dir /nonexistent --shell /usr/sbin/nologin signservice \
    && chown -R signservice:signservice /app

USER signservice

EXPOSE 8765

HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8765/health/live', timeout=2); raise SystemExit(0 if response.status == 200 else 1)"

ENTRYPOINT ["python", "-m", "uvicorn", "src.sign.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8765"]
