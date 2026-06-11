FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY uv.lock* ./
COPY config ./config
COPY src ./src

RUN uv sync --no-dev

ENV ULTIMATE_MEMORY_CONFIG=/app/config/docker.toml
ENV ULTIMATE_MEMORY_MCP_HOST=0.0.0.0
ENV ULTIMATE_MEMORY_MCP_PORT=8788

CMD ["uv", "run", "ultimate-memory-router", "--transport", "streamable-http"]
