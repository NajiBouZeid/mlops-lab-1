# ---- Stage 1: builder ----
# Installs uv and resolves/builds the virtual environment from the lockfile.
# Only pyproject.toml + uv.lock are copied here, so this layer only
# rebuilds when a dependency actually changes.

FROM python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9 AS builder
# Base image pinned to a digest: the python:3.11-slim tag was updated upstream
# between two of my builds, which invalidated the whole build cache.

RUN pip install --no-cache-dir uv

WORKDIR /app
ENV UV_HTTP_TIMEOUT=300
# Longer download timeout: uv's default 30s timed out on slow connection.

COPY pyproject.toml uv.lock ./
# --no-install-project: install only the dependencies, not the project itself,
# because src/ isn't copied yet at this stage. (pyproject.toml also sets
# package = false, since this project is an application, not a library.)
RUN uv sync --frozen --no-dev --no-install-project

# ---- Stage 2: runtime ----
# Slim final image: just the built venv + source code, no build tools.
FROM python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9 AS runtime
# Same pinned base image as the builder stage.
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000

EXPOSE 8000

CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
