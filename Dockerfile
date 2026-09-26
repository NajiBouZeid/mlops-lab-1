# ---- Stage 1: builder ----
# Installs uv and resolves/builds the virtual environment from the lockfile.
# Only pyproject.toml + uv.lock are copied here, so this layer only
# rebuilds when a dependency actually changes.
FROM python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9 AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
ENV UV_HTTP_TIMEOUT=300


COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ---- Stage 2: runtime ----
# Slim final image: just the built venv + source code, no build tools.
FROM python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9 AS runtime
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000

EXPOSE 8000

CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
