# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# GARCH Volatility Platform — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build:
#   stage 1 (builder)  – install Python dependencies into a virtual-env
#   stage 2 (runtime)  – copy only the venv and source; keeps the image lean
#
# Build:
#   docker build -t garch-platform .
#
# Run dashboard (default):
#   docker run -p 8501:8501 garch-platform
#
# Run API:
#   docker run -p 8000:8000 garch-platform api
#
# Environment variables:
#   STREAMLIT_PORT  (default 8501)
#   API_PORT        (default 8000)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies needed by scipy / arch / weasyprint (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only the venv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project source
COPY src/        ./src/
COPY app/        ./app/
COPY main.py     ./main.py

# Create persistent directories (mounted as volumes in production)
RUN mkdir -p data outputs/plots outputs/reports

# ── Default: launch Streamlit dashboard ──────────────────────────────────────
EXPOSE 8501 8000

# Entry-point script selects dashboard vs. API based on the CMD argument
COPY --chmod=755 run_dashboard.sh ./run_dashboard.sh
COPY --chmod=755 run_api.sh       ./run_api.sh

ENTRYPOINT ["bash", "-c"]
CMD ["streamlit run app/dashboard.py --server.port ${STREAMLIT_PORT:-8501} --server.address 0.0.0.0 --browser.gatherUsageStats false"]

# Override for API: docker run … api
# The "api" argument triggers the api CMD via a helper script — use:
#   docker run -p 8000:8000 garch-platform "uvicorn app.api:app --host 0.0.0.0 --port 8000"

LABEL maintainer="GARCH Volatility Platform" \
      description="GARCH family volatility modelling, forecasting, and risk dashboard" \
      version="1.0.0"
