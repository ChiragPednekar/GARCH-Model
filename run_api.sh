#!/usr/bin/env bash
# run_api.sh — Launch FastAPI server
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f ".venv/bin/activate" ] && source ".venv/bin/activate" || true
[ -f "venv/bin/activate"  ] && source "venv/bin/activate"  || true

API_PORT="${API_PORT:-8000}"
echo "🚀 Starting GARCH Volatility API on port ${API_PORT} …"
echo "   Docs  → http://localhost:${API_PORT}/docs"
echo "   ReDoc → http://localhost:${API_PORT}/redoc"

uvicorn app.api:app --host 0.0.0.0 --port "${API_PORT}" --reload "$@"
