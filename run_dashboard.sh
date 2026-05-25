#!/usr/bin/env bash
# run_dashboard.sh — Launch Streamlit dashboard
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f ".venv/bin/activate" ] && source ".venv/bin/activate" || true
[ -f "venv/bin/activate"  ] && source "venv/bin/activate"  || true

echo "🚀 Starting GARCH Volatility Dashboard …"
streamlit run app/dashboard.py \
  --server.port "${STREAMLIT_PORT:-8501}" \
  --server.address "0.0.0.0" \
  --browser.gatherUsageStats false \
  "$@"
