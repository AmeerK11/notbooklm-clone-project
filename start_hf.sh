#!/usr/bin/env bash
set -euo pipefail

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"
STREAMLIT_PORT="${PORT:-7860}"

export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT}}"
export STORAGE_BASE_DIR="${STORAGE_BASE_DIR:-/data}"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting FastAPI backend on ${BACKEND_HOST}:${BACKEND_PORT}"
uvicorn app:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" --workers 1 &
BACKEND_PID=$!

echo "Waiting for backend health endpoint..."
for _ in {1..60}; do
  if curl -fsS "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    echo "Backend is ready."
    break
  fi
  sleep 1
done

echo "Starting Streamlit frontend on ${STREAMLIT_HOST}:${STREAMLIT_PORT}"
exec streamlit run frontend/app.py \
  --server.address "${STREAMLIT_HOST}" \
  --server.port "${STREAMLIT_PORT}" \
  --server.headless true \
  --browser.gatherUsageStats false
