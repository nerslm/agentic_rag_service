#!/usr/bin/env bash
set -euo pipefail

OLLAMA_PORT="${OLLAMA_PORT:-11434}"
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:${OLLAMA_PORT}}"

RAG_OLLAMA_MODEL="${RAG_OLLAMA_MODEL:-qwen3:4b-instruct-2507-q4_K_M}"
PULL_ON_START="${OLLAMA_PULL_ON_START:-1}"

echo "[startup] starting ollama on ${OLLAMA_HOST}"
ollama serve >/tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

cleanup() {
  if kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[startup] waiting for ollama readiness..."
READY=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [ "${READY}" != "1" ]; then
  echo "[startup] ollama did not become ready in time"
  exit 1
fi

if [ "${PULL_ON_START}" = "1" ]; then
  echo "[startup] pulling model: ${RAG_OLLAMA_MODEL}"
  curl -fsS "http://127.0.0.1:${OLLAMA_PORT}/api/pull" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${RAG_OLLAMA_MODEL}\",\"stream\":false}" >/dev/null
fi

echo "[startup] launching rag api..."
cd /app/project
exec python api_server.py
