#!/bin/bash
# Start the backend (uvicorn) and frontend (Vite) for local development.
# Ctrl+C stops both. Logs go to backend.log / frontend.log.
set -euo pipefail
cd "$(dirname "$0")"

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
BACKEND_PORT="${REACT_BABY_PORT:-8000}"
FRONTEND_PORT=5173

if [ ! -x backend/venv/bin/python ]; then
  echo -e "${RED}No backend venv. Run:${NC} cd backend && python3.12 -m venv venv && venv/bin/pip install -e '.[dev]'"
  exit 1
fi

# Stop anything left over from a previous run of *this* script.
./stop-app.sh --quiet

echo -e "${BLUE}Starting backend on :${BACKEND_PORT}…${NC}"
(
  cd backend
  # Quiet MediaPipe's native logging (glog / TF Lite XNNPACK / clearcut telemetry).
  GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=3 \
    exec venv/bin/python -m react_baby.main
) > backend.log 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > .dev_pids

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${BACKEND_PORT}/api/health" > /dev/null; then break; fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo -e "${RED}Backend exited early. Last lines of backend.log:${NC}"; tail -20 backend.log; exit 1
  fi
  sleep 0.5
done
if ! curl -sf "http://127.0.0.1:${BACKEND_PORT}/api/health" > /dev/null; then
  echo -e "${RED}Backend did not become healthy in 60s. See backend.log.${NC}"; exit 1
fi
echo -e "${GREEN}Backend ready.${NC}"

echo -e "${BLUE}Starting frontend on :${FRONTEND_PORT}…${NC}"
(
  cd frontend
  [ -d node_modules ] || npm install --no-audit --no-fund
  exec npm run dev -- --port "$FRONTEND_PORT" --strictPort
) > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" >> .dev_pids

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${FRONTEND_PORT}" > /dev/null && break
  sleep 0.5
done

echo -e "${GREEN}Running.${NC}  UI: http://localhost:${FRONTEND_PORT}   API: http://localhost:${BACKEND_PORT}/docs"
open "http://localhost:${FRONTEND_PORT}" 2>/dev/null || true

cleanup() { echo; ./stop-app.sh; exit 0; }
trap cleanup INT TERM
tail -n +1 -f backend.log frontend.log
