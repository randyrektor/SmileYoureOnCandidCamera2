#!/bin/bash
# Stop the processes started by start-app.sh. Only touches PIDs we recorded.
cd "$(dirname "$0")"
QUIET=${1:-}

if [ -f .dev_pids ]; then
  while read -r pid; do
    [ -n "$pid" ] || continue
    # Kill the process group children first (npm -> vite, python -> workers), then the process.
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done < .dev_pids
  sleep 1
  while read -r pid; do
    [ -n "$pid" ] || continue
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
  done < .dev_pids
  rm -f .dev_pids
  [ "$QUIET" = "--quiet" ] || echo "Stopped."
else
  [ "$QUIET" = "--quiet" ] || echo "Nothing to stop (.dev_pids not found)."
fi
rm -f backend.log frontend.log
