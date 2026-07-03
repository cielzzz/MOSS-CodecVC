#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
WEB_ROOT="${WEB_ROOT:-$ROOT/outputs/listening_frontend}"
PORT="${PORT:-18600}"
HOST="${HOST:-0.0.0.0}"
PID_FILE="${PID_FILE:-$WEB_ROOT/server_${PORT}.pid}"
LOG_FILE="${LOG_FILE:-$WEB_ROOT/server_${PORT}.log}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"

mkdir -p "$WEB_ROOT"

if [ -f "$PID_FILE" ]; then
  old_pid=$(cat "$PID_FILE" || true)
  if [ -n "$old_pid" ] && kill -0 "$old_pid" >/dev/null 2>&1; then
    echo "listening frontend already running: pid=$old_pid port=$PORT"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

cd "$WEB_ROOT"
setsid -f "$PY" -m http.server --bind "$HOST" "$PORT" >"$LOG_FILE" 2>&1
sleep 0.5
pid=$(pgrep -f "$PY -m http.server --bind $HOST $PORT" | head -n 1 || true)
if [ -z "$pid" ]; then
  echo "failed to start listening frontend; log follows:" >&2
  cat "$LOG_FILE" >&2 || true
  exit 1
fi
echo "$pid" > "$PID_FILE"
echo "started listening frontend: pid=$pid port=$PORT root=$WEB_ROOT"
echo "log: $LOG_FILE"
