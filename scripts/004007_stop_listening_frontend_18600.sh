#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
WEB_ROOT="${WEB_ROOT:-$ROOT/outputs/listening_frontend}"
PORT="${PORT:-18600}"
PID_FILE="${PID_FILE:-$WEB_ROOT/server_${PORT}.pid}"

if [ -f "$PID_FILE" ]; then
  pid=$(cat "$PID_FILE" || true)
  if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid"
    echo "stopped listening frontend: pid=$pid"
  else
    echo "pid file exists but process is not running"
  fi
  rm -f "$PID_FILE"
else
  echo "no pid file: $PID_FILE"
fi
