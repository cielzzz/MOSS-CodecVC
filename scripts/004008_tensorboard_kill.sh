#!/usr/bin/env sh
set -eu

# 与 004009_tensorboard_moss_codecvc.sh 的默认端口保持一致。
# 旧默认端口：6006
PORT="${PORT:-18601}"

PIDS="$(ss -ltnp "( sport = :${PORT} )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)"

if [ -z "${PIDS}" ]; then
  echo "No process is listening on port ${PORT}."
  exit 0
fi

echo "Killing processes on port ${PORT}: ${PIDS}"
kill ${PIDS}
