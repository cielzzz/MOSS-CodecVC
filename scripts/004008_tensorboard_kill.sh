#!/usr/bin/env sh
set -eu

PORT="${PORT:-6006}"

PIDS="$(ss -ltnp "( sport = :${PORT} )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)"

if [ -z "${PIDS}" ]; then
  echo "No process is listening on port ${PORT}."
  exit 0
fi

echo "Killing processes on port ${PORT}: ${PIDS}"
kill ${PIDS}
