#!/usr/bin/env sh
set -eu

:<<EOF
LOGDIR=/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/ver2_3_ctc_clean_textrep5_spm_lora_r16_a32_gbs64/tensorboard \
PORT=18601 \
DAEMON=1 \
KILL_EXISTING=1 \
bash 004009_tensorboard_moss_codecvc.sh
EOF

# ver 2.8
# /inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/ver2_8_sideonly_wavlmbnf_codecres_textrep5_lora_r16_a32_gbs64/tensorboard

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
STATUS_FILE="${STATUS_FILE:-${ROOT}/trainset/zh45w_en22w_no_text/STATUS.md}"
TB_BIN="${TB_BIN:-$(command -v tensorboard 2>/dev/null || true)}"
PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"
DAEMON="${DAEMON:-0}"
KILL_EXISTING="${KILL_EXISTING:-1}"

if [ -z "${TB_BIN}" ]; then
  if [ -x /inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/tensorboard ]; then
    TB_BIN=/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/tensorboard
  elif [ -x /opt/conda/envs/speech/bin/tensorboard ]; then
    TB_BIN=/opt/conda/envs/speech/bin/tensorboard
  else
    echo "ERROR: tensorboard binary not found. Set TB_BIN=/path/to/tensorboard." >&2
    exit 1
  fi
fi

read_status_value() {
  key="$1"
  file="$2"
  [ -f "$file" ] || return 0
  awk -v k="$key" '
    $0 ~ "^" k "=" {
      sub("^" k "=", "", $0)
      print $0
      exit
    }
  ' "$file"
}

if [ -z "${LOGDIR:-}" ]; then
  LOGDIR="$(read_status_value VER2_TRAIN_TENSORBOARD_DIR "$STATUS_FILE" || true)"
fi
if [ -z "${LOGDIR:-}" ] || [ ! -d "${LOGDIR}" ]; then
  LOGDIR="$(find "${ROOT}/outputs/lora_runs" -type f -name 'events.out.tfevents*' -printf '%T@ %h\n' 2>/dev/null | sort -rn | awk 'NR == 1 { $1=""; sub(/^ /, ""); print; exit }')"
fi
if [ -z "${LOGDIR:-}" ] || [ ! -d "${LOGDIR}" ]; then
  echo "ERROR: TensorBoard logdir not found. Set LOGDIR=/path/to/tensorboard." >&2
  exit 1
fi

PID_FILE="${PID_FILE:-${LOGDIR%/}/../tensorboard_server_${PORT}.pid}"
TB_LOG="${TB_LOG:-${LOGDIR%/}/../tensorboard_server_${PORT}.log}"

kill_existing_port() {
  pids="$(ss -ltnp "( sport = :${PORT} )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u || true)"
  [ -n "$pids" ] || return 0
  echo "Killing existing process(es) on port ${PORT}: ${pids}"
  kill $pids 2>/dev/null || true
  sleep 1
}

echo "TensorBoard"
echo "  TB_BIN=${TB_BIN}"
echo "  LOGDIR=${LOGDIR}"
echo "  HOST=${HOST}"
echo "  PORT=${PORT}"
echo "  DAEMON=${DAEMON}"

if [ "${DAEMON}" = "1" ]; then
  if [ "${KILL_EXISTING}" = "1" ]; then
    kill_existing_port
  fi
  : > "${TB_LOG}"
  setsid -f "${TB_BIN}" \
    --logdir "${LOGDIR}" \
    --port "${PORT}" \
    --host "${HOST}" \
    --reload_interval 10 \
    --load_fast=false \
    >> "${TB_LOG}" 2>&1
  sleep 5
  pid="$(ss -ltnp "( sport = :${PORT} )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u | head -n 1 || true)"
  if [ -n "${pid}" ]; then
    printf '%s\n' "${pid}" > "${PID_FILE}"
    echo "TensorBoard started: pid=${pid}"
    echo "  log=${TB_LOG}"
    echo "  pid_file=${PID_FILE}"
    curl -I --max-time 5 "http://127.0.0.1:${PORT}/" 2>&1 | head -n 12 || true
    exit 0
  fi
  echo "ERROR: TensorBoard did not start on port ${PORT}. Log tail:" >&2
  tail -80 "${TB_LOG}" >&2 || true
  exit 1
fi

exec "${TB_BIN}" \
  --logdir "${LOGDIR}" \
  --port "${PORT}" \
  --host "${HOST}" \
  --reload_interval 10 \
  --load_fast=false
