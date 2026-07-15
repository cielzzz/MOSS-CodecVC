#!/usr/bin/env sh
set -eu

# 当前默认：在一个页面中对比 Batch-44 原始 r3 与从 step-10000 权重启动的 warm-start r3。
#   original_r3_to_step10660：原始任务，实际记录到 step 10660。
#   warmstart_from_step10000：新任务，训练 step 从 0 重新计数。
# 直接运行：sh scripts/004009_tensorboard_moss_codecvc.sh

# 原始 r3 v1 任务路径（已停止，保留）：
# LOGDIR=/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/ver2_9_5_final_r3_v1_30k/tensorboard

# warm-start r3 v1 单任务路径（保留）：
# LOGDIR=/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k/tensorboard

# 旧 ver2.3 配置（保留，需要时可取消注释后使用）：
# LOGDIR=/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/ver2_3_ctc_clean_textrep5_spm_lora_r16_a32_gbs64/tensorboard \
# PORT=18601 \
# DAEMON=1 \
# KILL_EXISTING=1 \
# sh scripts/004009_tensorboard_moss_codecvc.sh

# 旧 ver2.8 路径（保留）：
# LOGDIR=/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/ver2_8_sideonly_wavlmbnf_codecres_textrep5_lora_r16_a32_gbs64/tensorboard

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
STATUS_FILE="${STATUS_FILE:-${ROOT}/trainset/zh45w_en22w_no_text/STATUS.md}"
ORIGINAL_LOGDIR="${ROOT}/outputs/lora_runs/ver2_9_5_final_r3_v1_30k/tensorboard"
WARMSTART_LOGDIR="${ROOT}/outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k/tensorboard"
COMBINED_VIEW_DIR="${ROOT}/outputs/tensorboard_views/ver2_9_5_final_r3_original_and_warmstart"

if [ -z "${LOGDIR:-}" ]; then
  if [ ! -d "${ORIGINAL_LOGDIR}" ]; then
    echo "ERROR: original r3 TensorBoard directory not found: ${ORIGINAL_LOGDIR}" >&2
    exit 1
  fi
  if [ ! -d "${WARMSTART_LOGDIR}" ]; then
    echo "ERROR: warm-start r3 TensorBoard directory not found: ${WARMSTART_LOGDIR}" >&2
    exit 1
  fi
  mkdir -p "${COMBINED_VIEW_DIR}"
  ln -sfn "${ORIGINAL_LOGDIR}" "${COMBINED_VIEW_DIR}/original_r3_to_step10660"
  ln -sfn "${WARMSTART_LOGDIR}" "${COMBINED_VIEW_DIR}/warmstart_from_step10000"
  LOGDIR="${COMBINED_VIEW_DIR}"
fi

TB_BIN="${TB_BIN:-$(command -v tensorboard 2>/dev/null || true)}"
PORT="${PORT:-18601}"
HOST="${HOST:-0.0.0.0}"
DAEMON="${DAEMON:-1}"
KILL_EXISTING="${KILL_EXISTING:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-45}"

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
echo "  STARTUP_TIMEOUT=${STARTUP_TIMEOUT}"

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
  elapsed=0
  pid=""
  while [ "${elapsed}" -lt "${STARTUP_TIMEOUT}" ]; do
    pid="$(ss -ltnp "( sport = :${PORT} )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u | head -n 1 || true)"
    [ -n "${pid}" ] && break
    sleep 1
    elapsed=$((elapsed + 1))
  done
  if [ -n "${pid}" ]; then
    printf '%s\n' "${pid}" > "${PID_FILE}"
    echo "TensorBoard started: pid=${pid} wait_seconds=${elapsed}"
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
