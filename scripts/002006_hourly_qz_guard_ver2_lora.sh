#!/bin/sh
# Hourly guard for the Ver2 data-prep and LoRA-training QZ jobs.
#
# It records qzcli status/log snapshots and keeps the READY_FOR_VER2_TRAIN
# submit monitor alive. Human/Codex intervention is still required for code fixes.
#
# Usage:
#   setsid sh scripts/002006_hourly_qz_guard_ver2_lora.sh > trainset/zh45w_en22w_no_text/qz_jobs/hourly_guard/guard.log 2>&1 < /dev/null &

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
DATASET_NAME="${DATASET_NAME:-zh45w_en22w_no_text}"
STATUS_FILE="${STATUS_FILE:-$ROOT/trainset/$DATASET_NAME/STATUS.md}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
SUBMIT_MONITOR_SCRIPT="${SUBMIT_MONITOR_SCRIPT:-$ROOT/scripts/002005_wait_ready_and_submit_ver2_lora_68w_h200_qz.sh}"
SUBMIT_MONITOR_STATE="${SUBMIT_MONITOR_STATE:-$ROOT/trainset/$DATASET_NAME/qz_jobs/ver2_train_monitor}"
GUARD_ROOT="${GUARD_ROOT:-$ROOT/trainset/$DATASET_NAME/qz_jobs/hourly_guard}"
POLL_INTERVAL_SECS="${POLL_INTERVAL_SECS:-3600}"
MAX_CHECKS="${MAX_CHECKS:-10}"
STOP_FILE="${STOP_FILE:-$GUARD_ROOT/STOP}"
QZCLI_STATUS_TIMEOUT_SECS="${QZCLI_STATUS_TIMEOUT_SECS:-120}"
QZCLI_LOG_TIMEOUT_SECS="${QZCLI_LOG_TIMEOUT_SECS:-180}"

mkdir -p "$GUARD_ROOT/status" "$GUARD_ROOT/logs"

LOCK_DIR="$GUARD_ROOT/lock"
PID_FILE="$GUARD_ROOT/guard.pid"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: hourly guard already appears to be running: $LOCK_DIR" >&2
  [ -f "$PID_FILE" ] && echo "Existing pid: $(cat "$PID_FILE" 2>/dev/null || true)" >&2
  exit 1
fi
cleanup() {
  rm -f "$PID_FILE"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
printf '%s\n' "$$" > "$PID_FILE"

if [ ! -x "$QZCLI" ]; then
  echo "ERROR: qzcli wrapper is not executable: $QZCLI" >&2
  exit 1
fi
if [ ! -x "$SUBMIT_MONITOR_SCRIPT" ]; then
  echo "ERROR: submit monitor script is not executable: $SUBMIT_MONITOR_SCRIPT" >&2
  exit 1
fi

extract_status_value() {
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

run_qzcli_status() {
  timeout "${QZCLI_STATUS_TIMEOUT_SECS}s" env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy "$QZCLI" "$@"
}

run_qzcli_logs() {
  timeout "${QZCLI_LOG_TIMEOUT_SECS}s" env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy "$QZCLI" "$@"
}

qz_login() {
  timeout "${QZCLI_STATUS_TIMEOUT_SECS}s" env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy "$QZCLI" login
}

job_failed_by_status_file() {
  file="$1"
  grep -Eqi '失败|failed|fail|error|异常|崩溃|killed|oom|out of memory' "$file"
}

snapshot_job() {
  label="$1"
  job_id="$2"
  ts="$3"
  [ -n "$job_id" ] || return 0

  status_out="$GUARD_ROOT/status/${ts}_${label}_${job_id}_status.txt"
  logs_out="$GUARD_ROOT/logs/${ts}_${label}_${job_id}_tail500.log"
  echo "[guard] qz status: label=$label job_id=$job_id"
  set +e
  run_qzcli_status status "$job_id" > "$status_out" 2>&1
  status_code=$?
  set -e
  cat "$status_out"
  if [ "$status_code" -ne 0 ]; then
    echo "[guard] WARNING: qzcli status failed for $label $job_id, exit=$status_code"
    if grep -q 'Cookie 已过期或无效' "$status_out"; then
      login_out="$GUARD_ROOT/status/${ts}_${label}_${job_id}_login_retry.txt"
      echo "[guard] qz cookie expired; running qzcli login and retrying status"
      set +e
      qz_login > "$login_out" 2>&1
      login_code=$?
      if [ "$login_code" -eq 0 ]; then
        run_qzcli_status status "$job_id" > "$status_out" 2>&1
        status_code=$?
      fi
      set -e
      cat "$login_out"
      cat "$status_out"
      if [ "$status_code" -ne 0 ]; then
        echo "[guard] WARNING: qzcli status retry failed for $label $job_id, exit=$status_code"
      fi
    fi
  fi

  if job_failed_by_status_file "$status_out"; then
    echo "[guard] FAILURE suspected for $label $job_id; saving logs to $logs_out"
    set +e
    run_qzcli_logs logs --tail 500 "$job_id" > "$logs_out" 2>&1
    log_code=$?
    set -e
    if [ "$log_code" -ne 0 ]; then
      echo "[guard] WARNING: qzcli logs failed for $label $job_id, exit=$log_code"
      if grep -q 'Cookie 已过期或无效' "$logs_out"; then
        login_out="$GUARD_ROOT/logs/${ts}_${label}_${job_id}_login_retry.txt"
        echo "[guard] qz cookie expired; running qzcli login and retrying logs"
        set +e
        qz_login > "$login_out" 2>&1
        login_code=$?
        if [ "$login_code" -eq 0 ]; then
          run_qzcli_logs logs --tail 500 "$job_id" > "$logs_out" 2>&1
          log_code=$?
        fi
        set -e
        cat "$login_out"
        if [ "$log_code" -ne 0 ]; then
          echo "[guard] WARNING: qzcli logs retry failed for $label $job_id, exit=$log_code"
        fi
      fi
    fi
    {
      printf 'failed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'label=%s\n' "$label"
      printf 'job_id=%s\n' "$job_id"
      printf 'status_out=%s\n' "$status_out"
      printf 'logs_out=%s\n' "$logs_out"
    } > "$GUARD_ROOT/ACTION_REQUIRED_${label}_${job_id}.txt"
  fi
}

find_train_job_id() {
  status_job_id="$(extract_status_value VER2_TRAIN_JOB_ID "$STATUS_FILE" || true)"
  if [ -n "$status_job_id" ]; then
    printf '%s\n' "$status_job_id"
    return 0
  fi
  marker="$SUBMIT_MONITOR_STATE/submitted.marker"
  if [ -f "$marker" ]; then
    job_id="$(awk -F= '$1=="train_job_id" { print $2; exit }' "$marker")"
    if [ -n "$job_id" ]; then
      printf '%s\n' "$job_id"
      return 0
    fi
  fi
  find -L "$ROOT/trainset/qz_jobs" -maxdepth 2 -type f -name submitted_jobs.tsv -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn \
    | while read -r _ path; do
        awk -F '\t' 'NR > 1 && $1 ~ /^(moss-)?codecvc-ver2-68w-train-lora-/ { print $2; exit }' "$path"
      done \
    | awk 'NF { print; exit }'
}

submit_monitor_alive() {
  pid_file="$SUBMIT_MONITOR_STATE/nohup.pid"
  [ -f "$pid_file" ] || return 1
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  ps -p "$pid" >/dev/null 2>&1
}

ensure_submit_monitor() {
  if [ -f "$SUBMIT_MONITOR_STATE/submitted.marker" ]; then
    echo "[guard] submit monitor already completed: $SUBMIT_MONITOR_STATE/submitted.marker"
    return 0
  fi
  if submit_monitor_alive; then
    echo "[guard] submit monitor alive: pid=$(cat "$SUBMIT_MONITOR_STATE/nohup.pid")"
    return 0
  fi
  echo "[guard] submit monitor is not alive; restarting"
  rm -rf "$SUBMIT_MONITOR_STATE/lock" "$SUBMIT_MONITOR_STATE/monitor.pid"
  mkdir -p "$SUBMIT_MONITOR_STATE"
  : > "$SUBMIT_MONITOR_STATE/monitor.log"
  setsid sh "$SUBMIT_MONITOR_SCRIPT" > "$SUBMIT_MONITOR_STATE/monitor.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$SUBMIT_MONITOR_STATE/nohup.pid"
  sleep 2
  if submit_monitor_alive; then
    echo "[guard] submit monitor restarted: pid=$(cat "$SUBMIT_MONITOR_STATE/nohup.pid")"
  else
    echo "[guard] ERROR: submit monitor failed to restart"
    return 1
  fi
}

echo "=========================================="
echo "Hourly QZ guard"
echo "  ROOT=$ROOT"
echo "  STATUS_FILE=$STATUS_FILE"
echo "  GUARD_ROOT=$GUARD_ROOT"
echo "  POLL_INTERVAL_SECS=$POLL_INTERVAL_SECS"
echo "  MAX_CHECKS=$MAX_CHECKS"
echo "  STOP_FILE=$STOP_FILE"
echo "  QZCLI_STATUS_TIMEOUT_SECS=$QZCLI_STATUS_TIMEOUT_SECS"
echo "  QZCLI_LOG_TIMEOUT_SECS=$QZCLI_LOG_TIMEOUT_SECS"
echo "=========================================="

i=1
while [ "$i" -le "$MAX_CHECKS" ]; do
  if [ -f "$STOP_FILE" ]; then
    echo "[guard] stop file detected: $STOP_FILE"
    exit 0
  fi

  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  echo "=========================================="
  echo "[guard] check $i/$MAX_CHECKS at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  ensure_submit_monitor || true

  stage6_status="$(extract_status_value STAGE6_STATUS "$STATUS_FILE" || true)"
  ready="$(extract_status_value READY_FOR_VER2_TRAIN "$STATUS_FILE" || true)"
  stage6_job_id="$(extract_status_value STAGE6_JOB_ID "$STATUS_FILE" || true)"
  final_jsonl="$(extract_status_value FINAL_TRAIN_JSONL "$STATUS_FILE" || true)"
  train_job_id="$(find_train_job_id || true)"

  echo "[guard] STATUS.md: STAGE6_STATUS=${stage6_status:-unknown} READY_FOR_VER2_TRAIN=${ready:-unknown}"
  echo "[guard] STATUS.md: STAGE6_JOB_ID=${stage6_job_id:-unknown}"
  echo "[guard] STATUS.md: FINAL_TRAIN_JSONL=${final_jsonl:-unknown}"
  if [ -n "$final_jsonl" ] && [ -f "$final_jsonl.done.json" ]; then
    done_written="$(awk -F ':' '/"written"/ { gsub(/[, ]/, "", $2); print $2; exit }' "$final_jsonl.done.json")"
    echo "[guard] final_jsonl_rows=${done_written:-unknown}"
    echo "[guard] final_done_marker=present"
  elif [ -n "$final_jsonl" ] && [ -f "$final_jsonl" ]; then
    echo "[guard] final_jsonl=present"
    echo "[guard] final_done_marker=missing"
  else
    echo "[guard] final_done_marker=missing"
  fi

  snapshot_job "stage6" "$stage6_job_id" "$ts"
  if [ -n "$train_job_id" ]; then
    snapshot_job "ver2_train" "$train_job_id" "$ts"
  else
    echo "[guard] ver2 training job not submitted yet"
  fi

  if [ -f "$SUBMIT_MONITOR_STATE/submitted.marker" ]; then
    echo "[guard] submit monitor submitted marker:"
    sed -n '1,20p' "$SUBMIT_MONITOR_STATE/submitted.marker" 2>/dev/null || true
  else
    echo "[guard] submit monitor tail:"
    tail -20 "$SUBMIT_MONITOR_STATE/monitor.log" 2>/dev/null || true
  fi

  if [ "$i" -ge "$MAX_CHECKS" ]; then
    break
  fi
  i=$((i + 1))
  sleep "$POLL_INTERVAL_SECS"
done

echo "[guard] completed $MAX_CHECKS checks"
