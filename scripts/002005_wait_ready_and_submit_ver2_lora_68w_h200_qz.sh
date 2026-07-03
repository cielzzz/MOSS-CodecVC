#!/bin/sh
# Poll trainset STATUS.md and submit Ver2 LoRA training when the current
# sentinel value READY_FOR_VER2_TRAIN is 1.
#
# Usage:
#   sh scripts/002005_wait_ready_and_submit_ver2_lora_68w_h200_qz.sh
#   sh scripts/002005_wait_ready_and_submit_ver2_lora_68w_h200_qz.sh --once
#   setsid sh scripts/002005_wait_ready_and_submit_ver2_lora_68w_h200_qz.sh > monitor.log 2>&1 < /dev/null &

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
DATASET_NAME="${DATASET_NAME:-zh45w_en22w_no_text}"
STATUS_FILE="${STATUS_FILE:-$ROOT/trainset/$DATASET_NAME/STATUS.md}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$ROOT/scripts/002004_submit_ver2_lora_68w_h200_qz.sh}"
STATE_ROOT="${STATE_ROOT:-$ROOT/trainset/$DATASET_NAME/qz_jobs/ver2_train_monitor}"
POLL_INTERVAL_SECS="${POLL_INTERVAL_SECS:-60}"
ONCE="${ONCE:-0}"
SUBMIT_DRY_RUN="${SUBMIT_DRY_RUN:-0}"
FORCE_SUBMIT="${FORCE_SUBMIT:-0}"

usage() {
  cat <<EOF
Usage:
  sh scripts/002005_wait_ready_and_submit_ver2_lora_68w_h200_qz.sh [--once] [--submit-dry-run] [--force-submit]

Behavior:
  Polls STATUS_FILE until the first READY_FOR_VER2_TRAIN sentinel value is:
    READY_FOR_VER2_TRAIN=1
  Then reads FINAL_TRAIN_JSONL from STATUS_FILE and invokes:
    $SUBMIT_SCRIPT

Env overrides:
  STATUS_FILE=...          default: $STATUS_FILE
  POLL_INTERVAL_SECS=...   default: $POLL_INTERVAL_SECS
  STATE_ROOT=...           default: $STATE_ROOT
  FORCE_SUBMIT=1           ignore prior submitted marker
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --once)
      ONCE=1
      ;;
    --submit-dry-run)
      SUBMIT_DRY_RUN=1
      ;;
    --force-submit)
      FORCE_SUBMIT=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$STATE_ROOT"
LOCK_DIR="$STATE_ROOT/lock"
PID_FILE="$STATE_ROOT/monitor.pid"
SUBMITTED_MARKER="$STATE_ROOT/submitted.marker"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: another monitor appears to be running: $LOCK_DIR" >&2
  if [ -f "$PID_FILE" ]; then
    echo "Existing pid: $(cat "$PID_FILE" 2>/dev/null || true)" >&2
  fi
  exit 1
fi
cleanup() {
  rm -f "$PID_FILE"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
printf '%s\n' "$$" > "$PID_FILE"

if [ ! -x "$SUBMIT_SCRIPT" ]; then
  echo "ERROR: submit script is not executable: $SUBMIT_SCRIPT" >&2
  exit 1
fi

extract_status_value() {
  key="$1"
  file="$2"
  awk -v k="$key" '
    $0 ~ "^" k "=" {
      sub("^" k "=", "", $0)
      print $0
      exit
    }
  ' "$file"
}

submit_training() {
  final_jsonl="$(extract_status_value FINAL_TRAIN_JSONL "$STATUS_FILE")"
  if [ -z "$final_jsonl" ]; then
    echo "ERROR: FINAL_TRAIN_JSONL is missing in $STATUS_FILE" >&2
    return 1
  fi
  if [ ! -f "$final_jsonl" ]; then
    echo "ERROR: FINAL_TRAIN_JSONL does not exist: $final_jsonl" >&2
    return 1
  fi
  if [ ! -s "$final_jsonl.done.json" ]; then
    echo "ERROR: done marker is missing or empty: $final_jsonl.done.json" >&2
    return 1
  fi

  row_count="$(wc -l < "$final_jsonl" | tr -d ' ')"
  echo "[monitor] READY_FOR_VER2_TRAIN=1 detected"
  echo "[monitor] final_jsonl=$final_jsonl"
  echo "[monitor] final_rows=$row_count"

  if [ "$FORCE_SUBMIT" -ne 1 ] && [ -f "$SUBMITTED_MARKER" ]; then
    echo "ERROR: prior submitted marker exists: $SUBMITTED_MARKER" >&2
    echo "Use --force-submit only if you intentionally want to submit another training job." >&2
    return 1
  fi

  if [ "$SUBMIT_DRY_RUN" -eq 1 ]; then
    TRAIN_JSONL="$final_jsonl" "$SUBMIT_SCRIPT" --dry-run
    return 0
  fi

  TRAIN_JSONL="$final_jsonl" "$SUBMIT_SCRIPT"
  status=$?
  if [ "$status" -eq 0 ]; then
    latest_submitted_jobs="$(find -L "$ROOT/trainset/qz_jobs" -maxdepth 2 -type f -name submitted_jobs.tsv -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
    train_job_name=""
    train_job_id=""
    train_compute_group=""
    if [ -n "$latest_submitted_jobs" ] && [ -f "$latest_submitted_jobs" ]; then
      train_job_name="$(awk -F '\t' 'NR==2 { print $1 }' "$latest_submitted_jobs")"
      train_job_id="$(awk -F '\t' 'NR==2 { print $2 }' "$latest_submitted_jobs")"
      train_compute_group="$(awk -F '\t' 'NR==2 { print $3 }' "$latest_submitted_jobs")"
      cp "$latest_submitted_jobs" "$STATE_ROOT/train_submitted_jobs.tsv"
    fi
    {
      printf 'submitted_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'status_file=%s\n' "$STATUS_FILE"
      printf 'train_jsonl=%s\n' "$final_jsonl"
      printf 'row_count=%s\n' "$row_count"
      printf 'train_job_name=%s\n' "$train_job_name"
      printf 'train_job_id=%s\n' "$train_job_id"
      printf 'train_compute_group=%s\n' "$train_compute_group"
      printf 'submitted_jobs_tsv=%s\n' "$latest_submitted_jobs"
    } > "$SUBMITTED_MARKER"
  fi
  return "$status"
}

echo "=========================================="
echo "Ver2 train monitor"
echo "  STATUS_FILE=$STATUS_FILE"
echo "  SUBMIT_SCRIPT=$SUBMIT_SCRIPT"
echo "  STATE_ROOT=$STATE_ROOT"
echo "  POLL_INTERVAL_SECS=$POLL_INTERVAL_SECS"
echo "  ONCE=$ONCE"
echo "  SUBMIT_DRY_RUN=$SUBMIT_DRY_RUN"
echo "=========================================="

while :; do
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ ! -f "$STATUS_FILE" ]; then
    echo "[$now] waiting: missing STATUS_FILE=$STATUS_FILE"
  else
    stage6="$(extract_status_value STAGE6_STATUS "$STATUS_FILE" || true)"
    ready="$(extract_status_value READY_FOR_VER2_TRAIN "$STATUS_FILE" || true)"
    job_id="$(extract_status_value STAGE6_JOB_ID "$STATUS_FILE" || true)"
    if [ "$ready" = "1" ]; then
      submit_training
      exit $?
    fi
    echo "[$now] waiting: STAGE6_STATUS=${stage6:-unknown} READY_FOR_VER2_TRAIN=${ready:-unknown} STAGE6_JOB_ID=${job_id:-unknown}"
  fi

  if [ "$ONCE" -eq 1 ]; then
    exit 0
  fi
  sleep "$POLL_INTERVAL_SECS"
done
