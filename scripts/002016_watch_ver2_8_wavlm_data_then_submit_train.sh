#!/bin/sh
# Watch Ver2.8 WavLM-BNF prepared data and submit Ver2.8 training once ready.

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_8_prepared}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$ROOT/scripts/002012_submit_ver2_8_sideonly_content_memory_h200_qz.sh}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
STATE_ROOT="${STATE_ROOT:-$ROOT/trainset/qz_jobs/ver2_8_train_after_wavlm_$BATCH_ID}"
POLL_INTERVAL_SECS="${POLL_INTERVAL_SECS:-120}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-0}"
ONCE="${ONCE:-0}"
SUBMIT_DRY_RUN="${SUBMIT_DRY_RUN:-0}"
FORCE_SUBMIT="${FORCE_SUBMIT:-0}"
PREP_JOB_ID="${PREP_JOB_ID:-}"

usage() {
  cat <<EOF
Usage:
  sh scripts/002016_watch_ver2_8_wavlm_data_then_submit_train.sh [--once] [--submit-dry-run] [--force-submit]

Behavior:
  Polls PREPARED_DIR until these files are present and non-empty:
    no_text.train.jsonl
    no_text.valid.jsonl
    text.train.jsonl
    text.valid.jsonl
    summary.json
  Then invokes:
    $SUBMIT_SCRIPT

Env overrides:
  PREPARED_DIR=...          default: $PREPARED_DIR
  STATE_ROOT=...            default: $STATE_ROOT
  POLL_INTERVAL_SECS=...    default: $POLL_INTERVAL_SECS
  MAX_WAIT_SECS=0           0 means wait indefinitely
  PREP_JOB_ID=...           optional QZ preprocessing job id for records
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
LOG_STATE="$STATE_ROOT/state.tsv"
SUBMITTED_MARKER="$STATE_ROOT/submitted.marker"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: another Ver2.8 watcher appears to be running: $LOCK_DIR" >&2
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

check_ready() {
  for rel in no_text.train.jsonl no_text.valid.jsonl text.train.jsonl text.valid.jsonl summary.json; do
    if [ ! -s "$PREPARED_DIR/$rel" ]; then
      echo "missing:$rel"
      return 1
    fi
  done
  python - "$PREPARED_DIR/summary.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "complete":
    raise SystemExit(f"summary_status={summary.get('status')!r}")
split = summary.get("split") or {}
for mode in ("no_text", "text"):
    info = split.get(mode) or {}
    if int(info.get("train_rows") or 0) <= 0 or int(info.get("valid_rows") or 0) <= 0:
        raise SystemExit(f"bad_split_counts:{mode}:{info}")
print("ready")
PY
}

submit_training() {
  if [ "$FORCE_SUBMIT" -ne 1 ] && [ -f "$SUBMITTED_MARKER" ]; then
    echo "ERROR: prior submitted marker exists: $SUBMITTED_MARKER" >&2
    echo "Use --force-submit only if intentionally submitting another training job." >&2
    return 1
  fi

  no_text_rows="$(wc -l < "$PREPARED_DIR/no_text.train.jsonl" | tr -d ' ')"
  text_rows="$(wc -l < "$PREPARED_DIR/text.train.jsonl" | tr -d ' ')"
  valid_no_text_rows="$(wc -l < "$PREPARED_DIR/no_text.valid.jsonl" | tr -d ' ')"
  valid_text_rows="$(wc -l < "$PREPARED_DIR/text.valid.jsonl" | tr -d ' ')"
  echo "[ver2.8-watch] prepared data ready"
  echo "[ver2.8-watch] train no_text=$no_text_rows text=$text_rows; valid no_text=$valid_no_text_rows text=$valid_text_rows"

  if [ "$SUBMIT_DRY_RUN" -eq 1 ]; then
    sh "$SUBMIT_SCRIPT" --dry-run
    return 0
  fi

  sh "$SUBMIT_SCRIPT"
  status=$?
  if [ "$status" -eq 0 ]; then
    latest_submitted_jobs="$(find -L "$ROOT/trainset/qz_jobs" -maxdepth 3 -type f -name submitted_jobs.tsv -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
    train_job_name=""
    train_job_id=""
    train_priority=""
    train_compute_group=""
    if [ -n "$latest_submitted_jobs" ] && [ -f "$latest_submitted_jobs" ]; then
      train_job_name="$(awk -F '\t' 'NR==2 { print $1 }' "$latest_submitted_jobs")"
      train_job_id="$(awk -F '\t' 'NR==2 { print $2 }' "$latest_submitted_jobs")"
      train_priority="$(awk -F '\t' 'NR==2 { print $3 }' "$latest_submitted_jobs")"
      train_compute_group="$(awk -F '\t' 'NR==2 { print $4 }' "$latest_submitted_jobs")"
      cp "$latest_submitted_jobs" "$STATE_ROOT/train_submitted_jobs.tsv"
    fi
    {
      printf 'submitted_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'prep_job_id=%s\n' "$PREP_JOB_ID"
      printf 'prepared_dir=%s\n' "$PREPARED_DIR"
      printf 'no_text_train_rows=%s\n' "$no_text_rows"
      printf 'text_train_rows=%s\n' "$text_rows"
      printf 'no_text_valid_rows=%s\n' "$valid_no_text_rows"
      printf 'text_valid_rows=%s\n' "$valid_text_rows"
      printf 'train_job_name=%s\n' "$train_job_name"
      printf 'train_job_id=%s\n' "$train_job_id"
      printf 'train_priority=%s\n' "$train_priority"
      printf 'train_compute_group=%s\n' "$train_compute_group"
      printf 'submitted_jobs_tsv=%s\n' "$latest_submitted_jobs"
    } > "$SUBMITTED_MARKER"
  fi
  return "$status"
}

echo "=========================================="
echo "Ver2.8 WavLM data -> train watcher"
echo "  PREPARED_DIR=$PREPARED_DIR"
echo "  SUBMIT_SCRIPT=$SUBMIT_SCRIPT"
echo "  STATE_ROOT=$STATE_ROOT"
echo "  POLL_INTERVAL_SECS=$POLL_INTERVAL_SECS"
echo "  MAX_WAIT_SECS=$MAX_WAIT_SECS"
echo "  PREP_JOB_ID=$PREP_JOB_ID"
echo "  ONCE=$ONCE SUBMIT_DRY_RUN=$SUBMIT_DRY_RUN"
echo "=========================================="

start_ts="$(date +%s)"
printf 'timestamp_utc\tstatus\tdetail\n' > "$LOG_STATE"

while :; do
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  detail="$(check_ready 2>&1 || true)"
  if [ "$detail" = "ready" ]; then
    printf '%s\tready\t%s\n' "$now" "$detail" >> "$LOG_STATE"
    submit_training
    exit $?
  fi
  printf '%s\twaiting\t%s\n' "$now" "$detail" >> "$LOG_STATE"
  echo "[$now] waiting: $detail"

  if [ "$ONCE" -eq 1 ]; then
    exit 0
  fi
  if [ "$MAX_WAIT_SECS" -gt 0 ]; then
    elapsed=$(( $(date +%s) - start_ts ))
    if [ "$elapsed" -ge "$MAX_WAIT_SECS" ]; then
      echo "ERROR: max wait exceeded: ${MAX_WAIT_SECS}s" >&2
      exit 1
    fi
  fi
  sleep "$POLL_INTERVAL_SECS"
done
