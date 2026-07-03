#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
if [ ! -x "$PY" ]; then
  PY=python
fi

DATASET_NAME="${DATASET_NAME:-zh45w_en22w_no_text}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
INPUT_JSONL="${INPUT_JSONL:-$DATASET_ROOT/manifests/vc_manifest.$DATASET_NAME.jsonl}"
CONFIG="${CONFIG:-$ROOT/configs/remote_full.yaml}"
if [ ! -f "$CONFIG" ]; then
  CONFIG="$ROOT/configs/default.yaml"
fi

ENCODED_JSONL="${ENCODED_JSONL:-$DATASET_ROOT/encoded/vc_manifest.$DATASET_NAME.encoded.jsonl}"
CODES_DIR="${CODES_DIR:-$DATASET_ROOT/codes}"
SFT_JSONL="${SFT_JSONL:-$DATASET_ROOT/sft/moss_codecvc_sft.$DATASET_NAME.jsonl}"
SPEAKER_PLAN_JSONL="${SPEAKER_PLAN_JSONL:-$DATASET_ROOT/sft/moss_codecvc_sft.$DATASET_NAME.speaker_embedding_plan.ecapa.jsonl}"
EMBEDDING_ROOT="${EMBEDDING_ROOT:-$DATASET_ROOT/speaker_embeddings/ecapa}"
EMBEDDING_SUMMARY_JSON="${EMBEDDING_SUMMARY_JSON:-$DATASET_ROOT/speaker_embeddings/ecapa.extract_summary.json}"
ATTACHED_JSONL="${ATTACHED_JSONL:-$DATASET_ROOT/sft/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.jsonl}"
PROSODY_FEATURE_ROOT="${PROSODY_FEATURE_ROOT:-$DATASET_ROOT/prosody_features}"
PROSODY_JSONL="${PROSODY_JSONL:-$DATASET_ROOT/sft/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.with_prosody.jsonl}"
PROSODY_SUMMARY_JSON="${PROSODY_SUMMARY_JSON:-$DATASET_ROOT/sft/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.with_prosody.summary.json}"
TRAIN_VERSION="${TRAIN_VERSION:-ver2}"
TRAIN_COMMAND_SH="${TRAIN_COMMAND_SH:-$DATASET_ROOT/sft/train_${TRAIN_VERSION}_lora.$DATASET_NAME.sh}"

N_VQ="${N_VQ:-32}"
GPU_IDS="${GPU_IDS:-0}"
SHARD_ROOT="${SHARD_ROOT:-$DATASET_ROOT/shards}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-1}"
SPEAKER_SHARD_COUNT="${SPEAKER_SHARD_COUNT:-1}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-1}"
CODEC_DEVICE="${CODEC_DEVICE:-cuda:0}"
CODEC_DTYPE="${CODEC_DTYPE:-float32}"
CODEC_FIELDS="${CODEC_FIELDS:-source,timbre,target}"
EMIT_MODES="${EMIT_MODES:-no_text}"
NO_TEXT_TEXT_MODE="${NO_TEXT_TEXT_MODE:-placeholder}"
NO_TEXT_PLACEHOLDER="${NO_TEXT_PLACEHOLDER:-<NO_TEXT>}"
SPEAKER_MODEL_NAME="${SPEAKER_MODEL_NAME:-speechbrain_ecapa}"
SPEAKER_MODEL_SOURCE="${SPEAKER_MODEL_SOURCE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"
SPEAKER_DEVICE="${SPEAKER_DEVICE:-cuda:0}"
SPEAKER_LOG_EVERY="${SPEAKER_LOG_EVERY:-1000}"
RUN_PROSODY_FEATURES="${RUN_PROSODY_FEATURES:-0}"
INCLUDE_TARGET_PROSODY="${INCLUDE_TARGET_PROSODY:-1}"
PROSODY_OVERWRITE="${PROSODY_OVERWRITE:-0}"
PROSODY_SAMPLE_RATE="${PROSODY_SAMPLE_RATE:-24000}"
PROSODY_FRAME_MS="${PROSODY_FRAME_MS:-20.0}"
PROSODY_HOP_MS="${PROSODY_HOP_MS:-20.0}"
PROSODY_PAUSE_DB_BELOW_PEAK="${PROSODY_PAUSE_DB_BELOW_PEAK:-35.0}"
TRAIN_LAMBDA_ROUTE="${TRAIN_LAMBDA_ROUTE:-0.01}"
TRAIN_LAMBDA_PROSODY="${TRAIN_LAMBDA_PROSODY:-}"
TRAIN_LAMBDA_CONTENT="${TRAIN_LAMBDA_CONTENT:-0.0}"
TRAIN_PROSODY_F0_WEIGHT="${TRAIN_PROSODY_F0_WEIGHT:-0.0}"
TRAIN_PROSODY_VOICED_WEIGHT="${TRAIN_PROSODY_VOICED_WEIGHT:-0.0}"
TRAIN_PROSODY_ENERGY_WEIGHT="${TRAIN_PROSODY_ENERGY_WEIGHT:-0.5}"
TRAIN_PROSODY_PAUSE_WEIGHT="${TRAIN_PROSODY_PAUSE_WEIGHT:-1.0}"
TRAIN_PROSODY_DURATION_WEIGHT="${TRAIN_PROSODY_DURATION_WEIGHT:-0.5}"
TRAIN_CONTENT_EMBEDDING_DIM="${TRAIN_CONTENT_EMBEDDING_DIM:-0}"
TRAIN_TARGET_SPK_WEIGHT="${TRAIN_TARGET_SPK_WEIGHT:-0.05}"
TRAIN_SOURCE_SUPPRESS_WEIGHT="${TRAIN_SOURCE_SUPPRESS_WEIGHT:-0.05}"
TRAIN_SPEAKER_MARGIN="${TRAIN_SPEAKER_MARGIN:-0.1}"
MAX_ROWS="${MAX_ROWS:-0}"
ENCODE_NO_REUSE="${ENCODE_NO_REUSE:-0}"
FORCE="${FORCE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_CODEC="${RUN_CODEC:-1}"
RUN_SFT="${RUN_SFT:-1}"
RUN_SPEAKER_PLAN="${RUN_SPEAKER_PLAN:-1}"
RUN_SPEAKER_EXTRACT="${RUN_SPEAKER_EXTRACT:-1}"
RUN_ATTACH="${RUN_ATTACH:-1}"
RUN_PROSODY_FEATURE_STAGE="${RUN_PROSODY_FEATURE_STAGE:-$RUN_PROSODY_FEATURES}"
WRITE_TRAIN_COMMAND="${WRITE_TRAIN_COMMAND:-1}"
WAIT_HEARTBEAT_SECS="${WAIT_HEARTBEAT_SECS:-60}"
SFT_PROGRESS_EVERY="${SFT_PROGRESS_EVERY:-1000}"
SPEAKER_PLAN_PROGRESS_EVERY="${SPEAKER_PLAN_PROGRESS_EVERY:-10000}"
ATTACH_PROGRESS_EVERY="${ATTACH_PROGRESS_EVERY:-10000}"
PROSODY_PROGRESS_EVERY="${PROSODY_PROGRESS_EVERY:-1000}"
ATTACH_REQUIRE_EMBEDDING_EXISTS="${ATTACH_REQUIRE_EMBEDDING_EXISTS:-1}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-0}"
GPU_KEEPALIVE_STAGES="${GPU_KEEPALIVE_STAGES:-speaker_extract,attach,prosody_extract}"
GPU_KEEPALIVE_GPU_IDS="${GPU_KEEPALIVE_GPU_IDS:-$GPU_IDS}"
GPU_KEEPALIVE_LOG_DIR="${GPU_KEEPALIVE_LOG_DIR:-$SHARD_ROOT/gpu_keepalive}"
GPU_KEEPALIVE_MATMUL_SIZE="${GPU_KEEPALIVE_MATMUL_SIZE:-2048}"
GPU_KEEPALIVE_SLEEP_SEC="${GPU_KEEPALIVE_SLEEP_SEC:-0.05}"
GPU_KEEPALIVE_DTYPE="${GPU_KEEPALIVE_DTYPE:-float16}"
GPU_KEEPALIVE_PID=0
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

run_bg_cmd() {
  local log_file="$1"
  shift
  printf '+'
  printf ' %q' "$@"
  printf ' > %q 2>&1 &\n' "$log_file"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@" >"$log_file" 2>&1 &
    RUN_BG_PID=$!
  else
    RUN_BG_PID=0
  fi
}

stage_in_list() {
  local needle="$1"
  local list=",$2,"
  case "$list" in
    *,"$needle",*) return 0 ;;
    *) return 1 ;;
  esac
}

device_for_shard() {
  local base_device="${1:-cuda}"
  local gpu_id="$2"
  case "$base_device" in
    cpu|CPU)
      echo "cpu"
      ;;
    cuda|gpu|auto|"")
      echo "cuda:$gpu_id"
      ;;
    cuda:*)
      # In sharded mode, spread shards across GPU_IDS even if the scalar device
      # default is cuda:0. Set the stage device to cpu to force CPU execution.
      echo "cuda:$gpu_id"
      ;;
    *)
      echo "$base_device"
      ;;
  esac
}

start_gpu_keepalive() {
  local stage="$1"
  if ! truthy "$GPU_KEEPALIVE"; then
    return 0
  fi
  if ! stage_in_list "$stage" "$GPU_KEEPALIVE_STAGES"; then
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[gpu-keepalive] dry-run stage=$stage"
    return 0
  fi
  if [ "$GPU_KEEPALIVE_PID" -ne 0 ] && kill -0 "$GPU_KEEPALIVE_PID" 2>/dev/null; then
    return 0
  fi
  mkdir -p "$GPU_KEEPALIVE_LOG_DIR"
  local log_file="$GPU_KEEPALIVE_LOG_DIR/$stage.log"
  echo "[gpu-keepalive] start stage=$stage gpu_ids=$GPU_KEEPALIVE_GPU_IDS log=$log_file"
  "$PY" - "$GPU_KEEPALIVE_GPU_IDS" "$GPU_KEEPALIVE_MATMUL_SIZE" "$GPU_KEEPALIVE_SLEEP_SEC" "$GPU_KEEPALIVE_DTYPE" >"$log_file" 2>&1 <<'PY' &
from __future__ import annotations

import sys
import time

gpu_ids = [int(x) for x in sys.argv[1].split(",") if x.strip()]
size = int(sys.argv[2])
sleep_sec = float(sys.argv[3])
dtype_name = sys.argv[4]

try:
    import torch
except Exception as exc:
    print(f"[gpu-keepalive] torch import failed: {type(exc).__name__}: {exc}", flush=True)
    while True:
        time.sleep(60)

if not torch.cuda.is_available() or not gpu_ids:
    print("[gpu-keepalive] cuda unavailable or no gpu ids; sleeping only", flush=True)
    while True:
        time.sleep(60)

dtype = torch.float16 if dtype_name in {"float16", "fp16", "half"} else torch.float32
states = []
for gpu_id in gpu_ids:
    device = torch.device(f"cuda:{gpu_id}")
    with torch.cuda.device(device):
        a = torch.randn((size, size), device=device, dtype=dtype)
        b = torch.randn((size, size), device=device, dtype=dtype)
        states.append((device, a, b))
print(f"[gpu-keepalive] running devices={gpu_ids} size={size} dtype={dtype}", flush=True)

step = 0
while True:
    for device, a, b in states:
        with torch.cuda.device(device):
            c = a @ b
            a.add_(c.mean() * 0.0)
    torch.cuda.synchronize()
    step += 1
    if step % 100 == 0:
        print(f"[gpu-keepalive] step={step}", flush=True)
    if sleep_sec > 0:
        time.sleep(sleep_sec)
PY
  GPU_KEEPALIVE_PID=$!
}

stop_gpu_keepalive() {
  if [ "${GPU_KEEPALIVE_PID:-0}" -ne 0 ]; then
    if kill -0 "$GPU_KEEPALIVE_PID" 2>/dev/null; then
      echo "[gpu-keepalive] stop pid=$GPU_KEEPALIVE_PID"
      kill "$GPU_KEEPALIVE_PID" 2>/dev/null || true
      wait "$GPU_KEEPALIVE_PID" 2>/dev/null || true
    fi
    GPU_KEEPALIVE_PID=0
  fi
}

trap 'stop_gpu_keepalive' EXIT
trap 'stop_gpu_keepalive; exit 130' INT TERM

stage_should_skip() {
  local output="$1"
  if truthy "$SKIP_EXISTING" && [ "$FORCE" != "1" ] && [ -s "$output" ]; then
    return 0
  fi
  return 1
}

stage_should_skip_done() {
  local output="$1"
  local marker="$output.done.json"
  if ! truthy "$SKIP_EXISTING" || [ "$FORCE" = "1" ] || [ ! -s "$output" ]; then
    return 1
  fi
  if [ -s "$marker" ]; then
    return 0
  fi
  echo "[resume] existing output has no done marker, will rebuild safely: $output"
  return 1
}

expected_sft_rows() {
  local input_jsonl="$1"
  "$PY" - "$input_jsonl" "$EMIT_MODES" "$MAX_ROWS" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

input_jsonl = Path(sys.argv[1])
emit_modes = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
max_rows = int(sys.argv[3])
count = 0
with input_jsonl.open("r", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if not (
            row.get("source_audio_codes_path")
            and row.get("timbre_ref_audio_codes_path")
            and row.get("target_audio_codes_path")
        ):
            continue
        preferred = row.get("preferred_emit_mode")
        for mode in emit_modes:
            if preferred and preferred != mode:
                continue
            count += 1
        if max_rows > 0 and count >= max_rows:
            break
print(count)
PY
}

jsonl_line_count() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo 0
    return 0
  fi
  wc -l < "$path" | tr -d ' '
}

stage_should_skip_sft() {
  local output="$1"
  local input="$2"
  local marker="$output.done.json"
  if ! truthy "$SKIP_EXISTING" || [ "$FORCE" = "1" ] || [ ! -s "$output" ]; then
    return 1
  fi
  if [ -s "$marker" ]; then
    return 0
  fi
  echo "[check] validating existing SFT JSONL without done marker: $output"
  local expected actual
  expected=$(expected_sft_rows "$input")
  actual=$(jsonl_line_count "$output")
  if [ "$expected" = "$actual" ] && [ "$actual" -gt 0 ]; then
    "$PY" - "$marker" "$input" "$output" "$actual" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

marker = Path(sys.argv[1])
payload = {
    "status": "complete",
    "input_jsonl": str(Path(sys.argv[2]).resolve()),
    "output_jsonl": str(Path(sys.argv[3]).resolve()),
    "written": int(sys.argv[4]),
    "source": "validated_existing_output",
}
marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
    return 0
  fi
  echo "[resume] existing SFT JSONL is incomplete/unverified: actual=$actual expected=$expected; rebuilding via streaming writer"
  return 1
}

split_jsonl_round_robin() {
  local input_jsonl="$1"
  local out_dir="$2"
  local shard_count="$3"
  local marker="$out_dir/.split_${shard_count}.done"
  if truthy "$SKIP_EXISTING" && [ "$FORCE" != "1" ] && [ -s "$marker" ]; then
    echo "[skip] reuse split shards: $out_dir"
    return 0
  fi
  echo "[split] $input_jsonl -> $out_dir shards=$shard_count"
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  "$PY" - "$input_jsonl" "$out_dir" "$shard_count" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

input_jsonl = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
shard_count = int(sys.argv[3])
handles = []
try:
    for idx in range(shard_count):
        handles.append((out_dir / f"shard_{idx:03d}.jsonl").open("w", encoding="utf-8"))
    with input_jsonl.open("r", encoding="utf-8") as src:
        for line_idx, line in enumerate(src):
            handles[line_idx % shard_count].write(line)
finally:
    for handle in handles:
        handle.close()
PY
  printf '%s\n' "$input_jsonl" > "$marker"
}

concat_shards() {
  local out_jsonl="$1"
  shift
  echo "[concat] -> $out_jsonl"
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  : > "$out_jsonl"
  local shard
  for shard in "$@"; do
    if [ ! -s "$shard" ]; then
      echo "Missing shard output: $shard" >&2
      exit 2
    fi
    cat "$shard" >> "$out_jsonl"
  done
}

wait_for_pids() {
  local failed=0
  local heartbeat_pid=0
  local heartbeat_secs="${WAIT_HEARTBEAT_SECS:-60}"
  local heartbeat_label="${WAIT_HEARTBEAT_LABEL:-shards}"
  local heartbeat_log_glob="${WAIT_HEARTBEAT_LOG_GLOB:-}"
  local pid
  if [ "$DRY_RUN" -eq 0 ] && [ "$heartbeat_secs" -gt 0 ]; then
    (
      while true; do
        sleep "$heartbeat_secs" || exit 0
        printf '[heartbeat] %s waiting for %s pids=' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$heartbeat_label"
        printf ' %s' "$@"
        printf '\n'
        if [ -n "$heartbeat_log_glob" ]; then
          local log_file
          for log_file in $heartbeat_log_glob; do
            if [ -f "$log_file" ]; then
              printf '[heartbeat] %s %s: %s\n' \
                "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
                "$(basename "$log_file")" \
                "$(tail -n 1 "$log_file" 2>/dev/null || true)"
            fi
          done
        fi
      done
    ) &
    heartbeat_pid=$!
  fi
  for pid in "$@"; do
    if [ "$pid" -eq 0 ]; then
      continue
    fi
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$heartbeat_pid" -ne 0 ]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  if [ "$failed" -ne 0 ]; then
    echo "At least one shard process failed." >&2
    exit 1
  fi
}

write_sharded_summary() {
  local summary_json="$1"
  local stage="$2"
  local shard_dir="$3"
  local shard_count="$4"
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  "$PY" - "$summary_json" "$stage" "$shard_dir" "$shard_count" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
stage = sys.argv[2]
shard_dir = Path(sys.argv[3])
shard_count = int(sys.argv[4])
payload = {
    "stage": stage,
    "sharded": True,
    "shard_count": shard_count,
    "shard_dir": str(shard_dir),
    "shard_summaries": [],
}
for path in sorted(shard_dir.glob("*.summary.json")):
    try:
        payload["shard_summaries"].append(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        payload["shard_summaries"].append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
summary.parent.mkdir(parents=True, exist_ok=True)
summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

write_done_from_summary() {
  local output_jsonl="$1"
  local summary_json="$2"
  local done_json="$output_jsonl.done.json"
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  "$PY" - "$output_jsonl" "$summary_json" "$done_json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

output_jsonl = Path(sys.argv[1]).resolve()
summary_json = Path(sys.argv[2])
done_json = Path(sys.argv[3])
payload = json.loads(summary_json.read_text(encoding="utf-8"))
rows = 0
for shard in payload.get("shard_summaries", []):
    if isinstance(shard, dict):
        rows += int(shard.get("written") or shard.get("rows") or 0)
payload.update(
    {
        "status": "complete",
        "output_jsonl": str(output_jsonl),
        "written": rows,
    }
)
done_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

max_rows_args=()
if [ "$MAX_ROWS" -gt 0 ]; then
  max_rows_args=(--max-rows "$MAX_ROWS")
fi

encode_reuse_args=()
if truthy "$ENCODE_NO_REUSE"; then
  encode_reuse_args=(--no-reuse)
fi

include_target_prosody_args=(--no-include-target)
if truthy "$INCLUDE_TARGET_PROSODY"; then
  include_target_prosody_args=(--include-target)
fi

prosody_overwrite_args=(--no-overwrite)
if truthy "$PROSODY_OVERWRITE"; then
  prosody_overwrite_args=(--overwrite)
fi

attach_require_args=(--no-require-embedding-exists)
if truthy "$ATTACH_REQUIRE_EMBEDDING_EXISTS"; then
  attach_require_args=(--require-embedding-exists)
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
if [ "${#GPU_ID_ARRAY[@]}" -eq 0 ]; then
  GPU_ID_ARRAY=(0)
fi

ACTIVE_TRAIN_JSONL="$ATTACHED_JSONL"
ACTIVE_TRAIN_KIND="speaker_attached"
if truthy "$RUN_PROSODY_FEATURE_STAGE"; then
  ACTIVE_TRAIN_JSONL="$PROSODY_JSONL"
  ACTIVE_TRAIN_KIND="speaker_attached_with_prosody"
fi
TOTAL_STAGES=5
if truthy "$RUN_PROSODY_FEATURE_STAGE"; then
  TOTAL_STAGES=6
fi
if [ -z "${TRAIN_LAMBDA_PROSODY:-}" ]; then
  if truthy "$RUN_PROSODY_FEATURE_STAGE"; then
    TRAIN_LAMBDA_PROSODY="0.05"
  else
    TRAIN_LAMBDA_PROSODY="0.0"
  fi
fi

mkdir -p \
  "$(dirname "$ENCODED_JSONL")" \
  "$CODES_DIR" \
  "$(dirname "$SFT_JSONL")" \
  "$(dirname "$SPEAKER_PLAN_JSONL")" \
  "$EMBEDDING_ROOT" \
  "$(dirname "$ATTACHED_JSONL")" \
  "$PROSODY_FEATURE_ROOT" \
  "$(dirname "$PROSODY_JSONL")"

echo "=========================================="
echo "Build train-ready MOSS-CodecVC no-text data"
echo "  ROOT=$ROOT"
echo "  PY=$PY"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  INPUT_JSONL=$INPUT_JSONL"
echo "  CONFIG=$CONFIG"
echo "  ENCODED_JSONL=$ENCODED_JSONL"
echo "  CODES_DIR=$CODES_DIR"
echo "  SFT_JSONL=$SFT_JSONL"
echo "  SPEAKER_PLAN_JSONL=$SPEAKER_PLAN_JSONL"
echo "  EMBEDDING_ROOT=$EMBEDDING_ROOT"
echo "  EMBEDDING_SUMMARY_JSON=$EMBEDDING_SUMMARY_JSON"
echo "  ATTACHED_JSONL=$ATTACHED_JSONL"
echo "  RUN_PROSODY_FEATURES=$RUN_PROSODY_FEATURE_STAGE"
echo "  PROSODY_FEATURE_ROOT=$PROSODY_FEATURE_ROOT"
echo "  PROSODY_JSONL=$PROSODY_JSONL"
echo "  ACTIVE_TRAIN_JSONL=$ACTIVE_TRAIN_JSONL"
echo "  TRAIN_VERSION=$TRAIN_VERSION"
echo "  N_VQ=$N_VQ"
echo "  GPU_IDS=$GPU_IDS"
echo "  CODEC_SHARD_COUNT=$CODEC_SHARD_COUNT"
echo "  SPEAKER_SHARD_COUNT=$SPEAKER_SHARD_COUNT"
  echo "  PROSODY_SHARD_COUNT=$PROSODY_SHARD_COUNT"
  echo "  WAIT_HEARTBEAT_SECS=$WAIT_HEARTBEAT_SECS"
  echo "  SFT_PROGRESS_EVERY=$SFT_PROGRESS_EVERY"
  echo "  SPEAKER_PLAN_PROGRESS_EVERY=$SPEAKER_PLAN_PROGRESS_EVERY"
  echo "  ATTACH_PROGRESS_EVERY=$ATTACH_PROGRESS_EVERY"
  echo "  PROSODY_PROGRESS_EVERY=$PROSODY_PROGRESS_EVERY"
  echo "  ATTACH_REQUIRE_EMBEDDING_EXISTS=$ATTACH_REQUIRE_EMBEDDING_EXISTS"
  echo "  CODEC_DEVICE=$CODEC_DEVICE"
echo "  SPEAKER_DEVICE=$SPEAKER_DEVICE"
echo "  GPU_KEEPALIVE=$GPU_KEEPALIVE"
echo "  GPU_KEEPALIVE_STAGES=$GPU_KEEPALIVE_STAGES"
echo "  GPU_KEEPALIVE_GPU_IDS=$GPU_KEEPALIVE_GPU_IDS"
echo "  MAX_ROWS=$MAX_ROWS"
echo "  FORCE=$FORCE"
echo "  SKIP_EXISTING=$SKIP_EXISTING"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="

if [ ! -s "$INPUT_JSONL" ]; then
  echo "Input manifest is missing or empty: $INPUT_JSONL" >&2
  exit 2
fi

cd "$ROOT"
run_cmd "$PY" -m py_compile \
  scripts/001002_encode_codec_tokens.py \
  scripts/001003_build_moss_sft_jsonl.py \
  scripts/001007_build_speaker_embedding_plan.py \
  scripts/001010_attach_speaker_embeddings.py \
  scripts/001011_extract_speaker_embeddings.py \
  scripts/001015_extract_prosody_content_features.py

if truthy "$RUN_CODEC"; then
  if stage_should_skip "$ENCODED_JSONL"; then
    echo "[skip] codec encoded manifest exists: $ENCODED_JSONL"
  else
    echo "=========================================="
    echo "Stage 1/$TOTAL_STAGES: codec encode wav triples"
    echo "=========================================="
    if [ "$CODEC_SHARD_COUNT" -gt 1 ]; then
      codec_shard_dir="$SHARD_ROOT/codec_encode"
      split_jsonl_round_robin "$INPUT_JSONL" "$codec_shard_dir/input" "$CODEC_SHARD_COUNT"
      mkdir -p "$codec_shard_dir/logs" "$codec_shard_dir/output"
      pids=()
      outputs=()
      for shard_idx in $(seq 0 $((CODEC_SHARD_COUNT - 1))); do
        shard_name=$(printf 'shard_%03d' "$shard_idx")
        gpu_id="${GPU_ID_ARRAY[$((shard_idx % ${#GPU_ID_ARRAY[@]}))]}"
        shard_input="$codec_shard_dir/input/$shard_name.jsonl"
        shard_output="$codec_shard_dir/output/$shard_name.encoded.jsonl"
        shard_codes_dir="$CODES_DIR/shards/$shard_name"
        shard_log="$codec_shard_dir/logs/$shard_name.log"
        outputs+=("$shard_output")
        run_bg_cmd "$shard_log" "$PY" scripts/001002_encode_codec_tokens.py \
          --config "$CONFIG" \
          --input-jsonl "$shard_input" \
          --output-jsonl "$shard_output" \
          --codes-dir "$shard_codes_dir" \
          --n-vq "$N_VQ" \
          --device "cuda:$gpu_id" \
          --dtype "$CODEC_DTYPE" \
          --fields "$CODEC_FIELDS" \
          "${encode_reuse_args[@]}"
        pids+=("$RUN_BG_PID")
      done
      WAIT_HEARTBEAT_LABEL="codec_encode" \
      WAIT_HEARTBEAT_LOG_GLOB="$codec_shard_dir/logs/shard_*.log" \
      wait_for_pids "${pids[@]}"
      concat_shards "$ENCODED_JSONL" "${outputs[@]}"
    else
      run_cmd "$PY" scripts/001002_encode_codec_tokens.py \
        --config "$CONFIG" \
        --input-jsonl "$INPUT_JSONL" \
        --output-jsonl "$ENCODED_JSONL" \
        --codes-dir "$CODES_DIR" \
        --n-vq "$N_VQ" \
        --device "$CODEC_DEVICE" \
        --dtype "$CODEC_DTYPE" \
        --fields "$CODEC_FIELDS" \
        "${encode_reuse_args[@]}"
    fi
  fi
fi

if truthy "$RUN_SFT"; then
  if stage_should_skip_sft "$SFT_JSONL" "$ENCODED_JSONL"; then
    echo "[skip] SFT JSONL exists: $SFT_JSONL"
  else
    echo "=========================================="
    echo "Stage 2/$TOTAL_STAGES: build no-text SFT JSONL"
    echo "=========================================="
    run_cmd "$PY" scripts/001003_build_moss_sft_jsonl.py \
      --input-jsonl "$ENCODED_JSONL" \
      --output-jsonl "$SFT_JSONL" \
      --emit-modes "$EMIT_MODES" \
      --no-text-text-mode "$NO_TEXT_TEXT_MODE" \
      --no-text-placeholder "$NO_TEXT_PLACEHOLDER" \
      --progress-every "$SFT_PROGRESS_EVERY" \
      "${max_rows_args[@]}"
  fi
fi

if truthy "$RUN_SPEAKER_PLAN"; then
  if stage_should_skip_done "$SPEAKER_PLAN_JSONL"; then
    echo "[skip] speaker embedding plan exists: $SPEAKER_PLAN_JSONL"
  else
    echo "=========================================="
    echo "Stage 3/$TOTAL_STAGES: build speaker embedding plan"
    echo "=========================================="
    run_cmd "$PY" scripts/001007_build_speaker_embedding_plan.py \
      --input-jsonl "$SFT_JSONL" \
      --output-jsonl "$SPEAKER_PLAN_JSONL" \
      --embedding-root "$EMBEDDING_ROOT" \
      --model-name "$SPEAKER_MODEL_NAME" \
      --progress-every "$SPEAKER_PLAN_PROGRESS_EVERY" \
      "${max_rows_args[@]}"
  fi
fi

if truthy "$RUN_SPEAKER_EXTRACT"; then
  if stage_should_skip "$EMBEDDING_SUMMARY_JSON"; then
    echo "[skip] speaker embedding extraction summary exists: $EMBEDDING_SUMMARY_JSON"
  else
    echo "=========================================="
    echo "Stage 4/$TOTAL_STAGES: extract light ECAPA speaker embeddings"
    echo "=========================================="
    start_gpu_keepalive "speaker_extract"
    if [ "$SPEAKER_SHARD_COUNT" -gt 1 ]; then
      speaker_shard_dir="$SHARD_ROOT/speaker_extract"
      split_jsonl_round_robin "$SPEAKER_PLAN_JSONL" "$speaker_shard_dir/input" "$SPEAKER_SHARD_COUNT"
      mkdir -p "$speaker_shard_dir/logs" "$speaker_shard_dir/summaries"
      pids=()
      for shard_idx in $(seq 0 $((SPEAKER_SHARD_COUNT - 1))); do
        shard_name=$(printf 'shard_%03d' "$shard_idx")
        gpu_id="${GPU_ID_ARRAY[$((shard_idx % ${#GPU_ID_ARRAY[@]}))]}"
        shard_device=$(device_for_shard "$SPEAKER_DEVICE" "$gpu_id")
        shard_input="$speaker_shard_dir/input/$shard_name.jsonl"
        shard_summary="$speaker_shard_dir/summaries/$shard_name.summary.json"
        shard_log="$speaker_shard_dir/logs/$shard_name.log"
        run_bg_cmd "$shard_log" "$PY" scripts/001011_extract_speaker_embeddings.py \
          --embedding-plan-jsonl "$shard_input" \
          --backend speechbrain_ecapa \
          --model-source "$SPEAKER_MODEL_SOURCE" \
          --device "$shard_device" \
          --log-every "$SPEAKER_LOG_EVERY" \
          --summary-json "$shard_summary" \
          --no-allow-missing-audio \
          "${max_rows_args[@]}"
        pids+=("$RUN_BG_PID")
      done
      WAIT_HEARTBEAT_LABEL="speaker_extract" \
      WAIT_HEARTBEAT_LOG_GLOB="$speaker_shard_dir/logs/shard_*.log" \
      wait_for_pids "${pids[@]}"
      write_sharded_summary "$EMBEDDING_SUMMARY_JSON" "speaker_extract" "$speaker_shard_dir/summaries" "$SPEAKER_SHARD_COUNT"
    else
      run_cmd "$PY" scripts/001011_extract_speaker_embeddings.py \
        --embedding-plan-jsonl "$SPEAKER_PLAN_JSONL" \
        --backend speechbrain_ecapa \
        --model-source "$SPEAKER_MODEL_SOURCE" \
        --device "$SPEAKER_DEVICE" \
        --log-every "$SPEAKER_LOG_EVERY" \
        --summary-json "$EMBEDDING_SUMMARY_JSON" \
        --no-allow-missing-audio \
        "${max_rows_args[@]}"
    fi
    stop_gpu_keepalive
  fi
fi

if truthy "$RUN_ATTACH"; then
  if stage_should_skip_done "$ATTACHED_JSONL"; then
    echo "[skip] speaker-attached JSONL exists: $ATTACHED_JSONL"
  else
    echo "=========================================="
    echo "Stage 5/$TOTAL_STAGES: attach speaker embedding paths"
    echo "=========================================="
    start_gpu_keepalive "attach"
    run_cmd "$PY" scripts/001010_attach_speaker_embeddings.py \
      --input-jsonl "$SFT_JSONL" \
      --embedding-plan-jsonl "$SPEAKER_PLAN_JSONL" \
      --output-jsonl "$ATTACHED_JSONL" \
      "${attach_require_args[@]}" \
      --progress-every "$ATTACH_PROGRESS_EVERY" \
      "${max_rows_args[@]}"
    stop_gpu_keepalive
  fi
fi

if truthy "$RUN_PROSODY_FEATURE_STAGE"; then
  if stage_should_skip_done "$PROSODY_JSONL"; then
    echo "[skip] prosody feature JSONL exists: $PROSODY_JSONL"
  else
    echo "=========================================="
    echo "Stage 6/6: extract Ver2 source/target prosody features"
    echo "=========================================="
    start_gpu_keepalive "prosody_extract"
    if [ "$PROSODY_SHARD_COUNT" -gt 1 ]; then
      prosody_shard_dir="$SHARD_ROOT/prosody_extract"
      split_jsonl_round_robin "$ATTACHED_JSONL" "$prosody_shard_dir/input" "$PROSODY_SHARD_COUNT"
      mkdir -p "$prosody_shard_dir/logs" "$prosody_shard_dir/output"
      pids=()
      outputs=()
      for shard_idx in $(seq 0 $((PROSODY_SHARD_COUNT - 1))); do
        shard_name=$(printf 'shard_%03d' "$shard_idx")
        shard_input="$prosody_shard_dir/input/$shard_name.jsonl"
        shard_output="$prosody_shard_dir/output/$shard_name.with_prosody.jsonl"
        shard_log="$prosody_shard_dir/logs/$shard_name.log"
        outputs+=("$shard_output")
        run_bg_cmd "$shard_log" "$PY" scripts/001015_extract_prosody_content_features.py \
          --input-jsonl "$shard_input" \
          --output-jsonl "$shard_output" \
          --feature-root "$PROSODY_FEATURE_ROOT" \
          --sample-rate "$PROSODY_SAMPLE_RATE" \
          --frame-ms "$PROSODY_FRAME_MS" \
          --hop-ms "$PROSODY_HOP_MS" \
          --pause-db-below-peak "$PROSODY_PAUSE_DB_BELOW_PEAK" \
          --progress-every "$PROSODY_PROGRESS_EVERY" \
          "${include_target_prosody_args[@]}" \
          "${prosody_overwrite_args[@]}" \
          "${max_rows_args[@]}"
        pids+=("$RUN_BG_PID")
      done
      WAIT_HEARTBEAT_LABEL="prosody_extract" \
      WAIT_HEARTBEAT_LOG_GLOB="$prosody_shard_dir/logs/shard_*.log" \
      wait_for_pids "${pids[@]}"
      concat_shards "$PROSODY_JSONL" "${outputs[@]}"
      write_sharded_summary "$PROSODY_SUMMARY_JSON" "prosody_extract" "$prosody_shard_dir/output" "$PROSODY_SHARD_COUNT"
      write_done_from_summary "$PROSODY_JSONL" "$PROSODY_SUMMARY_JSON"
    else
      run_cmd "$PY" scripts/001015_extract_prosody_content_features.py \
        --input-jsonl "$ATTACHED_JSONL" \
        --output-jsonl "$PROSODY_JSONL" \
        --feature-root "$PROSODY_FEATURE_ROOT" \
        --sample-rate "$PROSODY_SAMPLE_RATE" \
        --frame-ms "$PROSODY_FRAME_MS" \
        --hop-ms "$PROSODY_HOP_MS" \
        --pause-db-below-peak "$PROSODY_PAUSE_DB_BELOW_PEAK" \
        --progress-every "$PROSODY_PROGRESS_EVERY" \
        "${include_target_prosody_args[@]}" \
        "${prosody_overwrite_args[@]}" \
        "${max_rows_args[@]}"
    fi
    stop_gpu_keepalive
  fi
fi

if truthy "$WRITE_TRAIN_COMMAND"; then
  if [ "$DRY_RUN" -eq 0 ]; then
    cat > "$TRAIN_COMMAND_SH" <<EOF
#!/usr/bin/env bash
set -euo pipefail

PY="\${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
cd "$ROOT"

"\$PY" scripts/002002_train_moss_codecvc_lora.py \\
  --config configs/remote_full.yaml \\
  --train-jsonl "$ACTIVE_TRAIN_JSONL" \\
  --output-dir outputs/lora_runs/${TRAIN_VERSION}_${DATASET_NAME}_light_ecapa \\
  --version "$TRAIN_VERSION" \\
  --use-timbre-memory \\
  --speaker-encoder-type embedding_loader \\
  --speaker-embedding-dim 192 \\
  --target-speaker-similarity-weight "$TRAIN_TARGET_SPK_WEIGHT" \\
  --source-speaker-suppression-weight "$TRAIN_SOURCE_SUPPRESS_WEIGHT" \\
  --speaker-loss-margin "$TRAIN_SPEAKER_MARGIN" \\
  --lambda-route "$TRAIN_LAMBDA_ROUTE" \\
  --lambda-prosody "$TRAIN_LAMBDA_PROSODY" \\
  --prosody-f0-weight "$TRAIN_PROSODY_F0_WEIGHT" \\
  --prosody-voiced-weight "$TRAIN_PROSODY_VOICED_WEIGHT" \\
  --prosody-energy-weight "$TRAIN_PROSODY_ENERGY_WEIGHT" \\
  --prosody-pause-weight "$TRAIN_PROSODY_PAUSE_WEIGHT" \\
  --prosody-duration-weight "$TRAIN_PROSODY_DURATION_WEIGHT" \\
  --lambda-content "$TRAIN_LAMBDA_CONTENT" \\
  --content-embedding-dim "$TRAIN_CONTENT_EMBEDDING_DIM" \\
  --learning-rate 1e-5 \\
  --per-device-batch-size 1 \\
  --gradient-accumulation-steps 8 \\
  --mixed-precision bf16
EOF
    chmod +x "$TRAIN_COMMAND_SH"
  fi
fi

echo "=========================================="
echo "Train-ready preprocessing finished"
echo "  input_manifest=$INPUT_JSONL"
echo "  encoded_manifest=$ENCODED_JSONL"
echo "  sft_jsonl=$SFT_JSONL"
echo "  speaker_plan=$SPEAKER_PLAN_JSONL"
echo "  embedding_root=$EMBEDDING_ROOT"
echo "  attached_jsonl=$ATTACHED_JSONL"
echo "  prosody_jsonl=$PROSODY_JSONL"
echo "  active_train_kind=$ACTIVE_TRAIN_KIND"
echo "  train_jsonl=$ACTIVE_TRAIN_JSONL"
echo "  train_command=$TRAIN_COMMAND_SH"
echo "=========================================="
