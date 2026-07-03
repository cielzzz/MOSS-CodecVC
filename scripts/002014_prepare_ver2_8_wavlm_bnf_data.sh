#!/usr/bin/env bash
# Ver2.8 data preparation:
# 1) extract WavLM-BNF continuous features for ASR-clean no-text/text data;
# 2) merge sharded manifests;
# 3) build ASR-clean train/valid splits for Ver2.8 training.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"
WAVLM_CACHE_DIR="${WAVLM_CACHE_DIR:-$DOWNLOAD_ROOT/huggingface}"

NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-zh45w_en22w_no_text}"
TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"

NO_TEXT_HUBERT_CLEAN="${NO_TEXT_HUBERT_CLEAN:-$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl}"
TEXT_HUBERT_CLEAN="${TEXT_HUBERT_CLEAN:-$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.ctc_clean.jsonl}"
NO_TEXT_WAVLM_CLEAN="${NO_TEXT_WAVLM_CLEAN:-$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl}"
TEXT_WAVLM_CLEAN="${TEXT_WAVLM_CLEAN:-$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl}"

NO_TEXT_WAVLM_FEATURE_ROOT="${NO_TEXT_WAVLM_FEATURE_ROOT:-$ROOT/trainset/$NO_TEXT_DATASET_NAME/semantic_features/wavlm_bnf}"
TEXT_WAVLM_FEATURE_ROOT="${TEXT_WAVLM_FEATURE_ROOT:-$ROOT/trainset/$TEXT_DATASET_NAME/semantic_features/wavlm_bnf}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_8_prepared}"

DEFAULT_WAVLM_MODEL="$(find "$WAVLM_CACHE_DIR/models--microsoft--wavlm-base-plus/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1 || true)"
if [ -z "$DEFAULT_WAVLM_MODEL" ] || [ ! -f "$DEFAULT_WAVLM_MODEL/config.json" ]; then
  DEFAULT_WAVLM_MODEL="microsoft/wavlm-base-plus"
fi
WAVLM_MODEL="${WAVLM_MODEL:-$DEFAULT_WAVLM_MODEL}"
if [ -f "$WAVLM_MODEL/config.json" ]; then
  WAVLM_LOCAL_FILES_ONLY="${WAVLM_LOCAL_FILES_ONLY:-1}"
else
  WAVLM_LOCAL_FILES_ONLY="${WAVLM_LOCAL_FILES_ONLY:-0}"
fi

WAVLM_LAYER="${WAVLM_LAYER:-9}"
WAVLM_DTYPE="${WAVLM_DTYPE:-auto}"
WAVLM_SAVE_DTYPE="${WAVLM_SAVE_DTYPE:-float16}"
WAVLM_DOWNSAMPLE_STRIDE="${WAVLM_DOWNSAMPLE_STRIDE:-1}"
WAVLM_USE_SAFETENSORS="${WAVLM_USE_SAFETENSORS:-false}"
WAVLM_NUM_SHARDS="${WAVLM_NUM_SHARDS:-8}"
NO_TEXT_WAVLM_NUM_SHARDS="${NO_TEXT_WAVLM_NUM_SHARDS:-$WAVLM_NUM_SHARDS}"
TEXT_WAVLM_NUM_SHARDS="${TEXT_WAVLM_NUM_SHARDS:-$WAVLM_NUM_SHARDS}"
WAVLM_DEVICES="${WAVLM_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
NO_TEXT_WAVLM_DEVICES="${NO_TEXT_WAVLM_DEVICES:-$WAVLM_DEVICES}"
TEXT_WAVLM_DEVICES="${TEXT_WAVLM_DEVICES:-$WAVLM_DEVICES}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"
MAX_ROWS="${MAX_ROWS:-0}"

RUN_NO_TEXT="${RUN_NO_TEXT:-1}"
RUN_TEXT="${RUN_TEXT:-1}"
RUN_PREPARE="${RUN_PREPARE:-1}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REUSE_EXISTING_FEATURES="${REUSE_EXISTING_FEATURES:-1}"
PREPARE_OVERWRITE="${PREPARE_OVERWRITE:-1}"
CHECK_FEATURE_FILES="${CHECK_FEATURE_FILES:-1}"
REQUIRE_NO_TEXT_TARGET_FEATURE="${REQUIRE_NO_TEXT_TARGET_FEATURE:-0}"
# repeat is the total sampled copies. 5 means original text data plus 4 extra
# copies, keeping the current Ver2.8 effective text/no-text train sizes balanced.
TEXT_REPEAT="${TEXT_REPEAT:-5}"

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

pick_csv_item() {
  local csv="$1"
  local index="$2"
  IFS=',' read -r -a items <<< "$csv"
  if [ "${#items[@]}" -eq 0 ]; then
    echo ""
    return 1
  fi
  echo "${items[$((index % ${#items[@]}))]}"
}

wait_for_pids() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    return 1
  fi
  return 0
}

merge_jsonl_shards() {
  local shard_dir="$1"
  local final_jsonl="$2"
  local num_shards="$3"
  local tmp_jsonl="${final_jsonl}.tmp"
  local shard
  local shard_file
  mkdir -p "$(dirname "$final_jsonl")"
  rm -f "$tmp_jsonl"
  for shard in $(seq 0 $((num_shards - 1))); do
    shard_file=$(printf "%s/shard_%05d.jsonl" "$shard_dir" "$shard")
    if [ ! -s "$shard_file" ]; then
      echo "ERROR: missing or empty shard output: $shard_file" >&2
      exit 1
    fi
    cat "$shard_file" >>"$tmp_jsonl"
  done
  mv "$tmp_jsonl" "$final_jsonl"
}

merge_summary_shards() {
  local shard_dir="$1"
  local final_jsonl="$2"
  local num_shards="$3"
  "$PY" - "$shard_dir" "$final_jsonl" "$num_shards" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

shard_dir = Path(sys.argv[1])
final_jsonl = Path(sys.argv[2])
num_shards = int(sys.argv[3])
stats: Counter[str] = Counter()
scalar_stats = {}
summaries = []
for shard in range(num_shards):
    base = shard_dir / f"shard_{shard:05d}.jsonl"
    summary_path = Path(str(base) + ".summary.json")
    if not summary_path.exists():
        summary_path = Path(str(base) + ".done.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"missing shard summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summaries.append(payload)
    for key, value in dict(payload.get("stats") or {}).items():
        if isinstance(value, (int, float)):
            if key == "feature_dim" or key.endswith("_feature_dim"):
                scalar_stats.setdefault(key, value)
            else:
                stats[key] += value
for key, value in scalar_stats.items():
    stats[key] = value
summary = dict(summaries[0]) if summaries else {}
summary.update(
    {
        "status": "complete",
        "rows": int(sum(int(item.get("rows") or 0) for item in summaries)),
        "output_jsonl": str(final_jsonl),
        "num_shards": num_shards,
        "shard_summaries": [str(shard_dir / f"shard_{idx:05d}.jsonl.summary.json") for idx in range(num_shards)],
        "stats": dict(stats),
    }
)
Path(str(final_jsonl) + ".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
Path(str(final_jsonl) + ".done.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

extract_manifest() {
  local label="$1"
  local input_jsonl="$2"
  local output_jsonl="$3"
  local feature_root="$4"
  local source="$5"
  local num_shards="$6"
  local devices="$7"

  if [ ! -s "$input_jsonl" ]; then
    echo "ERROR: missing input manifest for $label: $input_jsonl" >&2
    exit 1
  fi
  if truthy "$SKIP_EXISTING" && [ "$OVERWRITE" != "1" ] && [ -s "$output_jsonl" ]; then
    echo "[ver2.8-wavlm] reuse $label manifest: $output_jsonl"
    return 0
  fi
  if [ -e "$output_jsonl" ] && [ "$OVERWRITE" != "1" ]; then
    echo "ERROR: output exists for $label, set OVERWRITE=1 or SKIP_EXISTING=1: $output_jsonl" >&2
    exit 1
  fi

  local common_args=(
    "$ROOT/scripts/001020_extract_hubert_semantic_features.py"
    --input-jsonl "$input_jsonl"
    --feature-root "$feature_root"
    --extractor wavlm
    --model-name-or-path "$WAVLM_MODEL"
    --cache-dir "$WAVLM_CACHE_DIR"
    --source "$source"
    --layer "$WAVLM_LAYER"
    --dtype "$WAVLM_DTYPE"
    --save-dtype "$WAVLM_SAVE_DTYPE"
    --downsample-stride "$WAVLM_DOWNSAMPLE_STRIDE"
    --use-safetensors "$WAVLM_USE_SAFETENSORS"
    --progress-every "$PROGRESS_EVERY"
  )
  if truthy "$WAVLM_LOCAL_FILES_ONLY"; then
    common_args+=(--local-files-only)
  fi
  if truthy "$REUSE_EXISTING_FEATURES"; then
    common_args+=(--reuse-existing-features)
  fi
  if [ "$OVERWRITE" = "1" ]; then
    common_args+=(--overwrite)
  fi
  if [ "$MAX_ROWS" -gt 0 ]; then
    common_args+=(--max-rows "$MAX_ROWS")
  fi

  if [ "$num_shards" -le 1 ]; then
    run_cmd "$PY" "${common_args[@]}" --output-jsonl "$output_jsonl" --device "$(pick_csv_item "$devices" 0)"
    return 0
  fi

  local shard_dir="${output_jsonl}.shards"
  if [ -d "$shard_dir" ] && [ "$OVERWRITE" != "1" ]; then
    echo "ERROR: shard dir exists for $label, set OVERWRITE=1 to rebuild manifests while reusing feature files: $shard_dir" >&2
    exit 1
  fi
  if [ "$OVERWRITE" = "1" ]; then
    rm -rf "$shard_dir"
    rm -f "$output_jsonl" "${output_jsonl}.tmp" "${output_jsonl}.summary.json" "${output_jsonl}.done.json"
  fi
  mkdir -p "$shard_dir"

  local pids=()
  local shard
  for shard in $(seq 0 $((num_shards - 1))); do
    local shard_out
    local shard_log
    local shard_device
    shard_out=$(printf "%s/shard_%05d.jsonl" "$shard_dir" "$shard")
    shard_log=$(printf "%s/shard_%05d.log" "$shard_dir" "$shard")
    shard_device="$(pick_csv_item "$devices" "$shard")"
    echo "[ver2.8-wavlm] launch $label shard=$shard/$num_shards device=$shard_device out=$shard_out"
    if [ "$DRY_RUN" -eq 1 ]; then
      run_cmd "$PY" "${common_args[@]}" \
        --output-jsonl "$shard_out" \
        --device "$shard_device" \
        --num-shards "$num_shards" \
        --shard-index "$shard"
    else
      (
        "$PY" "${common_args[@]}" \
          --output-jsonl "$shard_out" \
          --device "$shard_device" \
          --num-shards "$num_shards" \
          --shard-index "$shard"
      ) >"$shard_log" 2>&1 &
      pids+=("$!")
    fi
  done

  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  if ! wait_for_pids "${pids[@]}"; then
    echo "ERROR: one or more $label WavLM shards failed. Recent logs:" >&2
    tail -n 80 "$shard_dir"/shard_*.log >&2 || true
    exit 1
  fi
  merge_jsonl_shards "$shard_dir" "$output_jsonl" "$num_shards"
  merge_summary_shards "$shard_dir" "$output_jsonl" "$num_shards"
  echo "[ver2.8-wavlm] merged $label manifest: $output_jsonl"
}

prepare_splits() {
  local args=(
    "$ROOT/scripts/002013_prepare_ver2_8_train_valid.py"
    --no-text-jsonl "$NO_TEXT_WAVLM_CLEAN"
    --text-jsonl "$TEXT_WAVLM_CLEAN"
    --output-dir "$PREPARED_DIR"
    --semantic-kind wavlm
    --text-repeat "$TEXT_REPEAT"
  )
  if truthy "$CHECK_FEATURE_FILES"; then
    args+=(--check-feature-files)
  fi
  if truthy "$REQUIRE_NO_TEXT_TARGET_FEATURE"; then
    args+=(--require-no-text-target-feature)
  fi
  if truthy "$PREPARE_OVERWRITE"; then
    args+=(--overwrite)
  fi
  run_cmd "$PY" "${args[@]}"
}

echo "=========================================="
echo "Ver2.8 WavLM-BNF data preparation"
echo "  ROOT=$ROOT"
echo "  PY=$PY"
echo "  WAVLM_MODEL=$WAVLM_MODEL"
echo "  WAVLM_LOCAL_FILES_ONLY=$WAVLM_LOCAL_FILES_ONLY"
echo "  NO_TEXT_HUBERT_CLEAN=$NO_TEXT_HUBERT_CLEAN"
echo "  TEXT_HUBERT_CLEAN=$TEXT_HUBERT_CLEAN"
echo "  NO_TEXT_WAVLM_CLEAN=$NO_TEXT_WAVLM_CLEAN"
echo "  TEXT_WAVLM_CLEAN=$TEXT_WAVLM_CLEAN"
echo "  NO_TEXT_WAVLM_NUM_SHARDS=$NO_TEXT_WAVLM_NUM_SHARDS devices=$NO_TEXT_WAVLM_DEVICES"
echo "  TEXT_WAVLM_NUM_SHARDS=$TEXT_WAVLM_NUM_SHARDS devices=$TEXT_WAVLM_DEVICES"
echo "  PREPARED_DIR=$PREPARED_DIR"
echo "  RUN_NO_TEXT=$RUN_NO_TEXT RUN_TEXT=$RUN_TEXT RUN_PREPARE=$RUN_PREPARE"
echo "  OVERWRITE=$OVERWRITE SKIP_EXISTING=$SKIP_EXISTING REUSE_EXISTING_FEATURES=$REUSE_EXISTING_FEATURES"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="

run_cmd "$PY" -m py_compile \
  "$ROOT/scripts/001020_extract_hubert_semantic_features.py" \
  "$ROOT/scripts/002013_prepare_ver2_8_train_valid.py"

if truthy "$RUN_NO_TEXT"; then
  extract_manifest \
    "no_text" \
    "$NO_TEXT_HUBERT_CLEAN" \
    "$NO_TEXT_WAVLM_CLEAN" \
    "$NO_TEXT_WAVLM_FEATURE_ROOT" \
    "both" \
    "$NO_TEXT_WAVLM_NUM_SHARDS" \
    "$NO_TEXT_WAVLM_DEVICES"
fi

if truthy "$RUN_TEXT"; then
  extract_manifest \
    "text" \
    "$TEXT_HUBERT_CLEAN" \
    "$TEXT_WAVLM_CLEAN" \
    "$TEXT_WAVLM_FEATURE_ROOT" \
    "target" \
    "$TEXT_WAVLM_NUM_SHARDS" \
    "$TEXT_WAVLM_DEVICES"
fi

if truthy "$RUN_PREPARE"; then
  prepare_splits
fi

echo "=========================================="
echo "Ver2.8 WavLM-BNF data preparation finished"
echo "  no_text_wavlm=$NO_TEXT_WAVLM_CLEAN"
echo "  text_wavlm=$TEXT_WAVLM_CLEAN"
echo "  prepared=$PREPARED_DIR"
echo "=========================================="
