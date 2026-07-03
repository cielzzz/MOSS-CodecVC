#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
if [ ! -x "$PY" ]; then
  PY=python
fi

INPUT_ROOT="${INPUT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_data_temp/mtd_pass_nonmulti_primary_le_0p3_split_10k}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$ROOT/trainset/vc_manifest.large_mixed_mode.seedvc_final.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT/trainset/vc_manifest.large_mixed_mode.seedvc_final.summary.json}"
PLAN_JSONL="${PLAN_JSONL:-$ROOT/trainset/intermediate/vc_manifest.large_mixed_mode.seedvc_plan.jsonl}"
PLAN_SUMMARY_JSON="${PLAN_SUMMARY_JSON:-$ROOT/trainset/intermediate/vc_manifest.large_mixed_mode.seedvc_plan.summary.json}"
SEEDVC_JOBS_JSONL="${SEEDVC_JOBS_JSONL:-$ROOT/trainset/seedvc_jobs/vc_manifest.large_mixed_mode.seedvc_jobs.jsonl}"
SEEDVC_RESULTS_JSONL="${SEEDVC_RESULTS_JSONL:-$ROOT/trainset/seedvc_results/vc_manifest.large_mixed_mode.seedvc_results.jsonl}"
SEEDVC_OUTPUT_ROOT="${SEEDVC_OUTPUT_ROOT:-$ROOT/trainset/seedvc_targets/large_mixed_mode_full}"
LOG_DIR="${LOG_DIR:-$ROOT/trainset/logs}"
RUN_NAME="${RUN_NAME:-large_mixed_mode_seedvc_final}"
EMIT_PAIR_TYPES="${EMIT_PAIR_TYPES:-no_text,text_prosody}"
LANGUAGES="${LANGUAGES:-zh,en}"
MAX_ROWS="${MAX_ROWS:-0}"
MAX_PAIRS="${MAX_PAIRS:-0}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-1.0}"
MAX_DURATION_SEC="${MAX_DURATION_SEC:-30.0}"
TEXT_SOURCE_POLICY="${TEXT_SOURCE_POLICY:-different_speaker}"
INFER_SPEAKER_FROM="${INFER_SPEAKER_FROM:-auto}"
ALLOW_MISSING_AUDIO="${ALLOW_MISSING_AUDIO:-false}"
ALLOW_MISSING_SPEAKER="${ALLOW_MISSING_SPEAKER:-false}"
AUGMENT_SOURCE_JSONL="${AUGMENT_SOURCE_JSONL:-true}"
REQUIRE_TARGET_AUDIO="${REQUIRE_TARGET_AUDIO:-true}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100000}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-true}"
RUN_SEEDVC="${RUN_SEEDVC:-true}"
SEEDVC_ROUTE_ROOT="${SEEDVC_ROUTE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes}"
SEEDVC_SHARDED_RUNNER="${SEEDVC_SHARDED_RUNNER:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/run_seedvc_jobs_sharded.sh}"
SEED_VC_DIR="${SEED_VC_DIR:-$SEEDVC_ROUTE_ROOT/third_party/seed-vc}"
SEEDVC_PY="${SEEDVC_PY:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python}"
SEEDVC_DEPS_DIR="${SEEDVC_DEPS_DIR:-$SEEDVC_ROUTE_ROOT/.deps/seedvc}"
SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-${GPU_IDS:-}}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-0}"
SEEDVC_MAX_JOBS="${SEEDVC_MAX_JOBS:-0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
SEEDVC_FP16="${SEEDVC_FP16:-true}"
SEEDVC_SKIP_EXISTING="${SEEDVC_SKIP_EXISTING:-1}"
SEEDVC_FAIL_FAST="${SEEDVC_FAIL_FAST:-0}"
SEEDVC_SHOW_MODEL_OUTPUT="${SEEDVC_SHOW_MODEL_OUTPUT:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p \
  "$(dirname "$OUTPUT_JSONL")" \
  "$(dirname "$SUMMARY_JSON")" \
  "$(dirname "$PLAN_JSONL")" \
  "$(dirname "$PLAN_SUMMARY_JSON")" \
  "$(dirname "$SEEDVC_JOBS_JSONL")" \
  "$(dirname "$SEEDVC_RESULTS_JSONL")" \
  "$LOG_DIR"

if { [ "$RESUME_EXISTING" = "1" ] || [ "$RESUME_EXISTING" = "true" ] || [ "$RESUME_EXISTING" = "TRUE" ]; } \
  && [ "$FORCE" != "1" ] \
  && [ -s "$SEEDVC_RESULTS_JSONL" ]; then
  results_stem="${SEEDVC_RESULTS_JSONL%.jsonl}"
  SEEDVC_RESULTS_JSONL="${results_stem}.resume_$(date -u +%Y%m%d-%H%M%S).jsonl"
fi

if [ -s "$OUTPUT_JSONL" ] && [ "$FORCE" != "1" ]; then
  echo "Output already exists and FORCE!=1: $OUTPUT_JSONL" >&2
  echo "Set FORCE=1 to overwrite." >&2
  exit 2
fi

echo "=========================================="
echo "Large mixed-mode VC manifest build"
echo "  ROOT=$ROOT"
echo "  PY=$PY"
echo "  INPUT_ROOT=$INPUT_ROOT"
echo "  OUTPUT_JSONL=$OUTPUT_JSONL"
echo "  SUMMARY_JSON=$SUMMARY_JSON"
echo "  PLAN_JSONL=$PLAN_JSONL"
echo "  PLAN_SUMMARY_JSON=$PLAN_SUMMARY_JSON"
echo "  SEEDVC_JOBS_JSONL=$SEEDVC_JOBS_JSONL"
echo "  SEEDVC_RESULTS_JSONL=$SEEDVC_RESULTS_JSONL"
echo "  SEEDVC_OUTPUT_ROOT=$SEEDVC_OUTPUT_ROOT"
echo "  RUN_NAME=$RUN_NAME"
echo "  EMIT_PAIR_TYPES=$EMIT_PAIR_TYPES"
echo "  LANGUAGES=$LANGUAGES"
echo "  MAX_ROWS=$MAX_ROWS"
echo "  MAX_PAIRS=$MAX_PAIRS"
echo "  TEXT_SOURCE_POLICY=$TEXT_SOURCE_POLICY"
echo "  INFER_SPEAKER_FROM=$INFER_SPEAKER_FROM"
echo "  REQUIRE_TARGET_AUDIO=$REQUIRE_TARGET_AUDIO"
echo "  RESUME_EXISTING=$RESUME_EXISTING"
echo "  RUN_SEEDVC=$RUN_SEEDVC"
echo "  SEEDVC_SHARDED_RUNNER=$SEEDVC_SHARDED_RUNNER"
echo "  SEED_VC_DIR=$SEED_VC_DIR"
echo "  SEEDVC_PY=$SEEDVC_PY"
echo "  SEEDVC_GPU_IDS=${SEEDVC_GPU_IDS:-auto}"
echo "  SEEDVC_SHARD_COUNT=$SEEDVC_SHARD_COUNT"
echo "  SEEDVC_MAX_JOBS=$SEEDVC_MAX_JOBS"
echo "=========================================="

cd "$ROOT"
"$PY" -m py_compile scripts/001013_build_large_mixed_mode_manifest.py

prepare_cmd=(
  "$PY" scripts/001013_build_large_mixed_mode_manifest.py
  --input-jsonl "$INPUT_ROOT"
  --output-jsonl "$PLAN_JSONL"
  --summary-json "$PLAN_SUMMARY_JSON"
  --seedvc-jobs-jsonl "$SEEDVC_JOBS_JSONL"
  --seedvc-output-root "$SEEDVC_OUTPUT_ROOT"
  --run-name "$RUN_NAME"
  --languages "$LANGUAGES"
  --emit-pair-types "$EMIT_PAIR_TYPES"
  --min-duration-sec "$MIN_DURATION_SEC"
  --max-duration-sec "$MAX_DURATION_SEC"
  --text-source-policy "$TEXT_SOURCE_POLICY"
  --infer-speaker-from "$INFER_SPEAKER_FROM"
  --allow-missing-audio "$ALLOW_MISSING_AUDIO"
  --allow-missing-speaker "$ALLOW_MISSING_SPEAKER"
  --augment-source-jsonl "$AUGMENT_SOURCE_JSONL"
  --require-target-audio false
  --progress-every "$PROGRESS_EVERY"
  --seed "$SEED"
)
if [ "$MAX_ROWS" -gt 0 ]; then
  prepare_cmd+=(--max-rows "$MAX_ROWS")
fi
if [ "$MAX_PAIRS" -gt 0 ]; then
  prepare_cmd+=(--max-pairs "$MAX_PAIRS")
fi

echo "=========================================="
echo "Stage 1/3: build Seed-VC plan and jobs"
echo "=========================================="
if { [ "$RESUME_EXISTING" = "1" ] || [ "$RESUME_EXISTING" = "true" ] || [ "$RESUME_EXISTING" = "TRUE" ]; } \
  && [ "$FORCE" != "1" ] \
  && [ -s "$PLAN_JSONL" ] \
  && [ -s "$PLAN_SUMMARY_JSON" ] \
  && [ -s "$SEEDVC_JOBS_JSONL" ]; then
  echo "Resume enabled: reuse existing plan/jobs and skip Stage 1."
  echo "  PLAN_JSONL=$PLAN_JSONL"
  echo "  PLAN_SUMMARY_JSON=$PLAN_SUMMARY_JSON"
  echo "  SEEDVC_JOBS_JSONL=$SEEDVC_JOBS_JSONL"
else
  "${prepare_cmd[@]}"
fi

if [ "$RUN_SEEDVC" = "1" ] || [ "$RUN_SEEDVC" = "true" ] || [ "$RUN_SEEDVC" = "TRUE" ]; then
  if [ ! -x "$SEEDVC_PY" ]; then
    echo "Seed-VC python is not executable: $SEEDVC_PY" >&2
    exit 2
  fi
  if [ ! -d "$SEED_VC_DIR" ]; then
    echo "Seed-VC dir does not exist: $SEED_VC_DIR" >&2
    exit 2
  fi
  if [ ! -x "$SEEDVC_SHARDED_RUNNER" ]; then
    echo "Seed-VC sharded runner is not executable: $SEEDVC_SHARDED_RUNNER" >&2
    exit 2
  fi
  echo "=========================================="
  echo "Stage 2/3: run Seed-VC jobs"
  echo "=========================================="
  export HF_ENDPOINT
  export PYTHONPATH="$SEEDVC_DEPS_DIR:$SEEDVC_ROUTE_ROOT/scripts:$SEED_VC_DIR:${PYTHONPATH:-}"
  JOBS_JSONL="$SEEDVC_JOBS_JSONL" \
  RESULTS_JSONL="$SEEDVC_RESULTS_JSONL" \
  SEED_VC_DIR="$SEED_VC_DIR" \
  PY="$SEEDVC_PY" \
  SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS" \
  SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT" \
  MAX_JOBS="$SEEDVC_MAX_JOBS" \
  DIFFUSION_STEPS="$DIFFUSION_STEPS" \
  LENGTH_ADJUST="$LENGTH_ADJUST" \
  INFERENCE_CFG_RATE="$INFERENCE_CFG_RATE" \
  FP16="$SEEDVC_FP16" \
  SKIP_EXISTING="$SEEDVC_SKIP_EXISTING" \
  FAIL_FAST="$SEEDVC_FAIL_FAST" \
  SHOW_MODEL_OUTPUT="$SEEDVC_SHOW_MODEL_OUTPUT" \
  bash "$SEEDVC_SHARDED_RUNNER"
else
  echo "RUN_SEEDVC=$RUN_SEEDVC, skip Stage 2/3."
fi

collect_cmd=(
  "$PY" scripts/001013_build_large_mixed_mode_manifest.py
  --input-jsonl "$INPUT_ROOT"
  --output-jsonl "$OUTPUT_JSONL"
  --summary-json "$SUMMARY_JSON"
  --seedvc-output-root "$SEEDVC_OUTPUT_ROOT"
  --run-name "$RUN_NAME"
  --languages "$LANGUAGES"
  --emit-pair-types "$EMIT_PAIR_TYPES"
  --min-duration-sec "$MIN_DURATION_SEC"
  --max-duration-sec "$MAX_DURATION_SEC"
  --text-source-policy "$TEXT_SOURCE_POLICY"
  --infer-speaker-from "$INFER_SPEAKER_FROM"
  --allow-missing-audio "$ALLOW_MISSING_AUDIO"
  --allow-missing-speaker "$ALLOW_MISSING_SPEAKER"
  --augment-source-jsonl "$AUGMENT_SOURCE_JSONL"
  --require-target-audio "$REQUIRE_TARGET_AUDIO"
  --progress-every "$PROGRESS_EVERY"
  --seed "$SEED"
)
if [ "$MAX_ROWS" -gt 0 ]; then
  collect_cmd+=(--max-rows "$MAX_ROWS")
fi
if [ "$MAX_PAIRS" -gt 0 ]; then
  collect_cmd+=(--max-pairs "$MAX_PAIRS")
fi

echo "=========================================="
echo "Stage 3/3: collect final trainable VC manifest"
echo "=========================================="
"${collect_cmd[@]}"

"$PY" - "$OUTPUT_JSONL" "$SUMMARY_JSON" "$SEEDVC_JOBS_JSONL" "$REQUIRE_TARGET_AUDIO" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
jobs_path = Path(sys.argv[3])
require_target_audio = sys.argv[4].lower() in {"1", "true", "yes", "y", "on"}
required = {
    "sample_id",
    "source_audio",
    "timbre_ref_audio",
    "target_audio",
    "source_text",
    "target_text",
    "language",
    "source_speaker_id",
    "timbre_ref_speaker_id",
    "target_speaker_id",
    "source_gender",
    "timbre_ref_gender",
    "target_gender",
    "pair_type",
}
rows = 0
missing = []
text_bad = []
pair_counts = {}
lang_counts = {}
sample_path_missing = []
source_eq_target = []
with jsonl.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        if rows <= 1000:
            for key in required:
                if key not in row:
                    missing.append((rows, key))
            check_keys = ("source_audio", "timbre_ref_audio", "target_audio") if require_target_audio else ("source_audio", "timbre_ref_audio")
            for key in check_keys:
                if not Path(row[key]).exists():
                    sample_path_missing.append((rows, key, row[key]))
        if row.get("source_audio") == row.get("target_audio"):
            source_eq_target.append(rows)
        if row.get("pair_type") == "text_prosody" and not row.get("target_text"):
            text_bad.append(rows)
        if row.get("pair_type") == "text_prosody" and row.get("target_text") != row.get("source_text"):
            text_bad.append(rows)
        pair_counts[row.get("pair_type")] = pair_counts.get(row.get("pair_type"), 0) + 1
        lang_counts[row.get("language")] = lang_counts.get(row.get("language"), 0) + 1

summary = json.loads(summary_path.read_text(encoding="utf-8"))
jobs = sum(1 for line in jobs_path.open("r", encoding="utf-8") if line.strip()) if jobs_path.exists() else 0
print("validation_rows", rows)
print("validation_seedvc_jobs", jobs)
print("validation_pair_counts", pair_counts)
print("validation_language_counts", lang_counts)
print("summary_stats", json.dumps(summary.get("stats", {}), ensure_ascii=False, sort_keys=True))
if missing:
    raise SystemExit(f"missing required fields in first 1000 rows: {missing[:10]}")
if sample_path_missing:
    raise SystemExit(f"missing sampled audio paths: {sample_path_missing[:5]}")
if source_eq_target:
    raise SystemExit(f"source_audio equals target_audio in rows: {source_eq_target[:10]}")
if text_bad:
    raise SystemExit(f"text_prosody rows with bad target_text: {text_bad[:10]}")
PY

echo "Done: $OUTPUT_JSONL"
echo "Seed-VC jobs: $SEEDVC_JOBS_JSONL"
echo "Seed-VC results: $SEEDVC_RESULTS_JSONL"
