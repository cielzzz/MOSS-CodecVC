#!/usr/bin/env bash
# Run Batch-22 full-v2 per-checkpoint quick20 eval on Case A/B/C.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/lora_runs/ver2_9_v1_v2full_cross_attn_lite_steps3000}"
STEP="${STEP:-}"
MODEL_PATH="${MODEL_PATH:-}"
RUN_LABEL="${RUN_LABEL:-v1_v2full_cross_attn_lite}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/testset/outputs/ver2_9_v2full_step_quick_eval}"
SEED="${SEED:-1234}"

CASE_A_JSONL="${CASE_A_JSONL:-$ROOT/testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
CASE_B_JSONL="${CASE_B_JSONL:-$ROOT/trainset/ver2_9_prepared_v2_pilot_10k_20260708/no_text.seen_valid.quick20_10zh10en.eval.jsonl}"
CASE_C_JSONL="${CASE_C_JSONL:-$ROOT/trainset/ver2_9_prepared_v2_pilot_10k_20260708/no_text.unseen_valid.quick20_10zh10en.eval.jsonl}"

GPU_COUNT="${GPU_COUNT:-4}"
NUM_SHARDS="${NUM_SHARDS:-$GPU_COUNT}"
ASR_NUM_SHARDS="${ASR_NUM_SHARDS:-$GPU_COUNT}"
BUILD_PAGE="${BUILD_PAGE:-0}"
RUN_ASR="${RUN_ASR:-1}"
OVERWRITE_INFER="${OVERWRITE_INFER:-1}"
RESET_MANIFESTS="${RESET_MANIFESTS:-1}"

if [ -z "$MODEL_PATH" ]; then
  if [ -z "$STEP" ]; then
    echo "ERROR: set STEP=500 or MODEL_PATH=/path/to/step-500" >&2
    exit 2
  fi
  MODEL_PATH="$RUN_DIR/step-$STEP"
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: checkpoint does not exist: $MODEL_PATH" >&2
  exit 1
fi
for required in "$CASE_A_JSONL" "$CASE_B_JSONL" "$CASE_C_JSONL"; do
  if [ ! -f "$required" ]; then
    echo "ERROR: missing validation jsonl: $required" >&2
    exit 1
  fi
done

step_label=$(basename "$MODEL_PATH")
mkdir -p "$EVAL_ROOT"

run_case() {
  case_key="$1"
  title="$2"
  validation_jsonl="$3"
  run_id="${RUN_LABEL}_case${case_key}_${step_label}_quick20_d2d3_seed${SEED}"
  output_dir="$EVAL_ROOT/$run_id"
  echo "[v2full-abc-eval] case=$case_key title=$title model=$MODEL_PATH output=$output_dir"
  SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH=0.0 \
  TEMPERATURE=0.7 \
  NO_TEXT_AUDIO_TEMPERATURE=1.1 \
  NO_TEXT_AUDIO_TOP_P=0.7 \
  NO_TEXT_AUDIO_TOP_K=20 \
  AUDIO_TEMPERATURE=1.1 \
  AUDIO_TOP_P=0.7 \
  AUDIO_TOP_K=20 \
  TIMBRE_SIDE_ONLY=1 \
  PYTHON="$PYTHON" \
  ASR_PYTHON="$ASR_PYTHON" \
  VALIDATION_JSONL="$validation_jsonl" \
  MODEL_PATH="$MODEL_PATH" \
  RUN_ID="$run_id" \
  RUN_LABEL="$run_id" \
  OUTPUT_DIR="$output_dir" \
  MODE=no_text \
  MAX_CASES=0 \
  DECODING_PROFILE=default \
  PERSISTENT_INFER=1 \
  OVERWRITE_INFER="$OVERWRITE_INFER" \
  RESET_MANIFESTS="$RESET_MANIFESTS" \
  RUN_ASR="$RUN_ASR" \
  RUN_SUMMARY=1 \
  BUILD_PAGE="$BUILD_PAGE" \
  PAGE_DIR="$output_dir/listening_page" \
  GPU_COUNT="$GPU_COUNT" \
  NUM_SHARDS="$NUM_SHARDS" \
  ASR_NUM_SHARDS="$ASR_NUM_SHARDS" \
  SEED="$SEED" \
  bash "$ROOT/scripts/004039_run_seedtts_validation_eval.sh"

  "$PYTHON" "$ROOT/scripts/004056_summarize_seedtts_ref_content_similarity.py" \
    --asr-jsonl "$output_dir/${run_id}.asr_eval.jsonl" \
    --output-json "$output_dir/${run_id}.ref_content_similarity_summary.json" \
    --output-md "$output_dir/${run_id}.ref_content_similarity_summary.md"

  "$PYTHON" "$ROOT/scripts/004050_summarize_seedtts_speaker_sim_only.py" \
    --validation-jsonl "$validation_jsonl" \
    --run "$run_id=$output_dir" \
    --output-csv "$output_dir/${run_id}.speaker_sim.csv" \
    --summary-json "$output_dir/${run_id}.speaker_sim_summary.json" \
    --summary-md "$output_dir/${run_id}.speaker_sim_summary.md" \
    --speaker-device cuda:0
}

run_case "A" "old_seedtts" "$CASE_A_JSONL"
run_case "B" "v2_same_episode" "$CASE_B_JSONL"
run_case "C" "v2_heldout_cross_channel" "$CASE_C_JSONL"

"$PYTHON" - "$EVAL_ROOT" "$RUN_LABEL" "$step_label" <<'PY'
import csv
import json
import sys
from pathlib import Path

eval_root = Path(sys.argv[1])
run_label = sys.argv[2]
step_label = sys.argv[3]

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def f(v):
    try:
        return float(v)
    except Exception:
        return None

print(f"[v2full-abc-eval] rollup step={step_label}")
for key in ["A", "B", "C"]:
    prefix = f"{run_label}_case{key}_{step_label}_quick20_d2d3_seed"
    dirs = sorted(p for p in eval_root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not dirs:
        print(f"case{key}: missing")
        continue
    d = dirs[-1]
    run_id = d.name
    summary = load_json(d / f"{run_id}.summary.json")
    sim_summary = load_json(d / f"{run_id}.speaker_sim_summary.json")
    overall = summary.get("overall") or {}
    sim_scope = {}
    if sim_summary.get("runs"):
        sim_scope = next(iter(sim_summary["runs"].values())).get("all") or {}
    ref_bound = None
    sim_csv = d / f"{run_id}.speaker_sim.csv"
    if sim_csv.exists():
        rows = list(csv.DictReader(sim_csv.open()))
        ref_bound = sum(float(r["sim_gen_ref"]) > float(r["sim_gen_source"]) for r in rows)
    print(
        f"case{key}: primary={overall.get('primary_error')} CER={overall.get('cer')} "
        f"keep={overall.get('keep')}/{overall.get('n')} sim_ref={sim_scope.get('sim_gen_ref_mean')} "
        f"sim_src={sim_scope.get('sim_gen_source_mean')} ref_bound={ref_bound}/{sim_scope.get('n')}"
    )
PY

echo "[v2full-abc-eval] done eval_root=$EVAL_ROOT step=$step_label"
