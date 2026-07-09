#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

RUN_DIR="${RUN_DIR:-$ROOT/outputs/lora_runs/ver2_8_timbre_repair_recipe_final_varlen_block_permuted_cref_prompt_a4_refsup_cosramp_infonce_dropout_steps30000}"
RUN_LABEL="${RUN_LABEL:-codecvc-ver2-8-timbre-repair-recipe_final_varlen_block_permuted_cref_prompt_a4_refsup_cosramp_infonce_dropout}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/testset/outputs/ver2_8_timbre_repair_quick_eval}"
DOCS_MD="${DOCS_MD:-$ROOT/docs/ver2_8_recipe_final_per_save_quick20_20260705.md}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT/docs/assets/ver2_8_recipe_final_per_save_quick20_20260705.json}"
SUMMARY_MD="${SUMMARY_MD:-$ROOT/docs/ver2_8_recipe_final_per_save_quick20_20260705.md}"
CHECKPOINTS="${CHECKPOINTS:-}"
SEED="${SEED:-1234}"
FORCE="${FORCE:-0}"
if [ -z "${QUICK_GPU_COUNT:-}" ]; then
  detected_gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  if [ -z "$detected_gpus" ] || [ "$detected_gpus" = "0" ]; then
    detected_gpus=1
  fi
  if [ "$detected_gpus" -gt 4 ]; then
    detected_gpus=4
  fi
  QUICK_GPU_COUNT="$detected_gpus"
fi
QUICK_NUM_SHARDS="${QUICK_NUM_SHARDS:-$QUICK_GPU_COUNT}"
QUICK_ASR_NUM_SHARDS="${QUICK_ASR_NUM_SHARDS:-$QUICK_GPU_COUNT}"

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR not found: $RUN_DIR" >&2
  exit 1
fi

if [ -z "$CHECKPOINTS" ]; then
  mapfile -t checkpoint_paths < <(find "$RUN_DIR" -maxdepth 1 -type d -name 'step-*' | sort -V)
else
  old_ifs="$IFS"
  IFS=','
  checkpoint_paths=()
  for item in $CHECKPOINTS; do
    if [ -n "$item" ]; then
      case "$item" in
        /*) checkpoint_paths+=("$item") ;;
        *) checkpoint_paths+=("$RUN_DIR/$item") ;;
      esac
    fi
  done
  IFS="$old_ifs"
fi

if [ "${#checkpoint_paths[@]}" -eq 0 ]; then
  echo "[recipe-final-quick20] no checkpoints found under $RUN_DIR"
  "$ROOT/scripts/004062_summarize_recipe_final_per_save_quick20.py" \
    --run-dir "$RUN_DIR" \
    --eval-root "$EVAL_ROOT" \
    --run-label "$RUN_LABEL" \
    --seed "$SEED" \
    --output-json "$SUMMARY_JSON" \
    --output-md "$SUMMARY_MD"
  exit 0
fi

for checkpoint in "${checkpoint_paths[@]}"; do
  if [ ! -d "$checkpoint" ]; then
    echo "ERROR: checkpoint not found: $checkpoint" >&2
    exit 1
  fi
  step_label=$(basename "$checkpoint")
  step_num=${step_label#step-}
  if [ $((step_num % 2000)) -ne 0 ]; then
    echo "[recipe-final-quick20] skip non-save checkpoint: $step_label"
    continue
  fi

  run_id="${RUN_LABEL}_${step_label}_quick20_d2d3_seed${SEED}"
  output_dir="$EVAL_ROOT/$run_id"
  if [ "$FORCE" != "1" ] \
    && [ -f "$output_dir/${run_id}.asr_eval.jsonl" ] \
    && [ -f "$output_dir/${run_id}.speaker_sim_summary.json" ] \
    && [ -f "$output_dir/${run_id}.ref_content_similarity_summary.json" ]; then
    echo "[recipe-final-quick20] skip existing complete output: $run_id"
    continue
  fi

  echo "[recipe-final-quick20] run $step_label -> $run_id"
  RUN_DIR="$RUN_DIR" \
  RUN_LABEL="$RUN_LABEL" \
  EVAL_ROOT="$EVAL_ROOT" \
  DOCS_MD="$DOCS_MD" \
  CHECKPOINTS="$step_label" \
  SEED="$SEED" \
  QUICK_GPU_COUNT="$QUICK_GPU_COUNT" \
  QUICK_NUM_SHARDS="$QUICK_NUM_SHARDS" \
  QUICK_ASR_NUM_SHARDS="$QUICK_ASR_NUM_SHARDS" \
  REF_PROMPT_CODEC_PERMUTATION="1" \
  REF_PROMPT_CODEC_PERMUTATION_MODE="block_shuffle" \
  REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS="8.0" \
  REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS="8.0" \
  REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE="12.5" \
  REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS="0.4" \
  REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP="block" \
  TIMBRE_CFG_SCALE="1.0" \
  RUN_T11="0" \
  TIMBRE_SIDE_ONLY="0" \
  bash "$ROOT/scripts/004054_run_ver2_8_timbre_quick_eval.sh"
done

"$ROOT/scripts/004062_summarize_recipe_final_per_save_quick20.py" \
  --run-dir "$RUN_DIR" \
  --eval-root "$EVAL_ROOT" \
  --run-label "$RUN_LABEL" \
  --seed "$SEED" \
  --output-json "$SUMMARY_JSON" \
  --output-md "$SUMMARY_MD"
