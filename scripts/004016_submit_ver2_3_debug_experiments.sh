#!/usr/bin/env sh
set -eu

# Submit controlled Ver2.3 regression-debug experiments.
# Usage:
#   sh scripts/004016_submit_ver2_3_debug_experiments.sh tiny32 --dry-run
#   sh scripts/004016_submit_ver2_3_debug_experiments.sh ablation_b_ctc
#   sh scripts/004016_submit_ver2_3_debug_experiments.sh ablation_all --dry-run

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
VALID_DIR="${VALID_DIR:-$ROOT/testset/validation/ver2_3_debug}"
LOSS_VALID_JSONL="${LOSS_VALID_JSONL:-$VALID_DIR/moss_codecvc_ver2_3_loss_valid_160.jsonl}"
TINY32_JSONL="${TINY32_JSONL:-$VALID_DIR/moss_codecvc_ver2_3_tiny_overfit_32.jsonl}"
TINY128_JSONL="${TINY128_JSONL:-$VALID_DIR/moss_codecvc_ver2_3_tiny_overfit_128.jsonl}"
TINY32_NO_TEXT_JSONL="${TINY32_NO_TEXT_JSONL:-$VALID_DIR/moss_codecvc_ver2_3_tiny_overfit_no_text_32.jsonl}"
TINY128_NO_TEXT_JSONL="${TINY128_NO_TEXT_JSONL:-$VALID_DIR/moss_codecvc_ver2_3_tiny_overfit_no_text_128.jsonl}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$ROOT/scripts/002004_submit_ver2_lora_68w_h200_qz.sh}"

NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-zh45w_en22w_no_text}"
TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"
NO_TEXT_TRAIN_JSONL_DEFAULT="$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl"
TEXT_TRAIN_JSONL_DEFAULT="$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.ctc_clean.jsonl"

TEXT_REPEAT="${TEXT_REPEAT:-5}"
DEBUG_OUT_ROOT="${DEBUG_OUT_ROOT:-$ROOT/outputs/lora_runs/ver2_3_debug}"
ABLATION_STEPS="${ABLATION_STEPS:-2000}"
TINY_STEPS="${TINY_STEPS:-800}"
ABLATION_SAVE_STEPS="${ABLATION_SAVE_STEPS:-500}"
TINY_SAVE_STEPS="${TINY_SAVE_STEPS:-200}"

DRY_RUN=0
EXPERIMENT="${1:-}"
if [ -z "$EXPERIMENT" ]; then
  echo "Usage: sh $0 {tiny32|tiny128|tiny_no_text32|tiny_no_text128|ablation_a_ce_route|ablation_b_ctc|ablation_b_ctc_headlr10|ablation_c_hubert|ablation_c_progress_stop|ablation_d_prosody|ablation_d_prosody_stop|ablation_e_progress_stop_only|ablation_all|ablation_control_all} [--dry-run]" >&2
  exit 2
fi
shift || true
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

run_submit() {
  exp_name="$1"
  train_spec="$2"
  max_steps="$3"
  save_steps="$4"
  content_ctc="$5"
  semantic="$6"
  lambda_prosody="$7"
  progress="$8"
  stop="$9"
  target_spk="${10}"
  source_sup="${11}"
  prosody_duration="${12:-0.5}"
  prosody_energy="${13:-0.5}"
  prosody_pause="${14:-1.0}"
  resume_adapter="${15:-}"
  if [ -z "$resume_adapter" ] && [ -n "${RESUME_OVERRIDE:-}" ]; then
    resume_adapter="$RESUME_OVERRIDE"
  fi
  out_dir="$DEBUG_OUT_ROOT/$exp_name"
  job_prefix="codecvc-ver2-3-debug-$exp_name"
  job_stamp="$(date +%Y%m%d-%H%M%S)-$exp_name-$$"

  echo "[debug-exp] exp=$exp_name"
  echo "[debug-exp] train_spec=$train_spec"
  echo "[debug-exp] out_dir=$out_dir"
  echo "[debug-exp] eval=$LOSS_VALID_JSONL save_steps=$save_steps max_steps=$max_steps"
  echo "[debug-exp] weights content_ctc=$content_ctc semantic=$semantic prosody=$lambda_prosody progress=$progress stop=$stop spk=$target_spk srcsup=$source_sup prosody_duration=$prosody_duration prosody_energy=$prosody_energy prosody_pause=$prosody_pause ctc_head_lr_mult=${CONTENT_CTC_HEAD_LR_MULTIPLIER:-1.0}"
  if [ -n "$resume_adapter" ]; then
    echo "[debug-exp] resume_adapter=$resume_adapter"
  fi

  if [ "$DRY_RUN" = "1" ]; then
    dry_arg="--dry-run"
  else
    dry_arg=""
  fi

  ROOT="$ROOT" \
  NO_TEXT_TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL_DEFAULT" \
  TEXT_TRAIN_JSONL="$TEXT_TRAIN_JSONL_DEFAULT" \
  TRAIN_JSONL_SPEC="$train_spec" \
  OUT_DIR="$out_dir" \
  QZ_RECORD_ROOT="$ROOT/trainset/qz_jobs/$job_stamp" \
  JOB_NAME="$job_prefix-$job_stamp" \
  JOB_NAME_PREFIX="$job_prefix" \
  TEXT_REPEAT="$TEXT_REPEAT" \
  MAX_TRAIN_STEPS="$max_steps" \
  SAVE_STEPS="$save_steps" \
  EVAL_JSONL="$LOSS_VALID_JSONL" \
  EVAL_STEPS="0" \
  EVAL_MAX_BATCHES="0" \
  EVAL_NUM_WORKERS="0" \
  CONTENT_CTC_WEIGHT="$content_ctc" \
  SEMANTIC_LOSS_WEIGHT="$semantic" \
  LAMBDA_PROSODY="$lambda_prosody" \
  PROSODY_DURATION_WEIGHT="$prosody_duration" \
  PROSODY_ENERGY_WEIGHT="$prosody_energy" \
  PROSODY_PAUSE_WEIGHT="$prosody_pause" \
  RESUME_ADAPTER_PATH="$resume_adapter" \
  CONTENT_CTC_HEAD_LR_MULTIPLIER="${CONTENT_CTC_HEAD_LR_MULTIPLIER:-1.0}" \
  PROGRESS_LOSS_WEIGHT="$progress" \
  STOP_LOSS_WEIGHT="$stop" \
  TARGET_SPK_WEIGHT="$target_spk" \
  SOURCE_SUPPRESS_WEIGHT="$source_sup" \
  sh "$SUBMIT_SCRIPT" $dry_arg
}

require_file() {
  if [ ! -f "$1" ]; then
    echo "ERROR: missing required file: $1" >&2
    echo "Run: $ROOT/scripts/004015_build_ver2_3_debug_valid_splits.py" >&2
    exit 1
  fi
}

require_file "$LOSS_VALID_JSONL"
require_file "$NO_TEXT_TRAIN_JSONL_DEFAULT"
require_file "$TEXT_TRAIN_JSONL_DEFAULT"

case "$EXPERIMENT" in
  tiny32)
    require_file "$TINY32_JSONL"
    run_submit "tiny32_overfit" "$TINY32_JSONL::repeat=1" "$TINY_STEPS" "$TINY_SAVE_STEPS" 0.02 0.03 0.03 0.03 0.03 0.05 0.05
    ;;
  tiny128)
    require_file "$TINY128_JSONL"
    run_submit "tiny128_overfit" "$TINY128_JSONL::repeat=1" "$TINY_STEPS" "$TINY_SAVE_STEPS" 0.02 0.03 0.03 0.03 0.03 0.05 0.05
    ;;
  tiny_no_text32)
    require_file "$TINY32_NO_TEXT_JSONL"
    run_submit "tiny_no_text32_ce_ctc_semantic" "$TINY32_NO_TEXT_JSONL::repeat=1" "$TINY_STEPS" "$TINY_SAVE_STEPS" 0.03 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
    ;;
  tiny_no_text128)
    require_file "$TINY128_NO_TEXT_JSONL"
    run_submit "tiny_no_text128_ce_ctc_semantic" "$TINY128_NO_TEXT_JSONL::repeat=1" "$TINY_STEPS" "$TINY_SAVE_STEPS" 0.03 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
    ;;
  ablation_a_ce_route)
    run_submit "ablation_a_ce_route" "$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1,$TEXT_TRAIN_JSONL_DEFAULT::repeat=$TEXT_REPEAT" "$ABLATION_STEPS" "$ABLATION_SAVE_STEPS" 0.00 0.00 0.00 0.00 0.00 0.00 0.00
    ;;
  ablation_b_ctc)
    run_submit "ablation_b_ctc" "$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1,$TEXT_TRAIN_JSONL_DEFAULT::repeat=$TEXT_REPEAT" "$ABLATION_STEPS" "$ABLATION_SAVE_STEPS" 0.02 0.00 0.00 0.00 0.00 0.00 0.00
    ;;
  ablation_b_ctc_headlr10)
    CONTENT_CTC_HEAD_LR_MULTIPLIER=10 run_submit "ablation_b_ctc_headlr10" "$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1,$TEXT_TRAIN_JSONL_DEFAULT::repeat=$TEXT_REPEAT" "$ABLATION_STEPS" "$ABLATION_SAVE_STEPS" 0.02 0.00 0.00 0.00 0.00 0.00 0.00
    ;;
  ablation_c_hubert)
    run_submit "ablation_c_hubert" "$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1,$TEXT_TRAIN_JSONL_DEFAULT::repeat=$TEXT_REPEAT" "$ABLATION_STEPS" "$ABLATION_SAVE_STEPS" 0.02 0.03 0.00 0.00 0.00 0.00 0.00
    ;;
  ablation_c_progress_stop)
    run_submit "ablation_c_progress_stop" "$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1,$TEXT_TRAIN_JSONL_DEFAULT::repeat=$TEXT_REPEAT" "$ABLATION_STEPS" "$ABLATION_SAVE_STEPS" 0.02 0.03 0.00 0.03 0.03 0.00 0.00
    ;;
  ablation_e_progress_stop_only)
    run_submit "ablation_e_progress_stop_only" "$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1,$TEXT_TRAIN_JSONL_DEFAULT::repeat=$TEXT_REPEAT" "$ABLATION_STEPS" "$ABLATION_SAVE_STEPS" 0.00 0.00 0.00 0.03 0.03 0.00 0.00 0.00 0.00 0.00
    ;;
  ablation_d_prosody)
    run_submit "ablation_d_prosody" "$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1,$TEXT_TRAIN_JSONL_DEFAULT::repeat=$TEXT_REPEAT" "$ABLATION_STEPS" "$ABLATION_SAVE_STEPS" 0.02 0.03 0.03 0.03 0.03 0.00 0.00 0.5 0.5 1.0
    ;;
  ablation_d_prosody_stop)
    run_submit "ablation_d_prosody_stop" "$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1,$TEXT_TRAIN_JSONL_DEFAULT::repeat=$TEXT_REPEAT" "$ABLATION_STEPS" "$ABLATION_SAVE_STEPS" 0.02 0.03 0.03 0.03 0.03 0.00 0.00
    ;;
  ablation_all)
    for exp in ablation_a_ce_route ablation_b_ctc ablation_c_hubert ablation_c_progress_stop ablation_d_prosody; do
      if [ "$DRY_RUN" = "1" ]; then
        sh "$0" "$exp" --dry-run
      else
        sh "$0" "$exp"
      fi
    done
    ;;
  ablation_control_all)
    for exp in tiny_no_text128 ablation_a_ce_route ablation_b_ctc ablation_c_hubert ablation_c_progress_stop ablation_d_prosody; do
      if [ "$DRY_RUN" = "1" ]; then
        sh "$0" "$exp" --dry-run
      else
        sh "$0" "$exp"
      fi
    done
    ;;
  *)
    echo "ERROR: unsupported experiment: $EXPERIMENT" >&2
    exit 2
    ;;
esac
