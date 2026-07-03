#!/usr/bin/env sh
set -eu

# Submit Ver2.5 SourceSemanticMemory ablations.
#
# Usage:
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_s1_source_semantic
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_s2_source_semantic_progress
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_s1_from_a_ce_route_2k
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_ctc_only_headlr10
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_source_semantic_ctc_headlr10
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_s1_from_a_gate_strong_2k
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_s1_from_a_adapter_warmup_stage1_1k
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_s1_from_a_adapter_warmup_stage1_notext_1k
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_s1_from_a_adapter_warmup_stage1_notext_savefix_1k
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_s1_from_a_posbias_2k
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_p0a_text_token_memory_2k
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_p0a_text_token_memory_textgate1_2k
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_p0c_codec_bottleneck_2k
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_selected_p0a_text_token_memory_5k
#   sh scripts/004029_submit_ver2_5_source_semantic_experiments.sh ver2_5_all --dry-run

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$ROOT/scripts/002004_submit_ver2_lora_68w_h200_qz.sh}"

NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-zh45w_en22w_no_text}"
TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"
NO_TEXT_TRAIN_JSONL_DEFAULT="$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl"
TEXT_TRAIN_JSONL_DEFAULT="$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.ctc_clean.jsonl"
LOSS_VALID_JSONL="${LOSS_VALID_JSONL:-$ROOT/testset/validation/ver2_3_debug/moss_codecvc_ver2_3_loss_valid_160.jsonl}"

TEXT_REPEAT="${TEXT_REPEAT:-5}"
TRAIN_SPEC_OVERRIDE="${TRAIN_SPEC_OVERRIDE:-}"
DEBUG_OUT_ROOT="${DEBUG_OUT_ROOT:-$ROOT/outputs/lora_runs/ver2_5_debug_5k}"
ABLATION_STEPS="${ABLATION_STEPS:-5000}"
ABLATION_SAVE_STEPS="${ABLATION_SAVE_STEPS:-1000}"
SOURCE_SEMANTIC_LAYERS="${SOURCE_SEMANTIC_LAYERS:-28,30,32,34,35}"
SOURCE_SEMANTIC_INIT_GATE="${SOURCE_SEMANTIC_INIT_GATE:--2.0}"
SOURCE_SEMANTIC_GATE_LR_MULTIPLIER="${SOURCE_SEMANTIC_GATE_LR_MULTIPLIER:-10.0}"
SOURCE_SEMANTIC_TEXT_GATE="${SOURCE_SEMANTIC_TEXT_GATE:-0.0}"
SOURCE_SEMANTIC_POSITION_SCALE="${SOURCE_SEMANTIC_POSITION_SCALE:-0.10}"
SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH="${SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH:-2.0}"
SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH="${SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH:-0.25}"
A_CE_ROUTE_CHECKPOINT="${A_CE_ROUTE_CHECKPOINT:-$ROOT/outputs/lora_runs/ver2_3_debug_resume_evalfix/ablation_a_ce_route/step-1000}"
SOURCE_UNIT_VOCAB_SIZE="${SOURCE_UNIT_VOCAB_SIZE:-0}"

DRY_RUN=0
EXPERIMENT="${1:-}"
if [ -z "$EXPERIMENT" ]; then
  echo "Usage: sh $0 {ver2_5_s1_source_semantic|ver2_5_s2_source_semantic_progress|ver2_5_s1_from_a_ce_route_2k|ver2_5_s2_from_a_ce_route_2k|ver2_5_ctc_only_headlr10|ver2_5_source_semantic_ctc_headlr10|ver2_5_s1_from_a_gate_strong_2k|ver2_5_s1_from_a_gate0_2k|ver2_5_s1_from_a_posbias_2k|ver2_5_p0a_text_token_memory_2k|ver2_5_p0a_text_token_memory_textgate1_2k|ver2_5_p0b_hubert_unit_memory_2k|ver2_5_p0c_codec_bottleneck_2k|ver2_5_selected_p0a_text_token_memory_5k|ver2_5_s1_from_a_adapter_warmup_stage1_1k|ver2_5_s1_from_a_adapter_warmup_stage1_notext_1k|ver2_5_s1_from_a_adapter_warmup_stage1_notext_savefix_1k|ver2_5_s1_from_a_adapter_warmup_joint_2k|ver2_5_all} [--dry-run]" >&2
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

require_file() {
  if [ ! -f "$1" ]; then
    echo "ERROR: missing required file: $1" >&2
    exit 1
  fi
}

run_submit() {
  exp_name="$1"
  enable_source_semantic_memory="$2"
  source_semantic_progress_weight="$3"
  content_ctc_weight="$4"
  content_ctc_head_lr_multiplier="$5"
  max_steps="${6:-$ABLATION_STEPS}"
  save_steps="${7:-$ABLATION_SAVE_STEPS}"
  resume_adapter_path="${8:-}"
  source_semantic_init_gate="${9:-$SOURCE_SEMANTIC_INIT_GATE}"
  train_source_semantic_only="${10:-0}"
  freeze_lora="${11:-0}"
  freeze_role_routing="${12:-0}"
  freeze_timbre_adapter="${13:-0}"
  source_semantic_position_scale="${14:-$SOURCE_SEMANTIC_POSITION_SCALE}"
  source_semantic_monotonic_bias_strength="${15:-$SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH}"
  source_semantic_monotonic_bias_width="${16:-$SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH}"
  source_content_memory_type="${17:-hubert_continuous}"
  source_content_vocab_size="${18:-0}"
  source_content_codec_bottleneck_dim="${19:-256}"
  source_content_codec_codebooks="${20:-first_4}"
  source_content_dedup_units="${21:-0}"
  source_semantic_text_gate="${22:-$SOURCE_SEMANTIC_TEXT_GATE}"
  out_dir="$DEBUG_OUT_ROOT/$exp_name"
  if [ -n "$TRAIN_SPEC_OVERRIDE" ]; then
    train_spec="$TRAIN_SPEC_OVERRIDE"
  else
    train_spec="$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1,$TEXT_TRAIN_JSONL_DEFAULT::repeat=$TEXT_REPEAT"
  fi
  job_stamp="$(date +%Y%m%d-%H%M%S)-$exp_name-$$"
  job_prefix="codecvc-ver2-5-debug-$exp_name"

  echo "[ver2.5-exp] exp=$exp_name"
  echo "[ver2.5-exp] train_spec=$train_spec"
  echo "[ver2.5-exp] out_dir=$out_dir"
  echo "[ver2.5-exp] enable_source_semantic_memory=$enable_source_semantic_memory"
  echo "[ver2.5-exp] source_semantic_progress_weight=$source_semantic_progress_weight"
  echo "[ver2.5-exp] content_ctc_weight=$content_ctc_weight content_ctc_head_lr_multiplier=$content_ctc_head_lr_multiplier"
  echo "[ver2.5-exp] max_steps=$max_steps save_steps=$save_steps"
  echo "[ver2.5-exp] resume_adapter_path=${resume_adapter_path:-<none>}"
  echo "[ver2.5-exp] source_semantic_init_gate=$source_semantic_init_gate"
  echo "[ver2.5-exp] source_semantic_position_scale=$source_semantic_position_scale"
  echo "[ver2.5-exp] source_semantic_monotonic_bias_strength=$source_semantic_monotonic_bias_strength"
  echo "[ver2.5-exp] source_semantic_monotonic_bias_width=$source_semantic_monotonic_bias_width"
  echo "[ver2.5-exp] source_semantic_text_gate=$source_semantic_text_gate"
  echo "[ver2.5-exp] source_content_memory_type=$source_content_memory_type"
  echo "[ver2.5-exp] source_content_vocab_size=$source_content_vocab_size"
  echo "[ver2.5-exp] source_content_codec_bottleneck_dim=$source_content_codec_bottleneck_dim"
  echo "[ver2.5-exp] source_content_codec_codebooks=$source_content_codec_codebooks"
  echo "[ver2.5-exp] source_content_dedup_units=$source_content_dedup_units"
  echo "[ver2.5-exp] train_source_semantic_only=$train_source_semantic_only freeze_lora=$freeze_lora freeze_role_routing=$freeze_role_routing freeze_timbre_adapter=$freeze_timbre_adapter"
  echo "[ver2.5-exp] text_repeat=$TEXT_REPEAT eval=$LOSS_VALID_JSONL"

  dry_arg=""
  if [ "$DRY_RUN" = "1" ]; then
    dry_arg="--dry-run"
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
  CONTENT_CTC_WEIGHT="$content_ctc_weight" \
  CONTENT_CTC_VOCAB_SIZE="0" \
  CONTENT_CTC_HEAD_LR_MULTIPLIER="$content_ctc_head_lr_multiplier" \
  SEMANTIC_LOSS_WEIGHT="0" \
  SEMANTIC_FEATURE_DIM="0" \
  LAMBDA_PROSODY="0" \
  PROGRESS_LOSS_WEIGHT="0" \
  STOP_LOSS_WEIGHT="0" \
  TARGET_SPK_WEIGHT="0" \
  SOURCE_SUPPRESS_WEIGHT="0" \
  RESUME_ADAPTER_PATH="$resume_adapter_path" \
  LAMBDA_CONTENT="0" \
  CONTENT_TOKEN_WEIGHT="0" \
  CONTENT_SOURCE_CODEC_WEIGHT="0" \
  ENABLE_SOURCE_SEMANTIC_MEMORY="$enable_source_semantic_memory" \
  SOURCE_SEMANTIC_FEATURE_DIM="768" \
  SOURCE_SEMANTIC_ADAPTER_LAYERS="$SOURCE_SEMANTIC_LAYERS" \
  SOURCE_SEMANTIC_NO_TEXT_GATE="1.0" \
  SOURCE_SEMANTIC_TEXT_GATE="$source_semantic_text_gate" \
  SOURCE_SEMANTIC_LEARNED_TEXT_GATE="0" \
  SOURCE_SEMANTIC_PROGRESS_WEIGHT="$source_semantic_progress_weight" \
  SOURCE_SEMANTIC_DROPOUT="0.1" \
  SOURCE_SEMANTIC_INIT_GATE="$source_semantic_init_gate" \
  SOURCE_SEMANTIC_POSITION_SCALE="$source_semantic_position_scale" \
  SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH="$source_semantic_monotonic_bias_strength" \
  SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH="$source_semantic_monotonic_bias_width" \
  SOURCE_CONTENT_MEMORY_TYPE="$source_content_memory_type" \
  SOURCE_CONTENT_VOCAB_SIZE="$source_content_vocab_size" \
  SOURCE_CONTENT_CODEC_BOTTLENECK_DIM="$source_content_codec_bottleneck_dim" \
  SOURCE_CONTENT_CODEC_CODEBOOKS="$source_content_codec_codebooks" \
  SOURCE_CONTENT_DEDUP_UNITS="$source_content_dedup_units" \
  SOURCE_SEMANTIC_GATE_LR_MULTIPLIER="$SOURCE_SEMANTIC_GATE_LR_MULTIPLIER" \
  TRAIN_SOURCE_SEMANTIC_ONLY="$train_source_semantic_only" \
  FREEZE_LORA="$freeze_lora" \
  FREEZE_ROLE_ROUTING="$freeze_role_routing" \
  FREEZE_TIMBRE_ADAPTER="$freeze_timbre_adapter" \
  sh "$SUBMIT_SCRIPT" $dry_arg
}

require_file "$SUBMIT_SCRIPT"
require_file "$NO_TEXT_TRAIN_JSONL_DEFAULT"
require_file "$TEXT_TRAIN_JSONL_DEFAULT"
require_file "$LOSS_VALID_JSONL"

case "$EXPERIMENT" in
  ver2_5_s1_source_semantic)
    run_submit "ver2_5_s1_source_semantic" "1" "0" "0" "1"
    ;;
  ver2_5_s2_source_semantic_progress)
    run_submit "ver2_5_s2_source_semantic_progress" "1" "0.02" "0" "1"
    ;;
  ver2_5_s1_from_a_ce_route_2k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    run_submit "ver2_5_s1_from_a_ce_route_2k" "1" "0" "0" "1" "2000" "500" "$A_CE_ROUTE_CHECKPOINT"
    ;;
  ver2_5_s2_from_a_ce_route_2k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    run_submit "ver2_5_s2_from_a_ce_route_2k" "1" "0.02" "0" "1" "2000" "500" "$A_CE_ROUTE_CHECKPOINT"
    ;;
  ver2_5_ctc_only_headlr10)
    run_submit "ver2_5_ctc_only_headlr10" "0" "0" "0.05" "10"
    ;;
  ver2_5_source_semantic_ctc_headlr10)
    run_submit "ver2_5_source_semantic_ctc_headlr10" "1" "0" "0.05" "10"
    ;;
  ver2_5_s1_from_a_gate_strong_2k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    run_submit "ver2_5_s1_from_a_gate_strong_2k" "1" "0" "0" "1" "2000" "500" "$A_CE_ROUTE_CHECKPOINT" "-1.0" "0" "0" "0" "0"
    ;;
  ver2_5_s1_from_a_gate0_2k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    run_submit "ver2_5_s1_from_a_gate0_2k" "1" "0" "0" "1" "2000" "500" "$A_CE_ROUTE_CHECKPOINT" "0.0" "0" "0" "0" "0"
    ;;
  ver2_5_s1_from_a_posbias_2k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    # Structural read-path ablation after uniform-attention diagnosis:
    # keep auxiliary losses off, but add source time position encoding and
    # target-progress-to-source-progress attention bias.
    run_submit "ver2_5_s1_from_a_posbias_2k" "1" "0" "0" "1" "2000" "500" "$A_CE_ROUTE_CHECKPOINT" "-1.0" "0" "0" "0" "0" "$SOURCE_SEMANTIC_POSITION_SCALE" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH"
    ;;
  ver2_5_p0a_text_token_memory_2k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    # Source transcript/SPM token memory. Vocab is inferred from shared
    # content_token metadata in the no_text/text manifests.
    run_submit "ver2_5_p0a_text_token_memory_2k" "1" "0" "0" "1" "2000" "500" "$A_CE_ROUTE_CHECKPOINT" "-1.0" "0" "0" "0" "0" "$SOURCE_SEMANTIC_POSITION_SCALE" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH" "text_tokens" "0" "256" "first_4" "0"
    ;;
  ver2_5_p0a_text_token_memory_textgate1_2k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    # Text-mode ablation only. Mainline keeps text_gate=0.0 because text mode
    # already has requested text in the prompt; this run tests whether adding
    # the same transcript token memory helps or overrides source/prosody.
    run_submit "ver2_5_p0a_text_token_memory_textgate1_2k" "1" "0" "0" "1" "2000" "500" "$A_CE_ROUTE_CHECKPOINT" "-1.0" "0" "0" "0" "0" "$SOURCE_SEMANTIC_POSITION_SCALE" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH" "text_tokens" "0" "256" "first_4" "0" "1.0"
    ;;
  ver2_5_p0b_hubert_unit_memory_2k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    if [ "$SOURCE_UNIT_VOCAB_SIZE" -le 1 ]; then
      echo "ERROR: ver2_5_p0b_hubert_unit_memory_2k requires SOURCE_UNIT_VOCAB_SIZE, and manifests must contain source_semantic_units." >&2
      exit 2
    fi
    run_submit "ver2_5_p0b_hubert_unit_memory_2k" "1" "0" "0" "1" "2000" "500" "$A_CE_ROUTE_CHECKPOINT" "-1.0" "0" "0" "0" "0" "$SOURCE_SEMANTIC_POSITION_SCALE" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH" "semantic_units" "$SOURCE_UNIT_VOCAB_SIZE" "256" "first_4" "1"
    ;;
  ver2_5_p0c_codec_bottleneck_2k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    run_submit "ver2_5_p0c_codec_bottleneck_2k" "1" "0" "0" "1" "2000" "500" "$A_CE_ROUTE_CHECKPOINT" "-1.0" "0" "0" "0" "0" "$SOURCE_SEMANTIC_POSITION_SCALE" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH" "codec_bottleneck" "0" "256" "first_4" "0"
    ;;
  ver2_5_selected_p0a_text_token_memory_5k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    # Selected after core48 comparison on 2026-07-01:
    # P0-A text-token memory is the best aggregate auxiliary candidate among
    # P0-A/P0-C. Keep CTC/prosody/semantic/speaker/stop/progress losses off and
    # keep P0-C codec_bottleneck disabled by selecting source_content_memory_type=text_tokens.
    run_submit "ver2_5_selected_p0a_text_token_memory_5k" "1" "0" "0" "1" "5000" "1000" "$A_CE_ROUTE_CHECKPOINT" "-1.0" "0" "0" "0" "0" "$SOURCE_SEMANTIC_POSITION_SCALE" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH" "$SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH" "text_tokens" "0" "256" "first_4" "0" "0.0"
    ;;
  ver2_5_s1_from_a_adapter_warmup_stage1_1k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    # Stage1 freezes every other trainable module and only optimizes
    # SourceSemanticMemory. Keep this no-text-only: text samples currently
    # have SOURCE_SEMANTIC_TEXT_GATE=0 and no source semantic feature, so an
    # all-text microbatch would produce a detached CE loss.
    TRAIN_SPEC_OVERRIDE="$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1"
    run_submit "ver2_5_s1_from_a_adapter_warmup_stage1_1k" "1" "0" "0" "1" "1000" "500" "$A_CE_ROUTE_CHECKPOINT" "-1.0" "1" "1" "1" "1"
    TRAIN_SPEC_OVERRIDE=""
    ;;
  ver2_5_s1_from_a_adapter_warmup_stage1_notext_1k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    # Clean retry of adapter-only warmup with an isolated output directory.
    TRAIN_SPEC_OVERRIDE="$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1"
    run_submit "ver2_5_s1_from_a_adapter_warmup_stage1_notext_1k" "1" "0" "0" "1" "1000" "500" "$A_CE_ROUTE_CHECKPOINT" "-1.0" "1" "1" "1" "1"
    TRAIN_SPEC_OVERRIDE=""
    ;;
  ver2_5_s1_from_a_adapter_warmup_stage1_notext_savefix_1k)
    require_file "$A_CE_ROUTE_CHECKPOINT/adapter_model.safetensors"
    require_file "$A_CE_ROUTE_CHECKPOINT/timbre_memory_adapter.pt"
    # Same as notext_1k, but submitted after the frozen-LoRA checkpoint save fix.
    TRAIN_SPEC_OVERRIDE="$NO_TEXT_TRAIN_JSONL_DEFAULT::repeat=1"
    run_submit "ver2_5_s1_from_a_adapter_warmup_stage1_notext_savefix_1k" "1" "0" "0" "1" "1000" "500" "$A_CE_ROUTE_CHECKPOINT" "-1.0" "1" "1" "1" "1"
    TRAIN_SPEC_OVERRIDE=""
    ;;
  ver2_5_s1_from_a_adapter_warmup_joint_2k)
    WARMUP_CHECKPOINT="${WARMUP_CHECKPOINT:-$DEBUG_OUT_ROOT/ver2_5_s1_from_a_adapter_warmup_stage1_notext_savefix_1k/step-1000}"
    require_file "$WARMUP_CHECKPOINT/adapter_model.safetensors"
    require_file "$WARMUP_CHECKPOINT/timbre_memory_adapter.pt"
    run_submit "ver2_5_s1_from_a_adapter_warmup_joint_2k" "1" "0" "0" "1" "2000" "500" "$WARMUP_CHECKPOINT" "-1.0" "0" "0" "0" "0"
    ;;
  ver2_5_all)
    if [ "$DRY_RUN" = "1" ]; then
      sh "$0" ver2_5_p0a_text_token_memory_2k --dry-run
      sh "$0" ver2_5_p0c_codec_bottleneck_2k --dry-run
    else
      sh "$0" ver2_5_p0a_text_token_memory_2k
      sh "$0" ver2_5_p0c_codec_bottleneck_2k
    fi
    ;;
  *)
    echo "ERROR: unsupported experiment: $EXPERIMENT" >&2
    exit 2
    ;;
esac
