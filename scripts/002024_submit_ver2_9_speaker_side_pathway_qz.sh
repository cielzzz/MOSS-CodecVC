#!/bin/sh
# Ver2.9 speaker-side pathway launcher.
#
# Default is DRY_RUN=1. Set ARM=smoke_v1/v1/v2/v3/v4/v5/v1_prime_cross_attn/v1_dprime_cross_attn_scaled/v1_tprime_cross_attn_lite/v1_v2pilot_r1/v1_tprime_cross_attn_medium/v1_tprime_cross_attn_alpha/v1_tprime_cross_attn_heavy/v1_dprime_cross_attn_strict/v1_seq_cross_attn/v1sec_seq_pure and
# ALLOW_VER2_9_SUBMIT=1 DRY_RUN=0 only after smoke/prepared data are reviewed.

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
ARM="${ARM:-smoke_v1}"
DRY_RUN="${DRY_RUN:-1}"
BASE_SCRIPT="$ROOT/scripts/002012_submit_ver2_8_sideonly_content_memory_h200_qz.sh"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_20260707}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"
BASE_BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"

require_speaker_vec_manifest() {
  path="$1"
  if [ ! -f "$path" ]; then
    echo "ERROR: missing prepared manifest: $path" >&2
    exit 1
  fi
  if ! head -n 1 "$path" | grep -q '"speaker_vec_path"'; then
    echo "ERROR: manifest lacks speaker_vec_path: $path" >&2
    echo "Run scripts/002023_prepare_ver2_9_speaker_vecs.sh first." >&2
    exit 1
  fi
}

require_speaker_seq_manifest() {
  path="$1"
  if [ ! -f "$path" ]; then
    echo "ERROR: missing prepared manifest: $path" >&2
    exit 1
  fi
  if ! head -n 1 "$path" | grep -q '"speaker_seq_path"'; then
    echo "ERROR: manifest lacks speaker_seq_path: $path" >&2
    echo "Run scripts/002031_precompute_wavlm_seq_features.py first." >&2
    exit 1
  fi
}

require_prepared_manifest() {
  path="$1"
  if [ ! -f "$path" ]; then
    echo "ERROR: missing prepared manifest: $path" >&2
    exit 1
  fi
}

setup_common() {
  suffix="$1"
  export PREPARED_DIR
  export TEXT_REPEAT
  export AUTO_PREPARE_VER2_8=0
  export SKIP_VER2_8_DATA_SUMMARY="${SKIP_VER2_8_DATA_SUMMARY:-1}"

  export TIMBRE_SIDE_ONLY=1
  export USE_TIMBRE_MEMORY=0
  export TIMBRE_MEMORY_TOKENS=0
  export TIMBRE_ADAPTER_LAYERS=""
  export REF_SPEAKER_PROMPT_TOKENS=0
  export REF_SPEAKER_PROMPT_MODE=memory
  export REF_SPEAKER_PROMPT_SLOT=0
  export REF_PROMPT_CODEC_PERMUTATION=0
  export POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION=0

  export TARGET_FRONT_CE_WEIGHT="${TARGET_FRONT_CE_WEIGHT:-4.0}"
  export TARGET_FRONT_CE_SECONDS="${TARGET_FRONT_CE_SECONDS:-0.75}"
  export TARGET_SPK_WEIGHT="${TARGET_SPK_WEIGHT:-0.0}"
  export SOURCE_SUPPRESS_WEIGHT="${SOURCE_SUPPRESS_WEIGHT:-0.0}"
  export SPEAKER_INFONCE_WEIGHT="${SPEAKER_INFONCE_WEIGHT:-0.0}"
  export SPEAKER_INFONCE_NEGATIVE_POOL_SIZE=0
  export REF_CONTENT_SUPPRESSION_WEIGHT="${REF_CONTENT_SUPPRESSION_WEIGHT:-0.0}"
  export SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.0}"

  export SPEAKER_ENCODER_TYPE="${SPEAKER_ENCODER_TYPE:-wavlm_sv}"
  export SPEAKER_ENCODER_PATH="${SPEAKER_ENCODER_PATH:-microsoft/wavlm-base-plus-sv}"
  export SPEAKER_EMBEDDING_DIM="${SPEAKER_EMBEDDING_DIM:-512}"
  export ENABLE_SPEAKER_SIDE_PATHWAY=1
  export SPEAKER_SIDE_PATHWAY_LAYERS="${SPEAKER_SIDE_PATHWAY_LAYERS:-all}"
  export SPEAKER_SIDE_PATHWAY_KV_BIAS="${SPEAKER_SIDE_PATHWAY_KV_BIAS:-1}"
  export SPEAKER_SIDE_PATHWAY_GATE_INIT="${SPEAKER_SIDE_PATHWAY_GATE_INIT:-0.0}"
  export SPEAKER_SIDE_PATHWAY_DROPOUT="${SPEAKER_SIDE_PATHWAY_DROPOUT:-0.15}"

  export ENABLE_SOURCE_SEMANTIC_MEMORY=0
  export SOURCE_SEMANTIC_ADAPTER_LAYERS=""
  export SOURCE_SEMANTIC_PROGRESS_WEIGHT=0.0
  export SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT=0.0

  export PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.10}"
  export STOP_LOSS_WEIGHT="${STOP_LOSS_WEIGHT:-0.20}"
  export PROGRESS_NUM_BINS="${PROGRESS_NUM_BINS:-32}"
  export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant_with_warmup}"

  export POST_TRAIN_QUICK_EVAL="${POST_TRAIN_QUICK_EVAL:-0}"
  export POST_TRAIN_RUN_T11=0
  export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-9-speaker-side-${suffix}}"
  export OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_9_speaker_side_${suffix}_steps${MAX_TRAIN_STEPS}}"
  export POST_TRAIN_EVAL_LABEL="${POST_TRAIN_EVAL_LABEL:-$JOB_NAME_PREFIX}"
  export BATCH_ID="${BASE_BATCH_ID}_$(printf '%s' "$ARM" | tr -c 'A-Za-z0-9_' '_')"
}

case "$ARM" in
  smoke_v1|smoke-v1)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-600}"
    SAVE_STEPS="${SAVE_STEPS:-600}"
    EVAL_STEPS="${EVAL_STEPS:-600}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    LOGGING_STEPS="${LOGGING_STEPS:-25}"
    GPU_COUNT="${GPU_COUNT:-8}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
    ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml}"
    LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant_with_warmup}"
    SMOKE_TEST="${SMOKE_TEST:-1}"
    DISABLE_EVAL="${DISABLE_EVAL:-1}"
    setup_common "smoke_v1_fixed"
    ;;
  v1|V1)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    setup_common "v1_wavlm_full_adaln_kv_stop"
    ;;
  v1_prime_cross_attn|v1-prime-cross-attn|v1_prime|V1_PRIME)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.15}"
    setup_common "v1_prime_wavlm_full_adaln_kv_cross8_stop"
    export ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-1}"
    export SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
    export SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-8}"
    export SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:-0.0}"
    export SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
    export SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-1.0}"
    export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-}"
    ;;
  v1_dprime_cross_attn_scaled|v1-dprime-cross-attn-scaled|v1_dprime|V1_DPRIME)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.15}"
    setup_common "v1_dprime_wavlm_full_adaln_kv_cross8_scaled_stop"
    export ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-1}"
    export SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
    export SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-8}"
    export SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:--1.0}"
    export SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
    export SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-0.1}"
    export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-0.001}"
    ;;
  v1_tprime_cross_attn_lite|v1-tprime-cross-attn-lite|v1_tprime|V1_TPRIME|v1_lite|V1_LITE|v1_v2pilot_cross_attn_lite|v1-v2pilot-cross-attn-lite|v1_v2pilot_r1|v1-v2pilot-r1|V1_V2PILOT|V1_V2PILOT_R1)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.15}"
    setup_common "v1_tprime_wavlm_full_adaln_kv_cross8_lite_stop"
    export ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-1}"
    export SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
    export SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-8}"
    export SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:--0.5}"
    export SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
    export SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-0.3}"
    export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-0.001}"
    export SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS="${SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS:-0}"
    ;;
  v1_tprime_cross_attn_medium|v1-tprime-cross-attn-medium|v1_medium|V1_MEDIUM)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.15}"
    setup_common "v1_tprime_wavlm_full_adaln_kv_cross8_medium_stop"
    export ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-1}"
    export SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
    export SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-8}"
    export SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:-0.0}"
    export SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
    export SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-0.5}"
    export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-0.001}"
    export SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS="${SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS:-0}"
    ;;
  v1_tprime_cross_attn_alpha|v1-tprime-cross-attn-alpha|v1_alpha|V1_ALPHA)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.15}"
    setup_common "v1_tprime_wavlm_full_adaln_kv_cross8_alpha_stop"
    export ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-1}"
    export SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
    export SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-8}"
    export SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:--0.5}"
    export SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
    export SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-0.3}"
    export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-0.001}"
    export SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS="${SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS:-1000}"
    ;;
  v1_tprime_cross_attn_heavy|v1-tprime-cross-attn-heavy|v1_heavy|V1_HEAVY)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.15}"
    setup_common "v1_tprime_wavlm_full_adaln_kv_cross16_heavy_stop"
    export ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-1}"
    export SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
    export SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-16}"
    export SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:-0.0}"
    export SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
    export SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-0.7}"
    export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-0.001}"
    export SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS="${SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS:-0}"
    ;;
  v1_dprime_cross_attn_strict|v1-dprime-cross-attn-strict|v1_strict|V1_STRICT)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.15}"
    setup_common "v1_dprime_wavlm_full_adaln_kv_cross8_strict_stop"
    export ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-1}"
    export SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
    export SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-8}"
    export SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:--1.0}"
    export SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
    export SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-0.05}"
    export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-0.001}"
    ;;
  v1_seq_cross_attn|v1-seq-cross-attn|v1_seq|V1_SEQ)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.15}"
    setup_common "v1_seq_wavlm_full_adaln_kv_seqcross_lite_stop"
    export ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-1}"
    export SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
    export SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-0}"
    export SPEAKER_CROSS_ATTN_SOURCE="${SPEAKER_CROSS_ATTN_SOURCE:-sequence}"
    export SPEAKER_CROSS_ATTN_SEQ_DIM="${SPEAKER_CROSS_ATTN_SEQ_DIM:-768}"
    export SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:--0.5}"
    export SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
    export SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-0.3}"
    export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-}"
    export SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS="${SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS:-0}"
    export QUICK_VALIDATION_JSONL="${QUICK_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_9_seq_ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
    export DOMAIN_VALIDATION_JSONL="${DOMAIN_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_9_seq_ver2_8_t11_domain_prepared_valid_no_text_50_20260704.jsonl}"
    export REQUIRE_SPEAKER_SEQ_MANIFEST="${REQUIRE_SPEAKER_SEQ_MANIFEST:-1}"
    ;;
  v1sec_seq_pure|v1sec-seq-pure|v1_second_seq_pure|v1-second-seq-pure|V1SEC_SEQ_PURE)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.15}"
    setup_common "v1sec_seq_pure_olddata_adaln_kv_seqcross_lite_stop"
    export ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-1}"
    export SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
    export SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-0}"
    export SPEAKER_CROSS_ATTN_SOURCE="${SPEAKER_CROSS_ATTN_SOURCE:-sequence}"
    export SPEAKER_CROSS_ATTN_SEQ_DIM="${SPEAKER_CROSS_ATTN_SEQ_DIM:-768}"
    export SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:--0.5}"
    export SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
    export SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-0.3}"
    export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-0.001}"
    export SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS="${SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS:-0}"
    export QUICK_VALIDATION_JSONL="${QUICK_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_9_seq_ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
    export DOMAIN_VALIDATION_JSONL="${DOMAIN_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_9_seq_ver2_8_t11_domain_prepared_valid_no_text_50_20260704.jsonl}"
    export REQUIRE_SPEAKER_SEQ_MANIFEST="${REQUIRE_SPEAKER_SEQ_MANIFEST:-1}"
    ;;
  v1sec_seq_alpha3|v1sec-seq-alpha3|v1_second_seq_alpha3|v1-second-seq-alpha3|V1SEC_SEQ_ALPHA3)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.15}"
    setup_common "v1sec_seq_alpha3_olddata_adaln_kv_seqcross_scale06_aw1000_stop"
    export ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-1}"
    export SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
    export SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-0}"
    export SPEAKER_CROSS_ATTN_SOURCE="${SPEAKER_CROSS_ATTN_SOURCE:-sequence}"
    export SPEAKER_CROSS_ATTN_SEQ_DIM="${SPEAKER_CROSS_ATTN_SEQ_DIM:-768}"
    export SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:--0.5}"
    export SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
    export SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-0.6}"
    export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-0.001}"
    export SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS="${SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS:-1000}"
    export QUICK_VALIDATION_JSONL="${QUICK_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_9_seq_ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
    export DOMAIN_VALIDATION_JSONL="${DOMAIN_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_9_seq_ver2_8_t11_domain_prepared_valid_no_text_50_20260704.jsonl}"
    export REQUIRE_SPEAKER_SEQ_MANIFEST="${REQUIRE_SPEAKER_SEQ_MANIFEST:-1}"
    ;;
  v2|V2)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    setup_common "v2_wavlm_full_adaln_kv_stopoff"
    export PROGRESS_LOSS_WEIGHT=0.0
    export STOP_LOSS_WEIGHT=0.0
    ;;
  v3|V3)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    setup_common "v3_sv_finetuned_ecapa_full_adaln_kv_stop"
    export SPEAKER_ENCODER_TYPE="${V3_SPEAKER_ENCODER_TYPE:-seed_tts_eval_ecapa}"
    export SPEAKER_ENCODER_PATH="${V3_SPEAKER_ENCODER_PATH:-}"
    export SPEAKER_EMBEDDING_DIM="${V3_SPEAKER_EMBEDDING_DIM:-256}"
    ;;
  v4|V4)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    setup_common "v4_wavlm_full_adaln_no_kv_stop"
    export SPEAKER_SIDE_PATHWAY_KV_BIAS=0
    ;;
  v5|V5)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    EVAL_STEPS="${EVAL_STEPS:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    setup_common "v5_wavlm_last8_adaln_kv_stop"
    export SPEAKER_SIDE_PATHWAY_LAYERS=last_8
    ;;
  *)
    echo "ERROR: unsupported ARM=$ARM; expected smoke_v1/v1/v1_prime_cross_attn/v1_dprime_cross_attn_scaled/v1_tprime_cross_attn_lite/v1_v2pilot_cross_attn_lite/v1_v2pilot_r1/v1_tprime_cross_attn_medium/v1_tprime_cross_attn_alpha/v1_tprime_cross_attn_heavy/v1_dprime_cross_attn_strict/v1_seq_cross_attn/v1sec_seq_pure/v1sec_seq_alpha3/v2/v3/v4/v5" >&2
    exit 2
    ;;
esac

export MAX_TRAIN_STEPS SAVE_STEPS EVAL_STEPS NUM_EPOCHS DRY_RUN
export LOGGING_STEPS GPU_COUNT PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS ACCELERATE_CONFIG SMOKE_TEST DISABLE_EVAL LR_SCHEDULER_TYPE
export ENABLE_SPEAKER_CROSS_ATTN SPEAKER_CROSS_ATTN_LAYERS SPEAKER_CROSS_ATTN_TOKENS SPEAKER_CROSS_ATTN_GATE_INIT SPEAKER_CROSS_ATTN_DROPOUT SPEAKER_CROSS_ATTN_OUTPUT_SCALE SPEAKER_CROSS_ATTN_TOKEN_INIT_STD SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS SPEAKER_CROSS_ATTN_SOURCE SPEAKER_CROSS_ATTN_SEQ_DIM

if [ "${REQUIRE_SPEAKER_VEC_MANIFEST:-1}" = "1" ]; then
  require_speaker_vec_manifest "$PREPARED_DIR/no_text.train.jsonl"
  require_speaker_vec_manifest "$PREPARED_DIR/text.train.jsonl"
else
  require_prepared_manifest "$PREPARED_DIR/no_text.train.jsonl"
  require_prepared_manifest "$PREPARED_DIR/text.train.jsonl"
fi
if [ "${REQUIRE_SPEAKER_SEQ_MANIFEST:-0}" = "1" ]; then
  require_speaker_seq_manifest "$PREPARED_DIR/no_text.train.jsonl"
  require_speaker_seq_manifest "$PREPARED_DIR/text.train.jsonl"
fi

if [ "$DRY_RUN" != "1" ] && [ "${ALLOW_VER2_9_SUBMIT:-0}" != "1" ]; then
  echo "ERROR: ver2.9 launch is guarded; set ALLOW_VER2_9_SUBMIT=1 to submit intentionally." >&2
  exit 1
fi

echo "[ver2.9-submit] arm=$ARM dry_run=$DRY_RUN prepared_dir=$PREPARED_DIR out=$OUT_DIR"
echo "[ver2.9-submit] speaker_encoder=$SPEAKER_ENCODER_TYPE dim=$SPEAKER_EMBEDDING_DIM side_layers=$SPEAKER_SIDE_PATHWAY_LAYERS kv=$SPEAKER_SIDE_PATHWAY_KV_BIAS stop=($PROGRESS_LOSS_WEIGHT,$STOP_LOSS_WEIGHT)"
echo "[ver2.9-submit] speaker_cross_attn=${ENABLE_SPEAKER_CROSS_ATTN:-0} source=${SPEAKER_CROSS_ATTN_SOURCE:-vector} seq_dim=${SPEAKER_CROSS_ATTN_SEQ_DIM:-0} layers=${SPEAKER_CROSS_ATTN_LAYERS:-all} tokens=${SPEAKER_CROSS_ATTN_TOKENS:-0} gate_init=${SPEAKER_CROSS_ATTN_GATE_INIT:-0.0} output_scale=${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-1.0} token_init_std=${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-} alpha_warmup_steps=${SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS:-0} condition_dropout=$SPEAKER_CONDITION_DROPOUT"

sh "$BASE_SCRIPT" "$@"
