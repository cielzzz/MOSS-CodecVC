#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DEBUG_OUT_ROOT="${DEBUG_OUT_ROOT:-$ROOT/outputs/lora_runs/ver2_3_debug_resume}"
BASE_OUT="$ROOT/outputs/lora_runs/ver2_3_debug"
SUBMIT="$ROOT/scripts/004016_submit_ver2_3_debug_experiments.sh"
DRY_RUN=0

usage() {
  cat >&2 <<EOF
Usage: bash $0 [--dry-run]

Resume tiny/A/B/C/C_stop/D debug experiments from salvaged checkpoints.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

submit_resume() {
  local exp="$1"
  local resume="$2"
  if [ ! -f "$resume/adapter_model.safetensors" ] || [ ! -f "$resume/timbre_memory_adapter.pt" ]; then
    echo "ERROR: incomplete resume checkpoint for $exp: $resume" >&2
    exit 1
  fi
  echo "[resume-submit] exp=$exp resume=$resume out_root=$DEBUG_OUT_ROOT"
  if [ "$DRY_RUN" = "1" ]; then
    DEBUG_OUT_ROOT="$DEBUG_OUT_ROOT" RESUME_OVERRIDE="$resume" sh "$SUBMIT" "$exp" --dry-run
  else
    DEBUG_OUT_ROOT="$DEBUG_OUT_ROOT" RESUME_OVERRIDE="$resume" sh "$SUBMIT" "$exp"
  fi
}

# Resume for remaining steps only: tiny 200 -> +600, ablations 500 -> +1500.
TINY_STEPS="${TINY_RESUME_STEPS:-600}" \
  submit_resume tiny_no_text128 "$BASE_OUT/tiny_no_text128_ce_ctc_semantic/step-200"

ABLATION_STEPS="${ABLATION_RESUME_STEPS:-1500}" \
  submit_resume ablation_a_ce_route "$BASE_OUT/ablation_a_ce_route/step-500"
ABLATION_STEPS="${ABLATION_RESUME_STEPS:-1500}" \
  submit_resume ablation_b_ctc "$BASE_OUT/ablation_b_ctc/step-500"
ABLATION_STEPS="${ABLATION_RESUME_STEPS:-1500}" \
  submit_resume ablation_c_hubert "$BASE_OUT/ablation_c_hubert/step-500"
ABLATION_STEPS="${ABLATION_RESUME_STEPS:-1500}" \
  submit_resume ablation_c_progress_stop "$BASE_OUT/ablation_c_progress_stop/step-500"
ABLATION_STEPS="${ABLATION_RESUME_STEPS:-1500}" \
  submit_resume ablation_d_prosody "$BASE_OUT/ablation_d_prosody/step-500"
