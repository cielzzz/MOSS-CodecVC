#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BASE_OUT="${BASE_OUT:-$ROOT/outputs/lora_runs/ver2_3_debug_resume_verified}"
DEBUG_OUT_ROOT="${DEBUG_OUT_ROOT:-$ROOT/outputs/lora_runs/ver2_3_debug_resume_evalfix}"
SUBMIT="$ROOT/scripts/004016_submit_ver2_3_debug_experiments.sh"
DRY_RUN=0
MAX_SUBMIT="${MAX_SUBMIT:-4}"

usage() {
  cat >&2 <<EOF_USAGE
Usage: bash $0 [--dry-run] exp...

Experiments: tiny_no_text128 ablation_a_ce_route ablation_b_ctc ablation_c_hubert ablation_c_progress_stop ablation_d_prosody
Default exp list: tiny_no_text128 ablation_a_ce_route ablation_b_ctc ablation_c_hubert
MAX_SUBMIT default: 4
EOF_USAGE
}

experiments=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) experiments+=("$1") ;;
  esac
  shift
done

if [ "${#experiments[@]}" -eq 0 ]; then
  experiments=(tiny_no_text128 ablation_a_ce_route ablation_b_ctc ablation_c_hubert)
fi
if [ "${#experiments[@]}" -gt "$MAX_SUBMIT" ]; then
  echo "ERROR: refusing to submit ${#experiments[@]} jobs; MAX_SUBMIT=$MAX_SUBMIT" >&2
  exit 2
fi

resume_for() {
  case "$1" in
    tiny_no_text128) echo "$BASE_OUT/tiny_no_text128_ce_ctc_semantic/step-200" ;;
    ablation_a_ce_route) echo "$BASE_OUT/ablation_a_ce_route/step-500" ;;
    ablation_b_ctc) echo "$BASE_OUT/ablation_b_ctc/step-500" ;;
    ablation_c_hubert) echo "$BASE_OUT/ablation_c_hubert/step-500" ;;
    ablation_c_progress_stop) echo "$BASE_OUT/ablation_c_progress_stop/step-500" ;;
    ablation_d_prosody) echo "$BASE_OUT/ablation_d_prosody/step-500" ;;
    *) echo "ERROR: unsupported experiment: $1" >&2; return 2 ;;
  esac
}

steps_for() {
  case "$1" in
    tiny_no_text128) echo "400" ;;
    *) echo "1000" ;;
  esac
}

submit_one() {
  local exp="$1"
  local resume steps dry_arg
  resume="$(resume_for "$exp")"
  steps="$(steps_for "$exp")"
  if [ ! -f "$resume/adapter_model.safetensors" ] || [ ! -f "$resume/timbre_memory_adapter.pt" ]; then
    echo "ERROR: incomplete resume checkpoint for $exp: $resume" >&2
    exit 1
  fi
  dry_arg=""
  if [ "$DRY_RUN" = "1" ]; then
    dry_arg="--dry-run"
  fi
  echo "[evalfix-submit] exp=$exp resume=$resume steps=$steps out_root=$DEBUG_OUT_ROOT dry_run=$DRY_RUN"
  if [ "$exp" = "tiny_no_text128" ]; then
    DEBUG_OUT_ROOT="$DEBUG_OUT_ROOT" RESUME_OVERRIDE="$resume" TINY_STEPS="$steps" \
      sh "$SUBMIT" "$exp" $dry_arg
  else
    DEBUG_OUT_ROOT="$DEBUG_OUT_ROOT" RESUME_OVERRIDE="$resume" ABLATION_STEPS="$steps" \
      sh "$SUBMIT" "$exp" $dry_arg
  fi
}

for exp in "${experiments[@]}"; do
  submit_one "$exp"
done
