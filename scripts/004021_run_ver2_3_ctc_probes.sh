#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DEBUG_OUT_ROOT="${DEBUG_OUT_ROOT:-$ROOT/outputs/lora_runs/ver2_3_debug}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
MAX_ROWS="${MAX_ROWS:-80}"
JSONL="${JSONL:-$ROOT/testset/validation/ver2_3_debug/moss_codecvc_ver2_3_loss_valid_160.no_text.jsonl}"
OVERWRITE="${OVERWRITE:-0}"
STEPS="${STEPS:-500 1000 1500 2000}"

run_probe() {
  local name="$1"
  local run_dir="$2"
  local step="$3"
  local model_path="$run_dir/step-$step"
  local out_jsonl="$run_dir/ctc_greedy_probe.no_text.step${step}.jsonl"
  if [ ! -f "$model_path/adapter_model.safetensors" ] || [ ! -f "$model_path/timbre_memory_adapter.pt" ]; then
    echo "[ctc-probes] skip missing/incomplete $name step-$step"
    return 0
  fi
  if [ -s "$out_jsonl" ] && [ "$OVERWRITE" != "1" ]; then
    echo "[ctc-probes] reuse $out_jsonl"
    return 0
  fi
  echo "[ctc-probes] run $name step-$step -> $out_jsonl"
  "$PY" "$ROOT/scripts/004019_ctc_greedy_decode_probe.py" \
    --model-path "$model_path" \
    --jsonl "$JSONL" \
    --output-jsonl "$out_jsonl" \
    --max-rows "$MAX_ROWS" \
    --device "$DEVICE"
}

run_all() {
  local runs=(
    "B:$DEBUG_OUT_ROOT/ablation_b_ctc"
    "C:$DEBUG_OUT_ROOT/ablation_c_hubert"
    "C_stop:$DEBUG_OUT_ROOT/ablation_c_progress_stop"
    "D:$DEBUG_OUT_ROOT/ablation_d_prosody"
  )
  local item name dir step
  for item in "${runs[@]}"; do
    name="${item%%:*}"
    dir="${item#*:}"
    for step in $STEPS; do
      run_probe "$name" "$dir" "$step"
    done
  done
}

run_tiny() {
  local run_dir="$DEBUG_OUT_ROOT/tiny_no_text128_ce_ctc_semantic"
  local jsonl="$ROOT/testset/validation/ver2_3_debug/moss_codecvc_ver2_3_tiny_overfit_no_text_128.jsonl"
  local step
  for step in 200 400 600 800; do
    JSONL="$jsonl" run_probe "tiny" "$run_dir" "$step"
  done
}

run_tiny
run_all
"$PY" "$ROOT/scripts/004020_summarize_ver2_3_debug_runs.py" \
  --runs-root "$DEBUG_OUT_ROOT" \
  --output-json "$DEBUG_OUT_ROOT/summary_latest.json"
