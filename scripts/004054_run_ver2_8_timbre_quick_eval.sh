#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
RUN_DIR="${RUN_DIR:-}"
RUN_LABEL="${RUN_LABEL:-$(basename "${RUN_DIR:-run}")}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/testset/outputs/ver2_8_timbre_repair_quick_eval}"
DOCS_MD="${DOCS_MD:-$ROOT/docs/ver2_8_timbre_repair_1a_1b_short_train_quick_eval_20260704.md}"

QUICK_VALIDATION_JSONL="${QUICK_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
DOMAIN_VALIDATION_JSONL="${DOMAIN_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_8_t11_domain_prepared_valid_no_text_50_20260704.jsonl}"
CHECKPOINTS="${CHECKPOINTS:-}"
MAX_CHECKPOINT_STEP="${MAX_CHECKPOINT_STEP:-0}"
SEED="${SEED:-1234}"

RUN_QUICK="${RUN_QUICK:-1}"
RUN_T11="${RUN_T11:-1}"
RUN_ASR="${RUN_ASR:-1}"
BUILD_PAGE="${BUILD_PAGE:-1}"
QUICK_TIMBRE_SIDE_ONLY="${TIMBRE_SIDE_ONLY:-1}"
QUICK_GPU_COUNT="${QUICK_GPU_COUNT:-4}"
QUICK_NUM_SHARDS="${QUICK_NUM_SHARDS:-$QUICK_GPU_COUNT}"
QUICK_ASR_NUM_SHARDS="${QUICK_ASR_NUM_SHARDS:-$QUICK_GPU_COUNT}"
REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES="${REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES:-10}"
REF_SPEAKER_PROMPT_ATTENTION_LAYERS="${REF_SPEAKER_PROMPT_ATTENTION_LAYERS:--4,-3,-2,-1}"
REF_PROMPT_CODEC_PERMUTATION="${REF_PROMPT_CODEC_PERMUTATION:-}"
REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS="${REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS:-}"
REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS="${REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS:-}"
REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE="${REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE:-}"
REF_PROMPT_CODEC_PERMUTATION_SEED="${REF_PROMPT_CODEC_PERMUTATION_SEED:-1234}"
REF_PROMPT_CODEC_PERMUTATION_MODE="${REF_PROMPT_CODEC_PERMUTATION_MODE:-}"
REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS="${REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS:-}"
REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP="${REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP:-}"
TIMBRE_CFG_SCALE="${TIMBRE_CFG_SCALE:-1.0}"

if [ -z "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR is required" >&2
  exit 2
fi
if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR does not exist: $RUN_DIR" >&2
  exit 1
fi
if [ ! -f "$QUICK_VALIDATION_JSONL" ]; then
  echo "ERROR: QUICK_VALIDATION_JSONL not found: $QUICK_VALIDATION_JSONL" >&2
  exit 1
fi
if [ "$RUN_T11" = "1" ] && [ ! -f "$DOMAIN_VALIDATION_JSONL" ]; then
  echo "ERROR: DOMAIN_VALIDATION_JSONL not found: $DOMAIN_VALIDATION_JSONL" >&2
  exit 1
fi

mkdir -p "$EVAL_ROOT" "$(dirname "$DOCS_MD")"

discover_checkpoints() {
  if [ -n "$CHECKPOINTS" ]; then
    old_ifs="$IFS"
    IFS=','
    for item in $CHECKPOINTS; do
      IFS="$old_ifs"
      if [ -n "$item" ]; then
        case "$item" in
          /*) printf '%s\n' "$item" ;;
          *) printf '%s\n' "$RUN_DIR/$item" ;;
        esac
      fi
      IFS=','
    done
    IFS="$old_ifs"
    return
  fi
  find "$RUN_DIR" -maxdepth 1 -type d -name 'step-*' | sort -V
}

step_number() {
  basename "$1" | sed 's/^step-//'
}

run_seedtts_eval() {
  local checkpoint="$1"
  local validation_jsonl="$2"
  local run_id="$3"
  local output_dir="$4"
  local page_flag="$5"
  local attention_frames="$6"
  mkdir -p "$output_dir"
  local attn_impl="${MOSS_TTS_ATTN_IMPLEMENTATION:-}"
  if [ "$attention_frames" != "0" ] && [ -z "$attn_impl" ]; then
    attn_impl="eager"
  fi
  SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH=0.0 \
  TEMPERATURE=0.7 \
  NO_TEXT_AUDIO_TEMPERATURE=1.1 \
  NO_TEXT_AUDIO_TOP_P=0.7 \
  NO_TEXT_AUDIO_TOP_K=20 \
  AUDIO_TEMPERATURE=1.1 \
  AUDIO_TOP_P=0.7 \
  AUDIO_TOP_K=20 \
  TIMBRE_SIDE_ONLY="$QUICK_TIMBRE_SIDE_ONLY" \
  MOSS_TTS_ATTN_IMPLEMENTATION="$attn_impl" \
  REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES="$attention_frames" \
  REF_SPEAKER_PROMPT_ATTENTION_LAYERS="$REF_SPEAKER_PROMPT_ATTENTION_LAYERS" \
  REF_PROMPT_CODEC_PERMUTATION="$REF_PROMPT_CODEC_PERMUTATION" \
  REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS="$REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS" \
  REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS="$REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS" \
  REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE="$REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE" \
  REF_PROMPT_CODEC_PERMUTATION_SEED="$REF_PROMPT_CODEC_PERMUTATION_SEED" \
  REF_PROMPT_CODEC_PERMUTATION_MODE="$REF_PROMPT_CODEC_PERMUTATION_MODE" \
  REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS="$REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS" \
  REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP="$REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP" \
  TIMBRE_CFG_SCALE="$TIMBRE_CFG_SCALE" \
  PYTHON="$PYTHON" \
  ASR_PYTHON="$ASR_PYTHON" \
  VALIDATION_JSONL="$validation_jsonl" \
  MODEL_PATH="$checkpoint" \
  RUN_ID="$run_id" \
  RUN_LABEL="$run_id" \
  OUTPUT_DIR="$output_dir" \
  MODE=no_text \
  MAX_CASES=0 \
  DECODING_PROFILE=default \
  PERSISTENT_INFER=1 \
  OVERWRITE_INFER=1 \
  RESET_MANIFESTS=1 \
  RUN_ASR="$RUN_ASR" \
  RUN_SUMMARY=1 \
  BUILD_PAGE="$page_flag" \
  PAGE_DIR="$output_dir/listening_page" \
  GPU_COUNT="$QUICK_GPU_COUNT" \
  NUM_SHARDS="$QUICK_NUM_SHARDS" \
  ASR_NUM_SHARDS="$QUICK_ASR_NUM_SHARDS" \
  SEED="$SEED" \
  bash "$ROOT/scripts/004039_run_seedtts_validation_eval.sh"
}

run_ref_content_similarity() {
  local run_id="$1"
  local output_dir="$2"
  local asr_jsonl="$output_dir/${run_id}.asr_eval.jsonl"
  if [ ! -f "$asr_jsonl" ]; then
    echo "[ref-content-sim] skip; missing ASR jsonl: $asr_jsonl"
    return 0
  fi
  "$PYTHON" "$ROOT/scripts/004056_summarize_seedtts_ref_content_similarity.py" \
    --asr-jsonl "$asr_jsonl" \
    --output-json "$output_dir/${run_id}.ref_content_similarity_summary.json" \
    --output-md "$output_dir/${run_id}.ref_content_similarity_summary.md"
}

run_speaker_sim() {
  local validation_jsonl="$1"
  local run_id="$2"
  local output_dir="$3"
  "$PYTHON" "$ROOT/scripts/004050_summarize_seedtts_speaker_sim_only.py" \
    --validation-jsonl "$validation_jsonl" \
    --run "$run_id=$output_dir" \
    --output-csv "$output_dir/${run_id}.speaker_sim.csv" \
    --summary-json "$output_dir/${run_id}.speaker_sim_summary.json" \
    --summary-md "$output_dir/${run_id}.speaker_sim_summary.md" \
    --speaker-device cuda:0
}

summarize_attention() {
  local output_dir="$1"
  local summary_json="$2"
  "$PYTHON" - "$output_dir" "$summary_json" <<'PY'
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

output_dir = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
rows = []
for manifest in sorted(output_dir.glob("manifest*.jsonl")):
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stats = row.get("ref_speaker_prompt_attention_stats") or {}
        if stats:
            rows.append({
                "case_id": row.get("case_id"),
                "stats": stats,
            })

def finite(v):
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None

def sort_layer_key(item):
    key = item[0]
    try:
        return int(key)
    except Exception:
        return key

case_rows = []
vals = []
max_head_vals = []
uniform_vals = []
over_uniform_vals = []
layer_vals = defaultdict(lambda: defaultdict(list))
for row in rows:
    stats = row.get("stats") or {}
    frames = stats.get("frames") or []
    case_frame_means = []
    case_frame_max_heads = []
    case_frame_uniforms = []
    case_frame_over_uniforms = []
    for frame in frames:
        frame_mean = finite(frame.get("slot_attention_mean"))
        frame_max_head = finite(frame.get("slot_attention_max_head"))
        frame_over_uniform = finite(frame.get("slot_attention_max_head_over_uniform"))
        layer_frame_max_heads = []
        layer_frame_over_uniforms = []
        layer_frame_uniforms = []
        for layer, payload in (frame.get("layers") or {}).items():
            value = finite(payload.get("slot_attention_mean"))
            max_head = finite(payload.get("slot_attention_max_head"))
            key_len = finite(payload.get("key_len"))
            slot_tokens = finite(payload.get("slot_tokens"))
            uniform = finite(payload.get("uniform_baseline"))
            if uniform is None and key_len and slot_tokens is not None:
                uniform = float(slot_tokens) / max(1.0, float(key_len))
            over_uniform = finite(payload.get("slot_attention_max_head_over_uniform"))
            if over_uniform is None and max_head is not None and uniform is not None and uniform > 0:
                over_uniform = max_head / uniform
            if value is not None:
                layer_vals[layer]["slot_attention_mean"].append(value)
            if max_head is not None:
                layer_vals[layer]["slot_attention_max_head"].append(max_head)
                layer_frame_max_heads.append(max_head)
            if uniform is not None:
                layer_vals[layer]["uniform_baseline"].append(uniform)
                layer_frame_uniforms.append(uniform)
            if over_uniform is not None:
                layer_vals[layer]["max_head_over_uniform"].append(over_uniform)
                layer_frame_over_uniforms.append(over_uniform)
        if frame_max_head is None and layer_frame_max_heads:
            frame_max_head = max(layer_frame_max_heads)
        if frame_over_uniform is None and layer_frame_over_uniforms:
            frame_over_uniform = max(layer_frame_over_uniforms)
        frame_uniform = mean(layer_frame_uniforms)
        if frame_mean is not None:
            case_frame_means.append(frame_mean)
            vals.append(frame_mean)
        if frame_max_head is not None:
            case_frame_max_heads.append(frame_max_head)
            max_head_vals.append(frame_max_head)
        if frame_uniform is not None:
            case_frame_uniforms.append(frame_uniform)
            uniform_vals.append(frame_uniform)
        if frame_over_uniform is not None:
            case_frame_over_uniforms.append(frame_over_uniform)
            over_uniform_vals.append(frame_over_uniform)
    case_mean = finite(stats.get("slot_attention_mean"))
    if case_mean is None:
        case_mean = mean(case_frame_means)
    if case_mean is not None and not case_frame_means:
        vals.append(case_mean)
    case_max_head_mean = finite(stats.get("slot_attention_max_head_mean"))
    if case_max_head_mean is None:
        case_max_head_mean = mean(case_frame_max_heads)
    case_max_head_max = finite(stats.get("slot_attention_max_head_max"))
    if case_max_head_max is None and case_frame_max_heads:
        case_max_head_max = max(case_frame_max_heads)
    case_uniform_mean = finite(stats.get("uniform_baseline_mean"))
    if case_uniform_mean is None:
        case_uniform_mean = mean(case_frame_uniforms)
    case_over_uniform_mean = finite(stats.get("max_head_over_uniform_mean"))
    if case_over_uniform_mean is None:
        case_over_uniform_mean = mean(case_frame_over_uniforms)
    case_over_uniform_max = finite(stats.get("max_head_over_uniform_max"))
    if case_over_uniform_max is None and case_frame_over_uniforms:
        case_over_uniform_max = max(case_frame_over_uniforms)
    case_rows.append({
        "case_id": row.get("case_id"),
        "captured_frames": stats.get("captured_frames"),
        "slot_attention_mean": case_mean,
        "slot_attention_max_frame": stats.get("slot_attention_max_frame"),
        "slot_attention_max_head_mean": case_max_head_mean,
        "slot_attention_max_head_max": case_max_head_max,
        "uniform_baseline_mean": case_uniform_mean,
        "max_head_over_uniform_mean": case_over_uniform_mean,
        "max_head_over_uniform_max": case_over_uniform_max,
        "layers": stats.get("layers") or {},
    })

payload = {
    "output_dir": str(output_dir),
    "cases_with_attention": len(rows),
    "slot_attention_mean": mean(vals),
    "slot_attention_max": max(vals) if vals else None,
    "slot_attention_max_head_mean": mean(max_head_vals),
    "slot_attention_max_head_max": max(max_head_vals) if max_head_vals else None,
    "uniform_baseline_mean": mean(uniform_vals),
    "max_head_over_uniform_mean": mean(over_uniform_vals),
    "max_head_over_uniform_max": max(over_uniform_vals) if over_uniform_vals else None,
    "by_layer": {
        layer: {
            "n": len(values.get("slot_attention_mean", [])),
            "slot_attention_mean": mean(values.get("slot_attention_mean", [])),
            "slot_attention_max": max(values.get("slot_attention_mean", [])) if values.get("slot_attention_mean") else None,
            "slot_attention_max_head_mean": mean(values.get("slot_attention_max_head", [])),
            "slot_attention_max_head_max": max(values.get("slot_attention_max_head", [])) if values.get("slot_attention_max_head") else None,
            "uniform_baseline_mean": mean(values.get("uniform_baseline", [])),
            "max_head_over_uniform_mean": mean(values.get("max_head_over_uniform", [])),
            "max_head_over_uniform_max": max(values.get("max_head_over_uniform", [])) if values.get("max_head_over_uniform") else None,
        }
        for layer, values in sorted(layer_vals.items(), key=sort_layer_key)
    },
    "cases": case_rows,
}
summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[slot-attn-summary] wrote {summary_json}")
PY
}

write_rollup() {
  "$PYTHON" - "$EVAL_ROOT" "$RUN_LABEL" "$DOCS_MD" <<'PY'
import json
import math
import sys
from pathlib import Path

eval_root = Path(sys.argv[1])
run_label = sys.argv[2]
docs_md = Path(sys.argv[3])

def finite(v):
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None

rows = []
for sim_json in sorted(eval_root.glob("*/*.speaker_sim_summary.json")):
    payload = json.loads(sim_json.read_text(encoding="utf-8"))
    run_name, summary = next(iter((payload.get("runs") or {}).items()))
    if (
        "timbre-repair-a1a" not in run_name
        and "timbre-repair-a1b" not in run_name
        and "timbre-repair-a1c" not in run_name
        and "timbre-repair-a1d" not in run_name
        and "timbre-repair-a3" not in run_name
        and "fixgate" not in run_name
    ):
        continue
    scope = summary.get("no_text") or summary.get("all") or {}
    out_dir = Path(summary.get("run_dir") or sim_json.parent)
    attn_path = out_dir / f"{run_name}.ref_slot_attention_summary.json"
    attn = json.loads(attn_path.read_text(encoding="utf-8")) if attn_path.exists() else {}
    asr_path = out_dir / f"{run_name}.summary.json"
    asr = json.loads(asr_path.read_text(encoding="utf-8")) if asr_path.exists() else {}
    ref_content_path = out_dir / f"{run_name}.ref_content_similarity_summary.json"
    ref_content = json.loads(ref_content_path.read_text(encoding="utf-8")) if ref_content_path.exists() else {}
    overall = asr.get("overall") or {}
    if "a1c" in run_name:
        arm = "1c"
    elif "a1d" in run_name:
        arm = "1d"
    elif "a3" in run_name:
        arm = "A3"
    elif "a1bprime_a4" in run_name or "a1bprime-a4" in run_name or "a1bprime+a4" in run_name:
        arm = "1bprime+A4"
    elif "a1bprime" in run_name:
        arm = "1bprime"
    elif "a1b" in run_name:
        arm = "1b"
    elif "a1a" in run_name:
        arm = "1a"
    else:
        arm = "fixgate"
    rows.append({
        "run": run_name,
        "arm": arm,
        "step": run_name.split("_")[1] if "_" in run_name else "",
        "n": scope.get("n"),
        "primary_error": finite(overall.get("primary_error")),
        "cer": finite(overall.get("cer")),
        "keep": overall.get("keep"),
        "sim_gen_ref": finite(scope.get("sim_gen_ref_mean")),
        "sim_gen_source": finite(scope.get("sim_gen_source_mean")),
        "slot_attn": finite(attn.get("slot_attention_mean")),
        "slot_attn_max_head": finite(attn.get("slot_attention_max_head_mean")),
        "slot_attn_over_uniform": finite(attn.get("max_head_over_uniform_mean")),
        "ref_content_lcs_f1": finite((ref_content.get("overall") or {}).get("ref_content_lcs_f1_mean")),
        "ref_content_lcs_recall": finite((ref_content.get("overall") or {}).get("ref_content_lcs_recall_mean")),
        "output_dir": str(out_dir),
    })
rows.sort(key=lambda row: (row["arm"], row["run"]))

lines = [
    "# Ver2.8 Timbre Repair 1a/1b Short-Train Quick Eval",
    "",
    "H-timbre prediction: `1a` memory-only should stay weak; `1b` S2 prompt-slot should show an early rise in `sim(gen,ref)` and measurable S2-slot attention.",
    "",
    f"run label: `{run_label}`",
    f"eval root: `{eval_root}`",
    "",
    "| arm | run | n | primary | CER | keep | sim gen-ref | sim gen-source | ref-content F1 | ref-content recall | S2 mean | S2 max-head | S2 max-head/uniform | output |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    def fmt(v):
        return "" if v is None else f"{v:.4f}"
    lines.append(
        f"| {row['arm']} | {row['run']} | {row['n'] or ''} | {fmt(row['primary_error'])} | "
        f"{fmt(row['cer'])} | {row['keep'] if row['keep'] is not None else ''} | "
        f"{fmt(row['sim_gen_ref'])} | {fmt(row['sim_gen_source'])} | "
        f"{fmt(row['ref_content_lcs_f1'])} | {fmt(row['ref_content_lcs_recall'])} | {fmt(row['slot_attn'])} | "
        f"{fmt(row['slot_attn_max_head'])} | {fmt(row['slot_attn_over_uniform'])} | `{row['output_dir']}` |"
    )
lines.append("")
preserved_suffix = ""
if docs_md.exists():
    existing = docs_md.read_text(encoding="utf-8")
    marker = "\n## "
    idx = existing.find(marker)
    if idx >= 0:
        preserved_suffix = existing[idx:].strip("\n")
body = "\n".join(lines).rstrip() + "\n"
if preserved_suffix:
    body = body.rstrip() + "\n\n" + preserved_suffix.rstrip() + "\n"
docs_md.write_text(body, encoding="utf-8")
print(f"[quick-rollup] wrote {docs_md}")
PY
}

latest_checkpoint=""
if [ "$RUN_QUICK" = "1" ]; then
  while IFS= read -r checkpoint; do
    if [ -z "$checkpoint" ]; then
      continue
    fi
    if [ ! -d "$checkpoint" ] || [ ! -f "$checkpoint/adapter_config.json" ]; then
      echo "[quick-eval] skip non-adapter checkpoint: $checkpoint"
      continue
    fi
    step=$(step_number "$checkpoint")
    if [ "$MAX_CHECKPOINT_STEP" != "0" ] && [ "$step" -gt "$MAX_CHECKPOINT_STEP" ]; then
      continue
    fi
    latest_checkpoint="$checkpoint"
    step_label=$(basename "$checkpoint")
    run_id="${RUN_LABEL}_${step_label}_quick20_d2d3_seed${SEED}"
    output_dir="$EVAL_ROOT/$run_id"
    echo "[quick-eval] checkpoint=$checkpoint run_id=$run_id"
    run_seedtts_eval "$checkpoint" "$QUICK_VALIDATION_JSONL" "$run_id" "$output_dir" "$BUILD_PAGE" "$REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES"
    run_ref_content_similarity "$run_id" "$output_dir"
    run_speaker_sim "$QUICK_VALIDATION_JSONL" "$run_id" "$output_dir"
    summarize_attention "$output_dir" "$output_dir/${run_id}.ref_slot_attention_summary.json"
    write_rollup
  done < <(discover_checkpoints)
fi

if [ "$RUN_T11" = "1" ]; then
  if [ -z "$latest_checkpoint" ]; then
    latest_checkpoint=$(discover_checkpoints | tail -n 1)
  fi
  if [ -n "$latest_checkpoint" ] && [ -d "$latest_checkpoint" ]; then
    step_label=$(basename "$latest_checkpoint")
    run_id="${RUN_LABEL}_${step_label}_t11_domain50_d2d3_seed${SEED}"
    output_dir="$EVAL_ROOT/$run_id"
    echo "[quick-eval] T11 checkpoint=$latest_checkpoint run_id=$run_id"
    run_seedtts_eval "$latest_checkpoint" "$DOMAIN_VALIDATION_JSONL" "$run_id" "$output_dir" "0" "0"
    run_ref_content_similarity "$run_id" "$output_dir"
    run_speaker_sim "$DOMAIN_VALIDATION_JSONL" "$run_id" "$output_dir"
    write_rollup
  else
    echo "[quick-eval] no checkpoint available for T11" >&2
  fi
fi

echo "[quick-eval] done run_label=$RUN_LABEL eval_root=$EVAL_ROOT docs=$DOCS_MD"
