#!/usr/bin/env bash
# Submit one MTTS-only 8xH200 job for a complete Ver2.3 final-eval family.
#
# The scientific protocol intentionally keeps two inference/ASR shards per
# configuration. Batch-33 used two shards with one RNG stream per shard, so
# changing the shard count would change the stochastic D2+D3 decoding stream.
#
# Dry-runs:
#   FAMILY=batch3436 RUN_TAG=20260711_mtts DRY_RUN=1 \
#     bash scripts/004072_submit_ver23_family_final_seedtts320_qz.sh
#   FAMILY=batch37 RUN_TAG=20260711_mtts DRY_RUN=1 \
#     bash scripts/004072_submit_ver23_family_final_seedtts320_qz.sh

set -euo pipefail

SELF_PATH=$(readlink -f "$0")
PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"  # MTTS-3-2-0715
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-NVIDIA_H200_SXM_141G}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"

FAMILY="${FAMILY:-batch3436}"
TRAIN_STAMP="${TRAIN_STAMP:-20260710_mtts}"
EVAL_STEP="${EVAL_STEP:-3000}"
SEED="${SEED:-1234}"
RUN_TAG="${RUN_TAG:-20260711_mtts}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENTRYPOINT="${VER23_FAMILY_FINAL320_ENTRYPOINT:-0}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
VALIDATION_JSONL="${VALIDATION_JSONL:-$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl}"
SPEECHBRAIN_ECAPA_MODEL_SOURCE="${SPEECHBRAIN_ECAPA_MODEL_SOURCE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"

case "$FAMILY" in
  batch3436)
    DEFAULT_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
    ;;
  batch37)
    # This snapshot additionally persists and asserts true ref-audio CFG
    # runtime statistics. CODE_ROOT remains overridable for a newer audited
    # cfg-stats snapshot.
    DEFAULT_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_cfg_eval_20260711_1092820"
    ;;
  *)
    echo "ERROR: FAMILY must be batch3436 or batch37; got $FAMILY" >&2
    exit 2
    ;;
esac
CODE_ROOT="${CODE_ROOT:-$DEFAULT_CODE_ROOT}"

RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_family_final_seedtts320_${FAMILY}_step${EVAL_STEP}_${RUN_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_family_final_seedtts320_${FAMILY}_step${EVAL_STEP}_${RUN_TAG}}"
RUNS_ROOT="$EVAL_ROOT/runs"
AGG_ROOT="$EVAL_ROOT/aggregate"
JOB_NAME="${JOB_NAME:-ver23_${FAMILY}_final320_step${EVAL_STEP}_${RUN_TAG}}"
FROZEN_DRIVER="$RECORD_ROOT/004072_family_final_seedtts320.frozen.sh"
RUNNER="$RECORD_ROOT/run_family_final_seedtts320_entrypoint.sh"
RESOLVED_TSV="$RECORD_ROOT/resolved_configs.tsv"

if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: final SeedTTS-320 is restricted to MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP); got $COMPUTE_GROUP" >&2
  exit 2
fi
case "$DRY_RUN:$FORCE:$ENTRYPOINT" in
  [01]:[01]:[01]) ;;
  *)
    echo "ERROR: DRY_RUN, FORCE, and VER23_FAMILY_FINAL320_ENTRYPOINT must be 0 or 1" >&2
    exit 2
    ;;
esac
if [ "$EVAL_STEP" != "3000" ]; then
  echo "ERROR: this final wrapper only accepts EVAL_STEP=3000; got $EVAL_STEP" >&2
  exit 2
fi

family_keys() {
  case "$FAMILY" in
    batch3436) printf '%s\n' B1 B2 A1 B3 A2 ;;
    batch37) printf '%s\n' C1 C2L10 C2L12 C2L14 C2L16 ;;
  esac
}

# Print: run_dir_name<TAB>run_label<TAB>ref_audio_cfg_scale
config_info() {
  case "$FAMILY:$1" in
    batch3436:B1)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch3436_B1_ver23_bnf_last16_3k_$TRAIN_STAMP" \
        "ver23_batch3436_B1_bnf_last16" "1.0"
      ;;
    batch3436:B2)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch3436_B2_ver23_text_r1_3k_$TRAIN_STAMP" \
        "ver23_batch3436_B2_text_r1" "1.0"
      ;;
    batch3436:A1)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch3436_A1_ver23_stronger_decouple_3k_$TRAIN_STAMP" \
        "ver23_batch3436_A1_stronger" "1.0"
      ;;
    batch3436:B3)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch3436_B3_ver23_weaker_decouple_3k_$TRAIN_STAMP" \
        "ver23_batch3436_B3_weaker" "1.0"
      ;;
    batch3436:A2)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch3436_A2_ver23_ctc_3k_$TRAIN_STAMP" \
        "ver23_batch3436_A2_ctc" "1.0"
      ;;
    batch37:C1)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch37_C1_ver23_compact_content_lr_warmup_3k_$TRAIN_STAMP" \
        "ver23_batch37_C1_compact" "1.0"
      ;;
    batch37:C2L10)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch37_C2_ver23_true_ref_audio_cfg_3k_$TRAIN_STAMP" \
        "ver23_batch37_C2_refcfg" "1.0"
      ;;
    batch37:C2L12)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch37_C2_ver23_true_ref_audio_cfg_3k_$TRAIN_STAMP" \
        "ver23_batch37_C2_refcfg" "1.2"
      ;;
    batch37:C2L14)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch37_C2_ver23_true_ref_audio_cfg_3k_$TRAIN_STAMP" \
        "ver23_batch37_C2_refcfg" "1.4"
      ;;
    batch37:C2L16)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch37_C2_ver23_true_ref_audio_cfg_3k_$TRAIN_STAMP" \
        "ver23_batch37_C2_refcfg" "1.6"
      ;;
    *)
      echo "ERROR: unsupported config key: $FAMILY/$1" >&2
      return 2
      ;;
  esac
}

config_gpu_pair() {
  case "$FAMILY:$1" in
    batch3436:B1|batch3436:A2|batch37:C1|batch37:C2L10) printf '%s\n' "0,1" ;;
    batch3436:B2|batch37:C2L12) printf '%s\n' "2,3" ;;
    batch3436:A1|batch37:C2L14) printf '%s\n' "4,5" ;;
    batch3436:B3|batch37:C2L16) printf '%s\n' "6,7" ;;
    *) return 2 ;;
  esac
}

resolved_values() {
  local key="$1"
  local run_dir_name label scale cfg_tag model_path run_id output_dir gpu_pair
  IFS=$'\t' read -r run_dir_name label scale <<<"$(config_info "$key")"
  cfg_tag=$(printf '%s' "$scale" | tr '.' 'p')
  model_path="$PROJECT_ROOT/outputs/lora_runs/$run_dir_name/step-$EVAL_STEP"
  run_id="${label}_step-${EVAL_STEP}_cfg${cfg_tag}_seedtts320_d2d3_seed${SEED}"
  output_dir="$RUNS_ROOT/$run_id"
  gpu_pair=$(config_gpu_pair "$key")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$key" "$label" "$scale" "$gpu_pair" "$model_path" "$run_id" "$output_dir" "$CODE_ROOT"
}

write_resolved_configs() {
  mkdir -p "$RECORD_ROOT" "$RUNS_ROOT" "$AGG_ROOT"
  {
    printf 'config_key\tlabel\tref_audio_cfg_scale\tgpu_pair\tmodel_path\trun_id\toutput_dir\tcode_root\n'
    local key
    while IFS= read -r key; do
      resolved_values "$key"
    done < <(family_keys)
  } > "$RESOLVED_TSV"
}

validate_checkpoint() {
  local key="$1"
  local _key _label _scale _gpu model_path _run_id _output _code
  IFS=$'\t' read -r _key _label _scale _gpu model_path _run_id _output _code <<<"$(resolved_values "$key")"
  local required=(adapter_model.safetensors adapter_config.json README.md timbre_memory_adapter.pt timbre_memory_config.json)
  local name
  if [ ! -d "$model_path" ]; then
    echo "ERROR: missing checkpoint for $key: $model_path" >&2
    return 1
  fi
  for name in "${required[@]}"; do
    if [ ! -s "$model_path/$name" ]; then
      echo "ERROR: missing or empty checkpoint file for $key: $model_path/$name" >&2
      return 1
    fi
  done
  "$PYTHON" - "$FAMILY" "$key" "$model_path/adapter_config.json" "$model_path/timbre_memory_config.json" <<'PY'
import json
import math
import sys

family, key, adapter_path, config_path = sys.argv[1:]
with open(adapter_path, encoding="utf-8") as handle:
    json.load(handle)
with open(config_path, encoding="utf-8") as handle:
    cfg = json.load(handle)

def close(name, expected, tol=1e-8):
    got = cfg.get(name)
    if got is None or not math.isclose(float(got), float(expected), rel_tol=0.0, abs_tol=tol):
        raise SystemExit(f"{family}/{key}: {name}={got!r}, expected {expected!r}")

def equal(name, expected):
    got = cfg.get(name)
    if str(got) != str(expected):
        raise SystemExit(f"{family}/{key}: {name}={got!r}, expected {expected!r}")

if cfg.get("content_cross_attn_enabled") is not True:
    raise SystemExit(f"{family}/{key}: content_cross_attn_enabled is not true")

if family == "batch3436":
    expected = {
        "B1": ("last_16", 0.02, 0.05, 0.0),
        "B2": ("all", 0.02, 0.05, 0.0),
        "A1": ("all", 0.05, 0.10, 0.0),
        "B3": ("all", 0.01, 0.02, 0.0),
        "A2": ("all", 0.02, 0.05, 0.10),
    }[key]
    equal("content_cross_attn_layers", expected[0])
    close("phoneme_classifier_loss_weight", expected[1])
    close("guided_attn_loss_weight", expected[2])
    close("content_ctc_weight", expected[3])
elif key == "C1":
    equal("content_cross_attn_layers", "7,15,23,31,35")
    equal("content_encoder_hidden_size", 512)
    close("ref_audio_cfg_dropout", 0.0)
else:
    equal("content_cross_attn_layers", "all")
    equal("content_encoder_hidden_size", 0)
    close("ref_audio_cfg_dropout", 0.15)
PY
}

audit_code_root() {
  local required=(
    scripts/004039_run_seedtts_validation_eval.sh
    scripts/004042_summarize_seedtts_validation_eval.py
    scripts/004044_run_seedtts_validation_infer_persistent.py
    scripts/004048_summarize_seedtts_ablation_metrics.py
    scripts/004063_analyze_seedtts320_diagnostics.py
  )
  local path
  if [ ! -d "$CODE_ROOT" ]; then
    echo "ERROR: CODE_ROOT does not exist: $CODE_ROOT" >&2
    return 1
  fi
  for path in "${required[@]}"; do
    if [ ! -s "$CODE_ROOT/$path" ]; then
      echo "ERROR: missing CODE_ROOT file: $CODE_ROOT/$path" >&2
      return 1
    fi
  done
  if ! grep -q 'HF_MODULES_CACHE' "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"; then
    echo "ERROR: CODE_ROOT lacks per-shard HF dynamic-module cache isolation: $CODE_ROOT" >&2
    return 1
  fi
  if [ "$FAMILY" = "batch37" ]; then
    if ! grep -q 'REF_AUDIO_CFG_SCALE' "$CODE_ROOT/scripts/004044_run_seedtts_validation_infer_persistent.py"; then
      echo "ERROR: Batch-37 CODE_ROOT lacks REF_AUDIO_CFG_SCALE support: $CODE_ROOT" >&2
      return 1
    fi
    if ! grep -q 'ref_audio_cfg_infer_stats' "$CODE_ROOT/scripts/004044_run_seedtts_validation_infer_persistent.py"; then
      echo "ERROR: Batch-37 CODE_ROOT lacks persisted true-CFG runtime stats: $CODE_ROOT" >&2
      return 1
    fi
  fi
}

validate_all() {
  audit_code_root
  if [ ! -x "$PYTHON" ] || [ ! -x "$ASR_PYTHON" ]; then
    echo "ERROR: missing Python interpreter" >&2
    return 1
  fi
  if [ ! -s "$VALIDATION_JSONL" ]; then
    echo "ERROR: missing validation JSONL: $VALIDATION_JSONL" >&2
    return 1
  fi
  if [ ! -d "$SPEECHBRAIN_ECAPA_MODEL_SOURCE" ]; then
    echo "ERROR: missing local SpeechBrain ECAPA model: $SPEECHBRAIN_ECAPA_MODEL_SOURCE" >&2
    return 1
  fi
  local key
  while IFS= read -r key; do
    validate_checkpoint "$key"
  done < <(family_keys)
}

run_config() {
  local key="$1"
  local _key label scale gpu_pair model_path run_id output_dir code_root
  IFS=$'\t' read -r _key label scale gpu_pair model_path run_id output_dir code_root <<<"$(resolved_values "$key")"
  local log="$RECORD_ROOT/eval_${key}.log"
  mkdir -p "$output_dir"
  (
    set -euo pipefail
    echo "[family-final320] config=$key gpu_pair=$gpu_pair model=$model_path"
    echo "[family-final320] run_id=$run_id scale=$scale output=$output_dir code_root=$code_root"
    CUDA_VISIBLE_DEVICES="$gpu_pair" \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
    HF_MODULES_CACHE_ROOT="$output_dir/.hf_modules_cache" \
    SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH=0.0 \
    TEMPERATURE=0.7 \
    NO_TEXT_AUDIO_TEMPERATURE=1.1 \
    NO_TEXT_AUDIO_TOP_P=0.7 \
    NO_TEXT_AUDIO_TOP_K=20 \
    AUDIO_TEMPERATURE=1.1 \
    AUDIO_TOP_P=0.7 \
    AUDIO_TOP_K=20 \
    SPEAKER_ENCODER_TYPE=embedding_loader \
    TIMBRE_SIDE_ONLY=0 \
    TIMBRE_CFG_SCALE=1.0 \
    REF_AUDIO_CFG_SCALE="$scale" \
    REF_PROMPT_CODEC_PERMUTATION=0 \
    REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES=0 \
    MOSS_TTS_ATTN_IMPLEMENTATION= \
    FILTER_V2_REAL_NO_TEXT_REF_CONTENT_LEAK=1 \
    PYTHON="$PYTHON" \
    ASR_PYTHON="$ASR_PYTHON" \
    VALIDATION_JSONL="$VALIDATION_JSONL" \
    MODEL_PATH="$model_path" \
    RUN_ID="$run_id" \
    RUN_LABEL="$label step-$EVAL_STEP cfg=$scale" \
    OUTPUT_DIR="$output_dir" \
    MODE=all \
    MAX_CASES=0 \
    PER_MODE=0 \
    PER_CELL=0 \
    DECODING_PROFILE=default \
    PERSISTENT_INFER=1 \
    INFER_SHARD_START_DELAY_SEC=0 \
    OVERWRITE_INFER=1 \
    RESET_MANIFESTS=1 \
    RUN_ASR=1 \
    RUN_SUMMARY=1 \
    BUILD_PAGE=0 \
    CONTENT_REFERENCE_MODE=text \
    GPU_COUNT=2 \
    NUM_SHARDS=2 \
    ASR_NUM_SHARDS=2 \
    SEED="$SEED" \
    bash "$code_root/scripts/004039_run_seedtts_validation_eval.sh"
    echo "[family-final320] config=$key complete"
  ) > >(tee -a "$log") 2>&1
}

run_family_configs() {
  local pids=()
  local failed=0
  if [ "$FAMILY" = "batch3436" ]; then
    (set -euo pipefail; run_config B1; run_config A2) & pids+=("$!")
    (set -euo pipefail; run_config B2) & pids+=("$!")
    (set -euo pipefail; run_config A1) & pids+=("$!")
    (set -euo pipefail; run_config B3) & pids+=("$!")
  else
    # C1 and lambda=1.0 are both one-forward paths; chaining them balances
    # the three two-forward CFG lanes at lambda 1.2/1.4/1.6.
    (set -euo pipefail; run_config C1; run_config C2L10) & pids+=("$!")
    (set -euo pipefail; run_config C2L12) & pids+=("$!")
    (set -euo pipefail; run_config C2L14) & pids+=("$!")
    (set -euo pipefail; run_config C2L16) & pids+=("$!")
  fi
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "ERROR: one or more final320 configurations failed" >&2
    return 1
  fi
}

audit_completed_runs() {
  "$PYTHON" - "$FAMILY" "$RESOLVED_TSV" "$AGG_ROOT/completeness.json" <<'PY'
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

family = sys.argv[1]
resolved_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
with resolved_path.open(encoding="utf-8", newline="") as handle:
    configs = list(csv.DictReader(handle, delimiter="\t"))
if len(configs) != 5:
    raise SystemExit(f"expected 5 resolved configs, got {len(configs)}")

payload = {"family": family, "runs": {}}
for cfg in configs:
    run_id = cfg["run_id"]
    run_dir = Path(cfg["output_dir"])
    manifest_paths = sorted(run_dir.glob("manifest.shard*.jsonl"))
    if len(manifest_paths) != 2:
        raise SystemExit(f"{run_id}: expected 2 manifests, got {len(manifest_paths)}")
    manifests = []
    for path in manifest_paths:
        manifests.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    case_ids = [str(row.get("case_id") or "") for row in manifests]
    modes = Counter(str(row.get("mode") or "") for row in manifests)
    statuses = Counter(str(row.get("status") or "") for row in manifests)
    if len(manifests) != 320 or len(set(case_ids)) != 320:
        raise SystemExit(f"{run_id}: manifest rows/unique != 320: {len(manifests)}/{len(set(case_ids))}")
    if modes != Counter({"no_text": 160, "text": 160}):
        raise SystemExit(f"{run_id}: unexpected manifest mode counts: {dict(modes)}")
    if statuses != Counter({"ok": 320}):
        raise SystemExit(f"{run_id}: expected 320 fresh ok rows, got {dict(statuses)}")
    wavs = list(run_dir.glob("*.wav"))
    if len(wavs) != 320:
        raise SystemExit(f"{run_id}: expected 320 WAVs, got {len(wavs)}")

    asr_path = run_dir / f"{run_id}.asr_eval.jsonl"
    asr_rows = [json.loads(line) for line in asr_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    asr_ids = [str(row.get("case_id") or row.get("sample_id") or "") for row in asr_rows]
    asr_modes = Counter(str(row.get("mode") or "") for row in asr_rows)
    if len(asr_rows) != 320 or len(set(asr_ids)) != 320:
        raise SystemExit(f"{run_id}: ASR rows/unique != 320: {len(asr_rows)}/{len(set(asr_ids))}")
    if asr_modes != Counter({"no_text": 160, "text": 160}):
        raise SystemExit(f"{run_id}: unexpected ASR mode counts: {dict(asr_modes)}")

    summary_path = run_dir / f"{run_id}.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["overall"]["n"]) != 320:
        raise SystemExit(f"{run_id}: summary overall n != 320")
    for mode in ("no_text", "text"):
        if int(summary["by_mode"][mode]["n"]) != 160:
            raise SystemExit(f"{run_id}: summary {mode} n != 160")

    infer_logs = sorted((run_dir / "logs").glob("infer.shard*.log"))
    if len(infer_logs) != 2:
        raise SystemExit(f"{run_id}: expected 2 inference logs, got {len(infer_logs)}")
    bnf_lines = sum(path.read_text(encoding="utf-8", errors="replace").count("source semantic memory type=") for path in infer_logs)
    if bnf_lines != 160:
        raise SystemExit(f"{run_id}: expected BNF extraction on exactly 160 no_text rows, got {bnf_lines}")

    scale = float(cfg["ref_audio_cfg_scale"])
    if family == "batch37":
        for row in manifests:
            got_scale = float(row.get("ref_audio_cfg_scale", float("nan")))
            if not math.isclose(got_scale, scale, rel_tol=0.0, abs_tol=1e-8):
                raise SystemExit(f"{run_id}: manifest scale {got_scale} != {scale}")
            stats = row.get("ref_audio_cfg_infer_stats") or {}
            active = float(stats.get("ref_audio_cfg_active", -1.0))
            steps = float(stats.get("ref_audio_cfg_uncond_forward_steps", -1.0))
            positions = float(stats.get("ref_audio_cfg_prompt_positions", -1.0))
            if math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-8):
                if active != 0.0 or steps != 0.0:
                    raise SystemExit(f"{run_id}: lambda=1 unexpectedly used uncond branch: {stats}")
            elif active != 1.0 or steps <= 0.0 or positions <= 0.0:
                raise SystemExit(f"{run_id}: lambda>1 true-CFG runtime gate failed: {stats}")

    payload["runs"][run_id] = {
        "config_key": cfg["config_key"],
        "ref_audio_cfg_scale": scale,
        "manifest_rows": len(manifests),
        "wav_count": len(wavs),
        "asr_rows": len(asr_rows),
        "mode_counts": dict(modes),
        "status_counts": dict(statuses),
        "bnf_extraction_lines": bnf_lines,
    }

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[family-final320-audit] PASS family={family} runs={len(configs)} output={output_path}")
PY
}

run_family_aggregates() {
  local run_args=()
  local _key _label _scale _gpu _model run_id output_dir _code
  while IFS=$'\t' read -r _key _label _scale _gpu _model run_id output_dir _code; do
    [ "$_key" = "config_key" ] && continue
    run_args+=(--run "$run_id=$output_dir")
  done < "$RESOLVED_TSV"

  local dual_cases="$AGG_ROOT/${FAMILY}_step${EVAL_STEP}.dual_encoder_cases.csv"
  CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
    "$PYTHON" "$CODE_ROOT/scripts/004048_summarize_seedtts_ablation_metrics.py" \
      --validation-jsonl "$VALIDATION_JSONL" \
      "${run_args[@]}" \
      --output-csv "$dual_cases" \
      --summary-json "$AGG_ROOT/${FAMILY}_step${EVAL_STEP}.dual_encoder_summary.json" \
      --summary-md "$AGG_ROOT/${FAMILY}_step${EVAL_STEP}.dual_encoder_summary.md" \
      --speaker-device cuda:0 \
      --extra-speaker-encoder speechbrain_ecapa \
      --extra-speaker-device cuda:1 \
      --speechbrain-ecapa-model-source "$SPEECHBRAIN_ECAPA_MODEL_SOURCE"

  "$PYTHON" "$CODE_ROOT/scripts/004063_analyze_seedtts320_diagnostics.py" \
    --validation-jsonl "$VALIDATION_JSONL" \
    --sim-cases-csv "$dual_cases" \
    "${run_args[@]}" \
    --output-dir "$AGG_ROOT/diagnostics" \
    --prefix "${FAMILY}_step${EVAL_STEP}"

  "$PYTHON" - "$FAMILY" "$RESOLVED_TSV" "$dual_cases" "$AGG_ROOT/${FAMILY}_step${EVAL_STEP}.official_matrix" <<'PY'
import csv
import json
import math
import re
import sys
from pathlib import Path

family = sys.argv[1]
resolved_path = Path(sys.argv[2])
dual_path = Path(sys.argv[3])
out_prefix = Path(sys.argv[4])

with resolved_path.open(encoding="utf-8", newline="") as handle:
    configs = list(csv.DictReader(handle, delimiter="\t"))
with dual_path.open(encoding="utf-8", newline="") as handle:
    dual_rows = list(csv.DictReader(handle))
if len(configs) != 5:
    raise SystemExit(f"expected 5 configs, got {len(configs)}")
if len(dual_rows) != 5 * 320:
    raise SystemExit(f"expected {5 * 320} dual-encoder rows, got {len(dual_rows)}")

def finite(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None

def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None

def bool_value(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "keep"}

def normalize_text(text):
    raw = str(text or "").lower()
    chars = []
    for ch in raw:
        code = ord(ch)
        if ch.isalnum() or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
            chars.append(ch)
    return "".join(chars)

def lcs_len(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for ch in long:
        cur = [0]
        for idx, other in enumerate(short, start=1):
            cur.append(prev[idx - 1] + 1 if ch == other else max(prev[idx], cur[-1]))
        prev = cur
    return prev[-1]

def ref_content_f1(row):
    generated = normalize_text(row.get("asr_tgt_text"))
    reference = normalize_text(row.get("timbre_ref_text"))
    hit = lcs_len(generated, reference)
    precision = hit / max(1, len(generated))
    recall = hit / max(1, len(reference))
    return 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)

by_run = {}
for row in dual_rows:
    by_run.setdefault(str(row.get("run") or ""), []).append(row)

matrix = []
for cfg in configs:
    run_id = cfg["run_id"]
    run_dir = Path(cfg["output_dir"])
    rows = by_run.get(run_id, [])
    if len(rows) != 320 or len({row.get("case_id") for row in rows}) != 320:
        raise SystemExit(f"{run_id}: dual rows/unique != 320")
    for row in rows:
        for field in ("sim_gen_ref", "sim_gen_source", "ecapa_sim_gen_ref", "ecapa_sim_gen_source"):
            if finite(row.get(field)) is None:
                raise SystemExit(f"{run_id}/{row.get('case_id')}: missing {field}")

    summary = json.loads((run_dir / f"{run_id}.summary.json").read_text(encoding="utf-8"))
    asr_rows = [
        json.loads(line)
        for line in (run_dir / f"{run_id}.asr_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(asr_rows) != 320:
        raise SystemExit(f"{run_id}: ASR rows != 320")

    for scope in ("all", "no_text", "text"):
        scope_rows = rows if scope == "all" else [row for row in rows if row.get("mode") == scope]
        scope_asr = asr_rows if scope == "all" else [row for row in asr_rows if row.get("mode") == scope]
        official = summary["overall"] if scope == "all" else summary["by_mode"][scope]
        expected_n = 320 if scope == "all" else 160
        if len(scope_rows) != expected_n or len(scope_asr) != expected_n or int(official["n"]) != expected_n:
            raise SystemExit(f"{run_id}/{scope}: inconsistent row counts")
        keep = int(official["keep"])
        dual_keep = sum(bool_value(row.get("content_keep")) for row in scope_rows)
        if dual_keep != keep:
            raise SystemExit(f"{run_id}/{scope}: 004042 keep={keep} != dual cases keep={dual_keep}")
        wavlm_ref = [finite(row.get("sim_gen_ref")) for row in scope_rows]
        wavlm_src = [finite(row.get("sim_gen_source")) for row in scope_rows]
        ecapa_ref = [finite(row.get("ecapa_sim_gen_ref")) for row in scope_rows]
        ecapa_src = [finite(row.get("ecapa_sim_gen_source")) for row in scope_rows]
        wavlm_bound = sum((ref - src) > 0.05 for ref, src in zip(wavlm_ref, wavlm_src)) / expected_n
        ecapa_bound = sum((ref - src) > 0.05 for ref, src in zip(ecapa_ref, ecapa_src)) / expected_n
        speechbrain_ref = mean(ecapa_ref)
        seedtts_src = mean(wavlm_src)
        matrix.append({
            "family": family,
            "config_key": cfg["config_key"],
            "run_id": run_id,
            "ref_audio_cfg_scale": float(cfg["ref_audio_cfg_scale"]),
            "scope": scope,
            "n": expected_n,
            "keep": keep,
            "fail_count": expected_n - keep,
            "fail_rate": (expected_n - keep) / expected_n,
            "cer": finite(official.get("cer")),
            "primary_error": finite(official.get("primary_error")),
            "seedtts_wavlm_ecapa_sim_ref": mean(wavlm_ref),
            "seedtts_wavlm_ecapa_sim_src": seedtts_src,
            "seedtts_wavlm_ecapa_ref_bound": wavlm_bound,
            "speechbrain_ecapa_sim_ref": speechbrain_ref,
            "speechbrain_ecapa_sim_src": mean(ecapa_src),
            "speechbrain_ecapa_ref_bound": ecapa_bound,
            "ref_content_lcs_f1": mean([ref_content_f1(row) for row in scope_asr]),
            "speechbrain_ref_ge_0p48": speechbrain_ref is not None and speechbrain_ref >= 0.48,
            "seedtts_src_le_0p40": seedtts_src is not None and seedtts_src <= 0.40,
            "objective_joint_pass": (
                speechbrain_ref is not None and speechbrain_ref >= 0.48
                and seedtts_src is not None and seedtts_src <= 0.40
            ),
        })

out_prefix.parent.mkdir(parents=True, exist_ok=True)
json_path = Path(str(out_prefix) + ".json")
tsv_path = Path(str(out_prefix) + ".tsv")
md_path = Path(str(out_prefix) + ".md")
json_path.write_text(json.dumps({"family": family, "rows": matrix}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
fields = list(matrix[0].keys())
with tsv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(matrix)

def fmt(value, percent=False):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return f"{100.0 * value:.2f}%" if percent else f"{value:.4f}"

lines = [
    f"# {family} SeedTTS-320 Official Matrix",
    "",
    "`fail` uses the official `content_keep=False` definition from 004042, not the 004048 CER>0.30 diagnostic rate.",
    "SeedTTS similarity is WavLM-Large + ECAPA-TDNN; the second encoder is SpeechBrain ECAPA.",
    "Reference-bound means sim(gen,ref) - sim(gen,src) > 0.05.",
    "",
    "| config | cfg | scope | n | fail | CER | SeedTTS ref | SeedTTS src | SeedTTS ref-bound | SpeechBrain ref | SpeechBrain src | SpeechBrain ref-bound | ref-content F1 | objective |",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in matrix:
    lines.append(
        "| {config} | {cfg:.1f} | {scope} | {n} | {fail} | {cer} | {wref} | {wsrc} | {wbound} | {eref} | {esrc} | {ebound} | {f1} | {obj} |".format(
            config=row["config_key"], cfg=row["ref_audio_cfg_scale"], scope=row["scope"], n=row["n"],
            fail=fmt(row["fail_rate"], True), cer=fmt(row["cer"]),
            wref=fmt(row["seedtts_wavlm_ecapa_sim_ref"]), wsrc=fmt(row["seedtts_wavlm_ecapa_sim_src"]),
            wbound=fmt(row["seedtts_wavlm_ecapa_ref_bound"], True),
            eref=fmt(row["speechbrain_ecapa_sim_ref"]), esrc=fmt(row["speechbrain_ecapa_sim_src"]),
            ebound=fmt(row["speechbrain_ecapa_ref_bound"], True), f1=fmt(row["ref_content_lcs_f1"]),
            obj=fmt(row["objective_joint_pass"]),
        )
    )
lines.extend(["", f"TSV: `{tsv_path}`", f"JSON: `{json_path}`", ""])
md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"[family-final320-matrix] wrote {md_path}")
PY
}

run_entrypoint() {
  mkdir -p "$RECORD_ROOT" "$RUNS_ROOT" "$AGG_ROOT"
  exec > >(tee -a "$RECORD_ROOT/run.log") 2>&1
  echo "[family-final320] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "[family-final320] family=$FAMILY step=$EVAL_STEP seed=$SEED"
  echo "[family-final320] project_root=$PROJECT_ROOT"
  echo "[family-final320] code_root=$CODE_ROOT"
  echo "[family-final320] record_root=$RECORD_ROOT"
  echo "[family-final320] eval_root=$EVAL_ROOT"
  nvidia-smi || true
  validate_all
  write_resolved_configs
  run_family_configs
  audit_completed_runs
  run_family_aggregates
  echo "[family-final320] complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

write_frozen_runner() {
  cp "$SELF_PATH" "$FROZEN_DRIVER"
  chmod +x "$FROZEN_DRIVER"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'export VER23_FAMILY_FINAL320_ENTRYPOINT=%q\n' 1
    printf 'export FAMILY=%q\n' "$FAMILY"
    printf 'export TRAIN_STAMP=%q\n' "$TRAIN_STAMP"
    printf 'export EVAL_STEP=%q\n' "$EVAL_STEP"
    printf 'export SEED=%q\n' "$SEED"
    printf 'export RUN_TAG=%q\n' "$RUN_TAG"
    printf 'export PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
    printf 'export CODE_ROOT=%q\n' "$CODE_ROOT"
    printf 'export RECORD_ROOT=%q\n' "$RECORD_ROOT"
    printf 'export EVAL_ROOT=%q\n' "$EVAL_ROOT"
    printf 'export PYTHON=%q\n' "$PYTHON"
    printf 'export ASR_PYTHON=%q\n' "$ASR_PYTHON"
    printf 'export VALIDATION_JSONL=%q\n' "$VALIDATION_JSONL"
    printf 'export SPEECHBRAIN_ECAPA_MODEL_SOURCE=%q\n' "$SPEECHBRAIN_ECAPA_MODEL_SOURCE"
    printf 'exec bash %q\n' "$FROZEN_DRIVER"
  } > "$RUNNER"
  chmod +x "$RUNNER"
  bash -n "$FROZEN_DRIVER"
  bash -n "$RUNNER"
  {
    sha256sum "$FROZEN_DRIVER" "$RUNNER" "$RESOLVED_TSV"
    sha256sum \
      "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh" \
      "$CODE_ROOT/scripts/004042_summarize_seedtts_validation_eval.py" \
      "$CODE_ROOT/scripts/004044_run_seedtts_validation_infer_persistent.py" \
      "$CODE_ROOT/scripts/004048_summarize_seedtts_ablation_metrics.py" \
      "$CODE_ROOT/scripts/004063_analyze_seedtts320_diagnostics.py"
  } > "$RECORD_ROOT/sha256sums.txt"
}

if [ "$ENTRYPOINT" = "1" ]; then
  run_entrypoint
  exit 0
fi

if [ ! -d "$PROJECT_ROOT" ] || [ ! -x "$QZCLI" ]; then
  echo "ERROR: missing PROJECT_ROOT or qzcli wrapper" >&2
  exit 1
fi
validate_all
mkdir -p "$RECORD_ROOT" "$EVAL_ROOT" "$QZCLI_HOME"
if [ "$FORCE" != "1" ] && {
  [ -s "$RECORD_ROOT/submitted_jobs.tsv" ] ||
  [ -s "$AGG_ROOT/${FAMILY}_step${EVAL_STEP}.official_matrix.json" ];
}; then
  echo "ERROR: existing submission/final matrix; use FORCE=1 only for an intentional full rerun" >&2
  exit 1
fi
write_resolved_configs
write_frozen_runner

COMMAND="bash $RUNNER"
SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"

echo "=========================================="
echo "QZ submit: Ver2.3 family final SeedTTS-320"
echo "  JOB_NAME=$JOB_NAME"
echo "  FAMILY=$FAMILY configs=5"
echo "  EVAL_STEP=$EVAL_STEP SEED=$SEED shards=2"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC"
echo "  CODE_ROOT=$CODE_ROOT"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  EVAL_ROOT=$EVAL_ROOT"
echo "  RUNNER=$RUNNER"
echo "  DRY_RUN=$DRY_RUN"
echo "  COMMAND=$COMMAND"
echo "=========================================="

qz_args=(
  create-job
  --name "$JOB_NAME"
  --command "$COMMAND"
  --workspace "$WORKSPACE"
  --project "$PROJECT"
  --compute-group "$COMPUTE_GROUP"
  --spec "$SPEC"
  --image "$IMAGE"
  --image-type "$IMAGE_TYPE"
  --instances "$INSTANCES"
  --shm "$SHM_GI"
  --priority "$PRIORITY"
  --framework "$FRAMEWORK"
)
if [ "$DRY_RUN" = "1" ]; then
  qz_args+=(--dry-run)
fi

set +e
output=$(
  env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    HOME="$QZCLI_HOME" \
    QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
    "$QZCLI" "${qz_args[@]}" 2>&1
)
status=$?
set -e
printf '%s\n' "$output" | tee "$SUBMIT_OUTPUT"
if [ "$status" -ne 0 ]; then
  echo "ERROR: QZ submission failed; see $SUBMIT_OUTPUT" >&2
  exit "$status"
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "[family-final320] dry-run passed; no job submitted"
  exit 0
fi

job_id=$(printf '%s\n' "$output" | grep -Eo 'job-[0-9a-fA-F-]{36}' | tail -n 1 || true)
{
  printf 'job_name\tjob_id\tfamily\tcompute_group\tspec\tcode_root\trecord_root\teval_root\trunner\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$job_id" "$FAMILY" "$COMPUTE_GROUP" "$SPEC" "$CODE_ROOT" "$RECORD_ROOT" "$EVAL_ROOT" "$RUNNER"
} > "$RECORD_ROOT/submitted_jobs.tsv"
echo "[family-final320] submitted job_id=$job_id record=$RECORD_ROOT/submitted_jobs.tsv"
