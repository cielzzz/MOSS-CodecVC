#!/usr/bin/env bash
# Batch-37 C1/C2 quick20 evaluation on one MTTS 8xH200 node.
#
# Each arm/mode keeps the historical D2+D3 protocol with two inference shards
# and two ASR shards.  The four independent evaluations run concurrently:
#   C1 no_text -> GPUs 0,1      C1 text -> GPUs 2,3
#   C2 no_text -> GPUs 4,5      C2 text -> GPUs 6,7
#
# The frozen training snapshot remains the CODE_ROOT default for provenance,
# but it predates the per-shard transformers dynamic-module cache fix.  Point
# CODE_ROOT at the audited Batch-37 eval snapshot before dry-run/submission:
#
#   CODE_ROOT=/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820 \
#   STEP=500 DRY_RUN=1 bash scripts/004070_submit_ver23_batch37_quick20_qz.sh
#
# Submit only after the dry-run passes:
#
#   CODE_ROOT=/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820 \
#   STEP=500 DRY_RUN=0 bash scripts/004070_submit_ver23_batch37_quick20_qz.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
DEFAULT_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_20260710_1092820"
RECOMMENDED_EVAL_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
CODE_ROOT="${CODE_ROOT:-$DEFAULT_CODE_ROOT}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"  # MTTS-3-2-0715
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"              # 8xH200
ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-$ALLOWED_GPU_TYPE}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"

# This is the immutable training-run stamp used by both Batch-37 arms.
TRAIN_STAMP="20260710_mtts"
STEP="${STEP:-500}"
EVAL_STAMP="${EVAL_STAMP:-20260711_mtts}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENTRYPOINT="${BATCH37_QUICK20_ENTRYPOINT:-0}"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_content_side_batch37_quick20_step${STEP}_${EVAL_STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_content_side_batch37_quick_eval_step${STEP}_${EVAL_STAMP}}"
JOB_NAME="${JOB_NAME:-ver23_batch37_quick20_step${STEP}_${EVAL_STAMP}}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
NO_TEXT20_JSONL="${NO_TEXT20_JSONL:-$PROJECT_ROOT/testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
TEXT_SOURCE_JSONL="${TEXT_SOURCE_JSONL:-$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl}"
TEXT20_JSONL="${TEXT20_JSONL:-$RECORD_ROOT/ver23_batch37_text_quick20_8cell_20260710.jsonl}"
TEXT20_SHA256="0952c4162e7ff7a9c2850f1f76f572f2f710e205b222c874016b05564f21bea8"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "$STEP" in
  500|1000|1500|2000|2500|3000) ;;
  *) die "STEP must be one of 500,1000,1500,2000,2500,3000; got $STEP" ;;
esac
case "$DRY_RUN:$FORCE:$ENTRYPOINT" in
  [01]:[01]:[01]) ;;
  *) die "DRY_RUN, FORCE, and BATCH37_QUICK20_ENTRYPOINT must be 0 or 1" ;;
esac
if [ "${STAMP:-$TRAIN_STAMP}" != "$TRAIN_STAMP" ]; then
  die "Batch-37 training STAMP is fixed to $TRAIN_STAMP; got STAMP=${STAMP:-}"
fi
if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  die "Batch-37 quick20 is restricted to MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP); got $COMPUTE_GROUP"
fi
if [ "$SPEC" != "$ALLOWED_SPEC" ]; then
  die "Batch-37 quick20 requires the registered 8xH200 spec $ALLOWED_SPEC; got $SPEC"
fi
if [ "$QZCLI_GPU_TYPE_OVERRIDE" != "$ALLOWED_GPU_TYPE" ]; then
  die "QZCLI_GPU_TYPE_OVERRIDE must be $ALLOWED_GPU_TYPE; got $QZCLI_GPU_TYPE_OVERRIDE"
fi
if [ "$INSTANCES" != "1" ]; then
  die "Batch-37 quick20 requires exactly one 8xH200 instance; got INSTANCES=$INSTANCES"
fi

arm_info() {
  case "$1" in
    C1)
      printf '%s\t%s\n' \
        "ver23_compact_content_lr_warmup_3k" \
        "ver23_batch37_C1_compact_content_lr_warmup"
      ;;
    C2)
      printf '%s\t%s\n' \
        "ver23_true_ref_audio_cfg_3k" \
        "ver23_batch37_C2_true_ref_audio_cfg"
      ;;
    *) die "unsupported Batch-37 arm: $1" ;;
  esac
}

arm_run_dir() {
  local arm="$1"
  local suffix label
  IFS=$'\t' read -r suffix label <<<"$(arm_info "$arm")"
  printf '%s\n' "$PROJECT_ROOT/outputs/lora_runs/ver23_content_side_batch37_${arm}_${suffix}_${TRAIN_STAMP}"
}

arm_checkpoint() {
  printf '%s/step-%s\n' "$(arm_run_dir "$1")" "$STEP"
}

arm_label() {
  local suffix label
  IFS=$'\t' read -r suffix label <<<"$(arm_info "$1")"
  printf '%s\n' "$label"
}

run_id_for() {
  local arm="$1"
  local mode="$2"
  local label middle=""
  label="$(arm_label "$arm")"
  if [ "$mode" = "text" ]; then
    middle="text_"
  fi
  printf '%s_step-%s_%squick20_d2d3_seed1234\n' "$label" "$STEP" "$middle"
}

audit_code_root() {
  [ -d "$CODE_ROOT" ] || die "CODE_ROOT does not exist: $CODE_ROOT"
  "$PYTHON" - "$CODE_ROOT" "$RECOMMENDED_EVAL_CODE_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
recommended = Path(sys.argv[2])
paths = {
    "eval": root / "scripts/004039_run_seedtts_validation_eval.sh",
    "infer": root / "scripts/004044_run_seedtts_validation_infer_persistent.py",
    "ref_content": root / "scripts/004056_summarize_seedtts_ref_content_similarity.py",
    "speaker": root / "scripts/004050_summarize_seedtts_speaker_sim_only.py",
    "wrapper": root / "moss_codecvc/models/moss_codecvc_wrapper.py",
}
missing = [str(path) for path in paths.values() if not path.is_file()]
if missing:
    raise SystemExit("missing CODE_ROOT files: " + ", ".join(missing))

eval_text = paths["eval"].read_text(encoding="utf-8")
cache_needles = (
    'local hf_modules_cache_root="${HF_MODULES_CACHE_ROOT:-$OUTPUT_DIR/.hf_modules_cache}"',
    'export HF_MODULES_CACHE="$hf_modules_cache_root/shard${shard}"',
    'mkdir -p "$HF_MODULES_CACHE"',
)
missing_cache = [needle for needle in cache_needles if needle not in eval_text]
if missing_cache:
    raise SystemExit(
        "CODE_ROOT lacks mandatory per-inference-shard HF dynamic-module cache isolation. "
        f"Use CODE_ROOT={recommended}. Missing snippets: {missing_cache}"
    )

infer_text = paths["infer"].read_text(encoding="utf-8")
wrapper_text = paths["wrapper"].read_text(encoding="utf-8")
if "REF_AUDIO_CFG_SCALE" not in infer_text or "ref_audio_cfg_scale" not in infer_text:
    raise SystemExit("CODE_ROOT persistent inference does not expose Batch-37 ref-audio CFG support")
for needle in ("content_encoder_hidden_size", "ref_audio_cfg_dropout", "ref_audio_cfg_scale"):
    if needle not in wrapper_text:
        raise SystemExit(f"CODE_ROOT wrapper lacks Batch-37 feature: {needle}")

sha = hashlib.sha256(paths["eval"].read_bytes()).hexdigest()
print(f"[batch37-quick20-audit] code_root={root}")
print(f"[batch37-quick20-audit] 004039_sha256={sha}")
print("[batch37-quick20-audit] per_shard_hf_modules_cache=PASS batch37_infer=PASS")
PY
  bash -n "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"
}

validate_checkpoint() {
  local arm="$1"
  local checkpoint
  checkpoint="$(arm_checkpoint "$arm")"
  local required=(
    adapter_model.safetensors
    adapter_config.json
    README.md
    timbre_memory_adapter.pt
    timbre_memory_config.json
  )
  [ -d "$checkpoint" ] || die "missing checkpoint directory for $arm: $checkpoint"
  local name
  for name in "${required[@]}"; do
    [ -s "$checkpoint/$name" ] || die "missing or empty checkpoint file for $arm: $checkpoint/$name"
  done
  "$PYTHON" - "$arm" "$checkpoint/timbre_memory_config.json" <<'PY'
import json
import math
import sys

arm, path = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    cfg = json.load(handle)
expected = {
    "C1": {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "7,15,23,31,35",
        "content_encoder_hidden_size": 512,
        "ref_audio_cfg_dropout": 0.0,
    },
    "C2": {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_encoder_hidden_size": 0,
        "ref_audio_cfg_dropout": 0.15,
    },
}[arm]
errors = []
for key, wanted in expected.items():
    got = cfg.get(key)
    if isinstance(wanted, float):
        try:
            ok = math.isclose(float(got), wanted, rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            ok = False
    else:
        ok = got == wanted
    if not ok:
        errors.append(f"{key}={got!r}, expected {wanted!r}")
if errors:
    raise SystemExit(f"{arm} checkpoint config mismatch: " + "; ".join(errors))
print(f"[batch37-quick20-checkpoint] arm={arm} config=PASS path={path}")
PY
}

prepare_text20() {
  mkdir -p "$RECORD_ROOT"
  "$PYTHON" - "$TEXT_SOURCE_JSONL" "$TEXT20_JSONL" "$TEXT20_SHA256" \
    seedtts_text_en_src_zh_ref_m2f_000000 \
    seedtts_text_en_src_zh_ref_m2f_000001 \
    seedtts_text_en_src_zh_ref_m2f_000002 \
    seedtts_text_en_src_zh_ref_f2m_000000 \
    seedtts_text_en_src_zh_ref_f2m_000001 \
    seedtts_text_en_src_zh_ref_f2m_000002 \
    seedtts_text_zh_src_en_ref_m2f_000000 \
    seedtts_text_zh_src_en_ref_m2f_000001 \
    seedtts_text_zh_src_en_ref_f2m_000000 \
    seedtts_text_zh_src_en_ref_f2m_000001 \
    seedtts_text_en_src_en_ref_same_gender_000000 \
    seedtts_text_en_src_en_ref_same_gender_000001 \
    seedtts_text_en_src_en_ref_same_gender_000002 \
    seedtts_text_zh_src_zh_ref_same_gender_000000 \
    seedtts_text_zh_src_zh_ref_same_gender_000001 \
    seedtts_text_en_src_zh_ref_same_gender_000000 \
    seedtts_text_en_src_zh_ref_same_gender_000001 \
    seedtts_text_en_src_zh_ref_same_gender_000002 \
    seedtts_text_zh_src_en_ref_same_gender_000000 \
    seedtts_text_zh_src_en_ref_same_gender_000001 <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
expected_sha = sys.argv[3]
case_ids = sys.argv[4:]
rows = {}
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    case_id = str(row.get("case_id") or "")
    if case_id in case_ids:
        if case_id in rows:
            raise SystemExit(f"duplicate case_id in source: {case_id}")
        rows[case_id] = row
missing = [case_id for case_id in case_ids if case_id not in rows]
if missing:
    raise SystemExit(f"missing text20 cases: {missing}")
selected = [rows[case_id] for case_id in case_ids]
counts = Counter(str(row.get("cell") or "") for row in selected)
expected_counts = sorted([2, 2, 2, 2, 3, 3, 3, 3])
if len(selected) != 20 or sorted(counts.values()) != expected_counts:
    raise SystemExit(f"unexpected text20 distribution: n={len(selected)} counts={dict(counts)}")
payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
actual_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
if actual_sha != expected_sha:
    raise SystemExit(f"text20 SHA256 mismatch: {actual_sha}, expected {expected_sha}")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(payload, encoding="utf-8")
print(f"[batch37-quick20] text20={output} n=20 sha256={actual_sha} cells={dict(counts)}")
PY
}

validate_no_text20() {
  "$PYTHON" - "$NO_TEXT20_JSONL" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 20:
    raise SystemExit(f"no_text quick set must contain 20 rows; got {len(rows)}: {path}")
bad = [row.get("case_id") for row in rows if row.get("mode") != "no_text"]
if bad:
    raise SystemExit(f"no_text quick set contains wrong modes: {bad}")
print(f"[batch37-quick20] no_text20={path} n=20 modes=PASS")
PY
}

ensure_submission_is_new() {
  if [ "$FORCE" = "1" ]; then
    return
  fi
  if [ -s "$RECORD_ROOT/metrics.tsv" ]; then
    die "metrics already exist; refusing duplicate evaluation: $RECORD_ROOT/metrics.tsv"
  fi
  if [ "$ENTRYPOINT" != "1" ]; then
    if [ -s "$RECORD_ROOT/submitted_jobs.tsv" ] && grep -Eq 'job-[0-9a-fA-F-]{36}' "$RECORD_ROOT/submitted_jobs.tsv"; then
      die "a QZ job is already recorded; refusing duplicate submission: $RECORD_ROOT/submitted_jobs.tsv"
    fi
    if [ -s "$RECORD_ROOT/submit_output.txt" ] && grep -Eq 'job-[0-9a-fA-F-]{36}' "$RECORD_ROOT/submit_output.txt"; then
      die "a prior QZ job ID exists in submit output; refusing duplicate submission: $RECORD_ROOT/submit_output.txt"
    fi
  fi
  local arm mode run_id output_dir
  for arm in C1 C2; do
    for mode in no_text text; do
      run_id="$(run_id_for "$arm" "$mode")"
      output_dir="$EVAL_ROOT/$run_id"
      if [ -s "$output_dir/${run_id}.summary.json" ] || compgen -G "$output_dir/manifest.shard*.jsonl" >/dev/null; then
        die "evaluation output already exists; use FORCE=1 only for an intentional rerun: $output_dir"
      fi
    done
  done
}

audit_runtime_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable in QZ entrypoint"
  local gpu_count
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d ' ')
  [ "$gpu_count" = "8" ] || die "QZ runtime must expose exactly 8 GPUs; got $gpu_count"
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Evq 'H200'; then
    die "QZ runtime contains a non-H200 GPU"
  fi
  echo "[batch37-quick20-audit] runtime_gpu_count=8 runtime_gpu_type=H200 PASS"
}

run_eval() {
  local arm="$1"
  local mode="$2"
  local gpu_pair="$3"
  local checkpoint validation_jsonl run_id output_dir log
  checkpoint="$(arm_checkpoint "$arm")"
  if [ "$mode" = "no_text" ]; then
    validation_jsonl="$NO_TEXT20_JSONL"
  else
    validation_jsonl="$TEXT20_JSONL"
  fi
  run_id="$(run_id_for "$arm" "$mode")"
  output_dir="$EVAL_ROOT/$run_id"
  log="$RECORD_ROOT/eval_${arm}_${mode}_step${STEP}.log"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu_pair"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=8
    export SPEAKER_ENCODER_TYPE=embedding_loader
    echo "[batch37-quick20] arm=$arm mode=$mode gpu_pair=$gpu_pair checkpoint=$checkpoint run_id=$run_id"

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
    REF_AUDIO_CFG_SCALE=1.0 \
    HF_MODULES_CACHE_ROOT="$output_dir/.hf_modules_cache" \
    INFER_SHARD_START_DELAY_SEC=3 \
    PYTHON="$PYTHON" \
    ASR_PYTHON="$ASR_PYTHON" \
    VALIDATION_JSONL="$validation_jsonl" \
    MODEL_PATH="$checkpoint" \
    RUN_ID="$run_id" \
    RUN_LABEL="$run_id" \
    OUTPUT_DIR="$output_dir" \
    MODE="$mode" \
    MAX_CASES=0 \
    PER_MODE=0 \
    PER_CELL=0 \
    DECODING_PROFILE=default \
    PERSISTENT_INFER=1 \
    OVERWRITE_INFER=1 \
    RESET_MANIFESTS=1 \
    RUN_ASR=1 \
    RUN_SUMMARY=1 \
    BUILD_PAGE=0 \
    GPU_COUNT=2 \
    NUM_SHARDS=2 \
    ASR_NUM_SHARDS=2 \
    SEED=1234 \
    bash "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"

    "$PYTHON" "$CODE_ROOT/scripts/004056_summarize_seedtts_ref_content_similarity.py" \
      --asr-jsonl "$output_dir/${run_id}.asr_eval.jsonl" \
      --output-json "$output_dir/${run_id}.ref_content_similarity_summary.json" \
      --output-md "$output_dir/${run_id}.ref_content_similarity_summary.md"

    HF_MODULES_CACHE="$output_dir/.hf_modules_cache/speaker_summary" \
    "$PYTHON" "$CODE_ROOT/scripts/004050_summarize_seedtts_speaker_sim_only.py" \
      --validation-jsonl "$validation_jsonl" \
      --run "$run_id=$output_dir" \
      --output-csv "$output_dir/${run_id}.speaker_sim.csv" \
      --summary-json "$output_dir/${run_id}.speaker_sim_summary.json" \
      --summary-md "$output_dir/${run_id}.speaker_sim_summary.md" \
      --speaker-device cuda:0

    echo "[batch37-quick20] arm=$arm mode=$mode complete output=$output_dir"
  ) > >(tee -a "$log") 2>&1
}

collect_metrics() {
  "$PYTHON" - "$EVAL_ROOT" "$RECORD_ROOT" "$STEP" <<'PY'
import csv
import json
import sys
from pathlib import Path

eval_root = Path(sys.argv[1])
record_root = Path(sys.argv[2])
step = int(sys.argv[3])
labels = {
    "C1": "ver23_batch37_C1_compact_content_lr_warmup",
    "C2": "ver23_batch37_C2_true_ref_audio_cfg",
}

rows_out = []
for arm in ("C1", "C2"):
    label = labels[arm]
    for mode in ("no_text", "text"):
        middle = "text_" if mode == "text" else ""
        run_id = f"{label}_step-{step}_{middle}quick20_d2d3_seed1234"
        out_dir = eval_root / run_id
        summary_path = out_dir / f"{run_id}.summary.json"
        speaker_path = out_dir / f"{run_id}.speaker_sim.csv"
        ref_content_path = out_dir / f"{run_id}.ref_content_similarity_summary.json"
        for path in (summary_path, speaker_path, ref_content_path):
            if not path.is_file():
                raise SystemExit(f"missing metric input: {path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))["overall"]
        speaker_rows = list(csv.DictReader(speaker_path.open(encoding="utf-8")))
        valid = [row for row in speaker_rows if row.get("status") in {"ok", "ok_after_rerun", "skipped_exists"}]
        n = int(summary["n"])
        keep = int(summary["keep"])
        if n != 20 or len(valid) != 20:
            raise SystemExit(f"incomplete quick20 metrics: run={run_id} summary_n={n} speaker_n={len(valid)}")
        sim_ref = sum(float(row["sim_gen_ref"]) for row in valid) / len(valid)
        sim_src = sum(float(row["sim_gen_source"]) for row in valid) / len(valid)
        ref_bound_count = sum(
            float(row["sim_gen_ref"]) - float(row["sim_gen_source"]) > 0.05
            for row in valid
        )
        ref_content = json.loads(ref_content_path.read_text(encoding="utf-8"))["overall"]
        rows_out.append({
            "arm": arm,
            "mode": mode,
            "n": n,
            "keep": keep,
            "fail": (n - keep) / n,
            "cer": float(summary["cer"]),
            "sim_ref": sim_ref,
            "sim_src": sim_src,
            "ref_bound_count": ref_bound_count,
            "ref_bound": ref_bound_count / len(valid),
            "ref_content_f1": float(ref_content["ref_content_lcs_f1_mean"]),
            "run_id": run_id,
            "output_dir": str(out_dir),
        })

record_root.mkdir(parents=True, exist_ok=True)
(record_root / "metrics.json").write_text(
    json.dumps(rows_out, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
fields = [
    "arm", "mode", "n", "keep", "fail", "cer", "sim_ref", "sim_src",
    "ref_bound_count", "ref_bound", "ref_content_f1", "run_id", "output_dir",
]
with (record_root / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows_out)

lines = [
    f"# Batch-37 quick20 step-{step}",
    "",
    "| Arm | Mode | fail | CER | sim(ref) | sim(src) | ref-bound | F1(ref-content) |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows_out:
    lines.append(
        f"| {row['arm']} | {row['mode']} | {row['fail']:.1%} | {row['cer']:.4f} | "
        f"{row['sim_ref']:.4f} | {row['sim_src']:.4f} | {row['ref_bound']:.1%} | "
        f"{row['ref_content_f1']:.4f} |"
    )
(record_root / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
for row in rows_out:
    print(
        "[batch37-quick20-metric] "
        f"{row['arm']} {row['mode']} fail={row['fail']:.1%} CER={row['cer']:.4f} "
        f"sim_ref={row['sim_ref']:.4f} sim_src={row['sim_src']:.4f} "
        f"ref_bound={row['ref_bound']:.1%} ref_content_f1={row['ref_content_f1']:.4f}"
    )
PY
}

run_entrypoint() {
  audit_code_root
  validate_checkpoint C1
  validate_checkpoint C2
  validate_no_text20
  prepare_text20
  ensure_submission_is_new
  audit_runtime_gpus

  mkdir -p "$RECORD_ROOT" "$EVAL_ROOT"
  exec > >(tee -a "$RECORD_ROOT/run.log") 2>&1
  echo "[batch37-quick20] start date=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "[batch37-quick20] step=$STEP train_stamp=$TRAIN_STAMP eval_stamp=$EVAL_STAMP"
  echo "[batch37-quick20] code_root=$CODE_ROOT compute_group=MTTS-3-2-0715 gpu_plan=C1-no_text:0,1+C1-text:2,3+C2-no_text:4,5+C2-text:6,7"
  nvidia-smi || true

  local pids=()
  local failed=0
  run_eval C1 no_text 0,1 & pids+=("$!")
  run_eval C1 text 2,3 & pids+=("$!")
  run_eval C2 no_text 4,5 & pids+=("$!")
  run_eval C2 text 6,7 & pids+=("$!")
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    die "one or more Batch-37 arm/mode evaluations failed"
  fi

  collect_metrics
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RECORD_ROOT/complete.marker"
  echo "[batch37-quick20] complete metrics=$RECORD_ROOT/metrics.tsv date=$(cat "$RECORD_ROOT/complete.marker")"
}

if [ "$ENTRYPOINT" = "1" ]; then
  run_entrypoint
  exit 0
fi

[ -d "$PROJECT_ROOT" ] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
[ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"
[ -x "$ASR_PYTHON" ] || die "ASR Python interpreter is not executable: $ASR_PYTHON"
[ -x "$QZCLI" ] || die "qzcli wrapper is not executable: $QZCLI"
[ -s "$NO_TEXT20_JSONL" ] || die "missing no_text quick20 JSONL: $NO_TEXT20_JSONL"
[ -s "$TEXT_SOURCE_JSONL" ] || die "missing text validation source JSONL: $TEXT_SOURCE_JSONL"

audit_code_root
validate_checkpoint C1
validate_checkpoint C2
validate_no_text20
prepare_text20
ensure_submission_is_new
mkdir -p "$QZCLI_HOME"

COMMAND="env BATCH37_QUICK20_ENTRYPOINT=1 STEP=$STEP EVAL_STAMP=$EVAL_STAMP FORCE=$FORCE PROJECT_ROOT=$PROJECT_ROOT CODE_ROOT=$CODE_ROOT RECORD_ROOT=$RECORD_ROOT EVAL_ROOT=$EVAL_ROOT PYTHON=$PYTHON ASR_PYTHON=$ASR_PYTHON bash $CODE_ROOT/scripts/004070_submit_ver23_batch37_quick20_qz.sh"
SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"

echo "=========================================="
echo "QZ submit: Batch-37 C1/C2 quick20"
echo "  JOB_NAME=$JOB_NAME"
echo "  STEP=$STEP TRAIN_STAMP=$TRAIN_STAMP EVAL_STAMP=$EVAL_STAMP"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC (8x $ALLOWED_GPU_TYPE)"
echo "  INSTANCES=$INSTANCES"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  EVAL_ROOT=$EVAL_ROOT"
echo "  CODE_ROOT=$CODE_ROOT"
echo "  GPU_PLAN=C1-no_text:0,1 C1-text:2,3 C2-no_text:4,5 C2-text:6,7"
echo "  SHARDS=2 inference + 2 ASR per arm/mode"
echo "  SPEAKER_ENCODER_TYPE=embedding_loader"
echo "  REF_AUDIO_CFG_SCALE=1.0"
echo "  DRY_RUN=$DRY_RUN FORCE=$FORCE"
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
  die "QZ dry-run/submission failed; see $SUBMIT_OUTPUT"
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "[batch37-quick20] dry-run passed; no QZ job submitted"
  exit 0
fi

job_id=$(printf '%s\n' "$output" | grep -Eo 'job-[0-9a-fA-F-]{36}' | tail -n 1 || true)
[ -n "$job_id" ] || die "QZ reported success but no job ID was parsed; inspect $SUBMIT_OUTPUT before retrying"
{
  printf 'job_name\tjob_id\tstep\tcompute_group\tspec\trecord_root\teval_root\tcode_root\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$job_id" "$STEP" "$COMPUTE_GROUP" "$SPEC" "$RECORD_ROOT" "$EVAL_ROOT" "$CODE_ROOT"
} > "$RECORD_ROOT/submitted_jobs.tsv"
echo "[batch37-quick20] submitted job_id=$job_id record=$RECORD_ROOT/submitted_jobs.tsv"
