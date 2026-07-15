#!/usr/bin/env bash
# Score the historical Ver2.3 and Batch-33 SeedTTS-320 baselines with the
# same dual-encoder protocol used by 004072.
#
# The historical Ver2.3 directory has 320 complete WAVs but no merged ASR
# JSONL.  This wrapper therefore builds an isolated read-only view under its
# own EVAL_ROOT and runs the same two-shard Qwen-ASR content pass used by
# 004072 before invoking 004048.
# Neither historical source directory is modified.
#
# Dry-run (default; never submits):
#   bash scripts/004076_submit_seedtts_baseline_dual_encoder_qz.sh
#
# Submit only after explicit review:
#   DRY_RUN=0 bash scripts/004076_submit_seedtts_baseline_dual_encoder_qz.sh

set -euo pipefail

SELF_PATH=$(readlink -f "$0")
PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
CODE_ROOT="${CODE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"  # MTTS-3-2-0715
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"
SPEC="${SPEC:-$ALLOWED_SPEC}"
ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-$ALLOWED_GPU_TYPE}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"

RUN_TAG="${RUN_TAG:-20260711_mtts}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENTRYPOINT="${SEEDTTS_BASELINE_DUAL_ENCODER_ENTRYPOINT:-0}"
KEEPALIVE_UNUSED_GPUS="${KEEPALIVE_UNUSED_GPUS:-1}"
ASR_NUM_SHARDS="${ASR_NUM_SHARDS:-2}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
VALIDATION_JSONL="${VALIDATION_JSONL:-$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl}"
VER23_SOURCE_DIR="${VER23_SOURCE_DIR:-$PROJECT_ROOT/testset/outputs/ver2_3_ctc_clean_seedtts_valid_full}"
BATCH33_SOURCE_DIR="${BATCH33_SOURCE_DIR:-$PROJECT_ROOT/testset/outputs/ver23_content_side_text_bypass_3k_seedtts320_20260710/ver23_content_side_text_bypass_3k_step-3000_seedtts320_all_d2d3_seed1234}"
SPEAKER_SIM_ROOT="${SPEAKER_SIM_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
SPEECHBRAIN_ECAPA_MODEL_SOURCE="${SPEECHBRAIN_ECAPA_MODEL_SOURCE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b}"
QWEN_ASR_DTYPE="${QWEN_ASR_DTYPE:-bfloat16}"
QWEN_ASR_MAX_BATCH_SIZE="${QWEN_ASR_MAX_BATCH_SIZE:-1}"
QWEN_ASR_MAX_NEW_TOKENS="${QWEN_ASR_MAX_NEW_TOKENS:-256}"

RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/seedtts_baseline_dual_encoder_${RUN_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/seedtts_baseline_dual_encoder_${RUN_TAG}}"
VIEW_ROOT="$EVAL_ROOT/run_views"
VER23_VIEW="$VIEW_ROOT/ver2_3"
BATCH33_VIEW="$VIEW_ROOT/Batch33"
DIAGNOSTICS_ROOT="$EVAL_ROOT/diagnostics"
DUAL_CASES="$EVAL_ROOT/dual_encoder_cases.csv"
DUAL_SUMMARY_JSON="$EVAL_ROOT/dual_encoder_summary.json"
DUAL_SUMMARY_MD="$EVAL_ROOT/dual_encoder_summary.md"
RESOLVED_RUNS="$RECORD_ROOT/resolved_runs.tsv"
FROZEN_DRIVER="$RECORD_ROOT/004076_baseline_dual_encoder.frozen.sh"
RUNNER="$RECORD_ROOT/run_seedtts_baseline_dual_encoder_entrypoint.sh"
JOB_NAME="${JOB_NAME:-seedtts_baseline_dual_encoder_${RUN_TAG}}"

ASR_SCRIPT="$CODE_ROOT/scripts/001017_asr_content_filter.py"
BUILD_EVAL_SCRIPT="$CODE_ROOT/scripts/004017_build_seedtts_generated_eval_jsonl.py"
SUMMARY_SCRIPT="$CODE_ROOT/scripts/004042_summarize_seedtts_validation_eval.py"
DUAL_ENCODER_SCRIPT="$CODE_ROOT/scripts/004048_summarize_seedtts_ablation_metrics.py"
DIAGNOSTICS_SCRIPT="$CODE_ROOT/scripts/004063_analyze_seedtts320_diagnostics.py"

if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: baseline scoring is restricted to MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP); got $COMPUTE_GROUP" >&2
  exit 2
fi
if [ "$SPEC" != "$ALLOWED_SPEC" ]; then
  echo "ERROR: this wrapper requires the MTTS 8xH200 spec $ALLOWED_SPEC; got $SPEC" >&2
  exit 2
fi
if [ "$QZCLI_GPU_TYPE_OVERRIDE" != "$ALLOWED_GPU_TYPE" ]; then
  echo "ERROR: this wrapper requires GPU type $ALLOWED_GPU_TYPE; got $QZCLI_GPU_TYPE_OVERRIDE" >&2
  exit 2
fi
if [ "$INSTANCES" != "1" ]; then
  echo "ERROR: this wrapper requires exactly one 8xH200 instance; got INSTANCES=$INSTANCES" >&2
  exit 2
fi
if [ "$ASR_NUM_SHARDS" != "2" ]; then
  echo "ERROR: matching 004072 requires ASR_NUM_SHARDS=2; got $ASR_NUM_SHARDS" >&2
  exit 2
fi
case "$DRY_RUN:$FORCE:$ENTRYPOINT:$KEEPALIVE_UNUSED_GPUS" in
  [01]:[01]:[01]:[01]) ;;
  *)
    echo "ERROR: DRY_RUN, FORCE, SEEDTTS_BASELINE_DUAL_ENCODER_ENTRYPOINT, and KEEPALIVE_UNUSED_GPUS must be 0 or 1" >&2
    exit 2
    ;;
esac

audit_code_root() {
  local required
  if [ ! -d "$CODE_ROOT" ]; then
    echo "ERROR: missing safe eval CODE_ROOT: $CODE_ROOT" >&2
    return 1
  fi
  for required in \
    "$ASR_SCRIPT" \
    "$BUILD_EVAL_SCRIPT" \
    "$SUMMARY_SCRIPT" \
    "$DUAL_ENCODER_SCRIPT" \
    "$DIAGNOSTICS_SCRIPT"; do
    if [ ! -s "$required" ]; then
      echo "ERROR: missing safe eval script: $required" >&2
      return 1
    fi
  done
  if ! grep -q 'extra-speaker-encoder' "$DUAL_ENCODER_SCRIPT" || \
     ! grep -q 'speechbrain_ecapa' "$DUAL_ENCODER_SCRIPT"; then
    echo "ERROR: safe 004048 lacks the SpeechBrain ECAPA dual-encoder path" >&2
    return 1
  fi
  if ! grep -q 'ecapa_binding' "$DIAGNOSTICS_SCRIPT"; then
    echo "ERROR: safe 004063 lacks dual-encoder binding diagnostics" >&2
    return 1
  fi
}

audit_root_separation() {
  "$PYTHON" - "$RECORD_ROOT" "$EVAL_ROOT" "$VER23_SOURCE_DIR" "$BATCH33_SOURCE_DIR" <<'PY'
import sys
from pathlib import Path

record, output, ver23, batch33 = [Path(value).expanduser().resolve() for value in sys.argv[1:]]

def overlaps(left, right):
    return left == right or left in right.parents or right in left.parents

if overlaps(record, output):
    raise SystemExit(f"RECORD_ROOT and EVAL_ROOT must be independent: {record} vs {output}")
if overlaps(ver23, batch33):
    raise SystemExit(f"baseline source directories unexpectedly overlap: {ver23} vs {batch33}")
for destination_name, destination in (("RECORD_ROOT", record), ("EVAL_ROOT", output)):
    for source_name, source in (("VER23_SOURCE_DIR", ver23), ("BATCH33_SOURCE_DIR", batch33)):
        if overlaps(destination, source):
            raise SystemExit(
                f"{destination_name} must not equal, contain, or be contained by {source_name}: "
                f"{destination} vs {source}"
            )
print(f"[baseline-root-audit] PASS record={record} eval={output}")
PY
}

audit_source_runs() {
  "$PYTHON" - "$VALIDATION_JSONL" "$VER23_SOURCE_DIR" "$BATCH33_SOURCE_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

validation_path, ver23_dir, batch33_dir = map(Path, sys.argv[1:])

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

validation = rows(validation_path)
validation_ids = [str(row.get("case_id") or "") for row in validation]
validation_modes = Counter(str(row.get("mode") or "") for row in validation)
if len(validation) != 320 or len(set(validation_ids)) != 320:
    raise SystemExit(f"validation rows/unique != 320: {len(validation)}/{len(set(validation_ids))}")
if validation_modes != Counter({"no_text": 160, "text": 160}):
    raise SystemExit(f"validation modes != 160+160: {dict(validation_modes)}")
expected = set(validation_ids)
for row in validation:
    case_id = str(row.get("case_id") or "")
    for field in ("source_audio", "timbre_ref_audio"):
        path = Path(str(row.get(field) or ""))
        if not path.is_file():
            raise SystemExit(f"validation {case_id}: missing {field}: {path}")

def final_manifests(root, include_rerun):
    by_id = {}
    shard_paths = sorted(root.glob("manifest.shard*.jsonl"))
    if len(shard_paths) != 2:
        raise SystemExit(f"{root}: expected exactly 2 primary manifests, got {len(shard_paths)}")
    for path in shard_paths:
        for row in rows(path):
            case_id = str(row.get("case_id") or "")
            if case_id:
                by_id[case_id] = row
    if include_rerun:
        rerun = root / "manifest.rerun_failed.jsonl"
        if rerun.exists():
            for row in rows(rerun):
                case_id = str(row.get("case_id") or "")
                if case_id:
                    by_id[case_id] = row
    return by_id

def audit_run(name, root, include_rerun):
    if not root.is_dir():
        raise SystemExit(f"{name}: missing source directory: {root}")
    wavs = list(root.glob("*.wav"))
    wav_ids = {path.stem for path in wavs}
    manifests = final_manifests(root, include_rerun)
    manifest_modes = Counter(str(row.get("mode") or "") for row in manifests.values())
    if len(wavs) != 320 or wav_ids != expected:
        raise SystemExit(f"{name}: WAV rows/set mismatch: rows={len(wavs)} unique={len(wav_ids)}")
    if set(manifests) != expected or len(manifests) != 320:
        raise SystemExit(f"{name}: manifest rows/set mismatch: rows={len(manifests)}")
    if manifest_modes != Counter({"no_text": 160, "text": 160}):
        raise SystemExit(f"{name}: manifest modes != 160+160: {dict(manifest_modes)}")
    bad_status = Counter(str(row.get("status") or "") for row in manifests.values())
    if set(bad_status) - {"ok", "ok_after_rerun", "skipped_exists"}:
        raise SystemExit(f"{name}: unexpected final statuses: {dict(bad_status)}")
    for case_id, row in manifests.items():
        target = Path(str(row.get("output_wav") or root / f"{case_id}.wav"))
        if not target.is_file() or target.stem != case_id:
            raise SystemExit(f"{name}/{case_id}: missing or mismatched target WAV: {target}")
    print(f"[baseline-source-audit] {name} PASS wav=320 manifest=320 modes=160+160 status={dict(bad_status)}")

audit_run("ver2_3", ver23_dir, True)
audit_run("Batch33", batch33_dir, False)

merged = [path for path in batch33_dir.glob("*.asr_eval.jsonl") if ".shard" not in path.name]
if len(merged) != 1:
    raise SystemExit(f"Batch33: expected one merged ASR JSONL, got {len(merged)}")
asr = rows(merged[0])
asr_ids = [str(row.get("case_id") or row.get("sample_id") or "") for row in asr]
asr_modes = Counter(str(row.get("mode") or "") for row in asr)
if len(asr) != 320 or set(asr_ids) != expected or len(set(asr_ids)) != 320:
    raise SystemExit(f"Batch33: ASR rows/set mismatch: rows={len(asr)} unique={len(set(asr_ids))}")
if asr_modes != Counter({"no_text": 160, "text": 160}):
    raise SystemExit(f"Batch33: ASR modes != 160+160: {dict(asr_modes)}")
print(f"[baseline-source-audit] Batch33 merged ASR PASS rows=320 modes=160+160 path={merged[0]}")
PY
}

validate_inputs() {
  audit_code_root
  if [ ! -x "$PYTHON" ] || [ ! -x "$ASR_PYTHON" ] || [ ! -x "$QZCLI" ]; then
    echo "ERROR: missing Python interpreter or qzcli wrapper" >&2
    return 1
  fi
  if [ ! -s "$VALIDATION_JSONL" ]; then
    echo "ERROR: missing validation JSONL: $VALIDATION_JSONL" >&2
    return 1
  fi
  if [ ! -d "$SPEAKER_SIM_ROOT" ] || [ ! -d "$SPEECHBRAIN_ECAPA_MODEL_SOURCE" ]; then
    echo "ERROR: missing local WavLM/ECAPA scorer source or SpeechBrain ECAPA model" >&2
    return 1
  fi
  if [ ! -d "$QWEN_ASR_MODEL" ]; then
    echo "ERROR: missing local Qwen-ASR model: $QWEN_ASR_MODEL" >&2
    return 1
  fi
  audit_root_separation
  audit_source_runs
}

write_resolved_runs() {
  mkdir -p "$RECORD_ROOT"
  {
    printf 'run\tsource_dir\tview_dir\tasr_policy\n'
    printf 'ver2_3\t%s\t%s\t%s\n' "$VER23_SOURCE_DIR" "$VER23_VIEW" "qwen_asr_2shard_isolated_view_matching_004072"
    printf 'Batch33\t%s\t%s\t%s\n' "$BATCH33_SOURCE_DIR" "$BATCH33_VIEW" "reuse_audited_merged_asr_in_isolated_view"
  } > "$RESOLVED_RUNS"
}

prepare_run_views() {
  mkdir -p "$VER23_VIEW" "$BATCH33_VIEW" "$DIAGNOSTICS_ROOT"
  "$PYTHON" - \
    "$VALIDATION_JSONL" \
    "$VER23_SOURCE_DIR" "$BATCH33_SOURCE_DIR" \
    "$VER23_VIEW" "$BATCH33_VIEW" <<'PY'
import json
import sys
from pathlib import Path

validation_path, ver23_source, batch33_source, ver23_view, batch33_view = map(Path, sys.argv[1:])

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

validation = read_rows(validation_path)
order = [str(row.get("case_id") or "") for row in validation]

def collect_manifests(root, include_rerun):
    by_id = {}
    for path in sorted(root.glob("manifest.shard*.jsonl")):
        for row in read_rows(path):
            by_id[str(row.get("case_id") or "")] = row
    if include_rerun:
        rerun = root / "manifest.rerun_failed.jsonl"
        if rerun.exists():
            for row in read_rows(rerun):
                by_id[str(row.get("case_id") or "")] = row
    return [by_id[case_id] for case_id in order]

write_rows(ver23_view / "manifest.jsonl", collect_manifests(ver23_source, True))
write_rows(batch33_view / "manifest.jsonl", collect_manifests(batch33_source, False))

merged = [path for path in batch33_source.glob("*.asr_eval.jsonl") if ".shard" not in path.name]
if len(merged) != 1:
    raise SystemExit(f"Batch33: expected one merged ASR JSONL, got {len(merged)}")
batch33_rows = {str(row.get("case_id") or row.get("sample_id") or ""): row for row in read_rows(merged[0])}
write_rows(batch33_view / "Batch33.asr_eval.jsonl", [batch33_rows[case_id] for case_id in order])
print(f"[baseline-view] prepared manifests and Batch33 ASR under {ver23_view.parent}")
PY

  "$PYTHON" "$BUILD_EVAL_SCRIPT" \
    --validation-jsonl "$VALIDATION_JSONL" \
    --output-dir "$VER23_SOURCE_DIR" \
    --manifest-jsonl "$VER23_VIEW/manifest.jsonl" \
    --run-id ver2_3 \
    --output-jsonl "$VER23_VIEW/ver2_3.generated_eval_input.jsonl" \
    --status "ok,ok_after_rerun,skipped_exists"
}

run_ver23_asr() {
  local eval_input="$VER23_VIEW/ver2_3.generated_eval_input.jsonl"
  local merged_asr="$VER23_VIEW/ver2_3.asr_eval.jsonl"
  local log_root="$RECORD_ROOT/asr_logs"
  mkdir -p "$log_root" "$RECORD_ROOT/hf_modules_cache"

  local pids=()
  local shard
  for shard in $(seq 0 $((ASR_NUM_SHARDS - 1))); do
    local out="$VER23_VIEW/ver2_3.asr_eval.shard${shard}.jsonl"
    local log="$log_root/ver2_3.asr.shard${shard}.log"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="$shard"
      export TOKENIZERS_PARALLELISM=false
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
      export OMP_NUM_THREADS=6
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export HF_MODULES_CACHE="$RECORD_ROOT/hf_modules_cache/asr_shard${shard}"
      mkdir -p "$HF_MODULES_CACHE"
      "$ASR_PYTHON" "$ASR_SCRIPT" \
        --input-jsonl "$eval_input" \
        --output-jsonl "$out" \
        --asr-backend qwen_asr \
        --qwen-asr-model "$QWEN_ASR_MODEL" \
        --qwen-asr-dtype "$QWEN_ASR_DTYPE" \
        --qwen-asr-max-batch-size "$QWEN_ASR_MAX_BATCH_SIZE" \
        --qwen-asr-max-new-tokens "$QWEN_ASR_MAX_NEW_TOKENS" \
        --device cuda:0 \
        --content-reference-mode text \
        --skip-source-asr \
        --zh-cer-threshold 0.20 \
        --en-wer-threshold 0.25 \
        --no-text-zh-cer-threshold 0.35 \
        --no-text-en-wer-threshold 0.30 \
        --max-repeat-score 0.30 \
        --num-shards "$ASR_NUM_SHARDS" \
        --shard-index "$shard" \
        --progress-every 20 \
        --overwrite
    ) > >(tee -a "$log") 2>&1 &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "ERROR: one or more Ver2.3 ASR shards failed; see $log_root" >&2
    return 1
  fi

  "$PYTHON" - "$eval_input" "$merged_asr" "$VER23_VIEW" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

eval_input, output_path, root = map(Path, sys.argv[1:])

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

expected_rows = rows(eval_input)
order = {str(row.get("case_id") or row.get("sample_id") or ""): idx for idx, row in enumerate(expected_rows)}
merged = []
for shard in range(2):
    path = root / f"ver2_3.asr_eval.shard{shard}.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing ASR shard: {path}")
    merged.extend(rows(path))
merged.sort(key=lambda row: order.get(str(row.get("case_id") or row.get("sample_id") or ""), 10**12))
ids = [str(row.get("case_id") or row.get("sample_id") or "") for row in merged]
modes = Counter(str(row.get("mode") or "") for row in merged)
if len(merged) != 320 or len(set(ids)) != 320 or set(ids) != set(order):
    raise SystemExit(f"merged Ver2.3 ASR rows/set mismatch: rows={len(merged)} unique={len(set(ids))}")
if modes != Counter({"no_text": 160, "text": 160}):
    raise SystemExit(f"merged Ver2.3 ASR modes != 160+160: {dict(modes)}")
with output_path.open("w", encoding="utf-8") as handle:
    for row in merged:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[baseline-asr] Ver2.3 PASS rows=320 modes=160+160 output={output_path}")
PY

  "$PYTHON" "$SUMMARY_SCRIPT" \
    --asr-jsonl "$merged_asr" \
    --metrics-csv "$VER23_VIEW/ver2_3.metrics.csv" \
    --summary-md "$VER23_VIEW/ver2_3.content_summary.md" \
    --summary-json "$VER23_VIEW/ver2_3.content_summary.json" \
    --run-id ver2_3 \
    --run-label "Ver2.3 historical SeedTTS-320" \
    --model-path "historical-ver2.3-output"
}

audit_run_views() {
  "$PYTHON" - "$VALIDATION_JSONL" \
    "ver2_3=$VER23_VIEW=$VER23_SOURCE_DIR" \
    "Batch33=$BATCH33_VIEW=$BATCH33_SOURCE_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

validation_path = Path(sys.argv[1])
specs = sys.argv[2:]

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

validation = rows(validation_path)
expected = {str(row.get("case_id") or "") for row in validation}
if len(specs) != 2 or {spec.split("=", 1)[0] for spec in specs} != {"ver2_3", "Batch33"}:
    raise SystemExit(f"expected exactly the ver2_3 and Batch33 run specs, got {specs}")

for spec in specs:
    name, view_raw, source_raw = spec.split("=", 2)
    view = Path(view_raw)
    source = Path(source_raw).resolve()
    manifests = rows(view / "manifest.jsonl")
    asr_paths = [path for path in view.glob("*.asr_eval.jsonl") if ".shard" not in path.name]
    if len(asr_paths) != 1:
        raise SystemExit(f"{name}: expected one merged ASR view, got {len(asr_paths)}")
    asr = rows(asr_paths[0])
    for kind, payload in (("manifest", manifests), ("ASR", asr)):
        ids = [str(row.get("case_id") or row.get("sample_id") or "") for row in payload]
        modes = Counter(str(row.get("mode") or "") for row in payload)
        if len(payload) != 320 or len(set(ids)) != 320 or set(ids) != expected:
            raise SystemExit(f"{name}: {kind} rows/set mismatch: rows={len(payload)} unique={len(set(ids))}")
        if modes != Counter({"no_text": 160, "text": 160}):
            raise SystemExit(f"{name}: {kind} modes != 160+160: {dict(modes)}")
    for row in asr:
        case_id = str(row.get("case_id") or row.get("sample_id") or "")
        target = Path(str(row.get("target_audio") or ""))
        if not target.is_file() or target.stem != case_id or target.resolve().parent != source:
            raise SystemExit(f"{name}/{case_id}: target is not the audited source WAV: {target}")
    print(f"[baseline-view-audit] {name} PASS manifest=320 ASR=320 modes=160+160 source={source}")
PY
}

KEEPALIVE_PIDS=()

start_gpu_keepalive() {
  KEEPALIVE_PIDS=()
  if [ "$KEEPALIVE_UNUSED_GPUS" != "1" ]; then
    echo "[baseline-keepalive] disabled"
    return 0
  fi
  mkdir -p "$RECORD_ROOT/keepalive_logs"
  local gpu
  for gpu in $(seq 2 7); do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" - "$gpu" \
      >"$RECORD_ROOT/keepalive_logs/gpu${gpu}.log" 2>&1 <<'PY' &
import signal
import sys
import time

import torch

physical_gpu = sys.argv[1]
running = True

def stop(*_args):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
device = torch.device("cuda:0")
a = torch.randn((4096, 4096), device=device, dtype=torch.float16)
b = torch.randn((4096, 4096), device=device, dtype=torch.float16)
c = torch.empty_like(a)
print(f"keepalive started physical_gpu={physical_gpu}", flush=True)
with torch.inference_mode():
    while running:
        for _ in range(4):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        time.sleep(2.0)
print(f"keepalive stopped physical_gpu={physical_gpu}", flush=True)
PY
    KEEPALIVE_PIDS+=("$!")
  done
  sleep 1
  local pid
  for pid in "${KEEPALIVE_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: a GPU keepalive process failed during startup" >&2
      stop_gpu_keepalive
      return 1
    fi
  done
  echo "[baseline-keepalive] started GPUs 2-7 pids=${KEEPALIVE_PIDS[*]}"
}

stop_gpu_keepalive() {
  if [ "${#KEEPALIVE_PIDS[@]}" -eq 0 ]; then
    return 0
  fi
  kill "${KEEPALIVE_PIDS[@]}" 2>/dev/null || true
  local pid
  for pid in "${KEEPALIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  KEEPALIVE_PIDS=()
  echo "[baseline-keepalive] stopped"
}

run_dual_encoder_and_diagnostics() {
  mkdir -p "$RECORD_ROOT/hf_modules_cache/scorers" "$DIAGNOSTICS_ROOT"
  CUDA_VISIBLE_DEVICES=0,1 \
  TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_MODULES_CACHE="$RECORD_ROOT/hf_modules_cache/scorers" \
  OMP_NUM_THREADS=8 \
    "$PYTHON" "$DUAL_ENCODER_SCRIPT" \
      --validation-jsonl "$VALIDATION_JSONL" \
      --run "ver2_3=$VER23_VIEW" \
      --run "Batch33=$BATCH33_VIEW" \
      --output-csv "$DUAL_CASES" \
      --summary-json "$DUAL_SUMMARY_JSON" \
      --summary-md "$DUAL_SUMMARY_MD" \
      --speaker-device cuda:0 \
      --speaker-sim-root "$SPEAKER_SIM_ROOT" \
      --extra-speaker-encoder speechbrain_ecapa \
      --extra-speaker-device cuda:1 \
      --speechbrain-ecapa-model-source "$SPEECHBRAIN_ECAPA_MODEL_SOURCE"

  "$PYTHON" "$DIAGNOSTICS_SCRIPT" \
    --validation-jsonl "$VALIDATION_JSONL" \
    --sim-cases-csv "$DUAL_CASES" \
    --run "ver2_3=$VER23_VIEW" \
    --run "Batch33=$BATCH33_VIEW" \
    --output-dir "$DIAGNOSTICS_ROOT" \
    --prefix baseline_dual_encoder

}

audit_final_outputs() {
  "$PYTHON" - "$DUAL_CASES" "$DUAL_SUMMARY_JSON" "$DIAGNOSTICS_ROOT/baseline_dual_encoder.summary.json" <<'PY'
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

cases_path, summary_path, diagnostics_path = map(Path, sys.argv[1:])
with cases_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 640:
    raise SystemExit(f"dual_encoder_cases rows != 640: {len(rows)}")
counts = Counter((str(row.get("run") or ""), str(row.get("mode") or "")) for row in rows)
expected = Counter({("ver2_3", "no_text"): 160, ("ver2_3", "text"): 160,
                    ("Batch33", "no_text"): 160, ("Batch33", "text"): 160})
if counts != expected:
    raise SystemExit(f"dual_encoder_cases run/mode counts mismatch: {dict(counts)}")
if len({(row.get("run"), row.get("case_id")) for row in rows}) != 640:
    raise SystemExit("dual_encoder_cases run/case_id pairs are not unique")
for row in rows:
    for field in ("sim_gen_ref", "sim_gen_source", "ecapa_sim_gen_ref", "ecapa_sim_gen_source"):
        try:
            value = float(row.get(field, ""))
        except Exception as exc:
            raise SystemExit(f"{row.get('run')}/{row.get('case_id')}: invalid {field}") from exc
        if not math.isfinite(value):
            raise SystemExit(f"{row.get('run')}/{row.get('case_id')}: non-finite {field}={value}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
if set(summary.get("runs", {})) != {"ver2_3", "Batch33"}:
    raise SystemExit(f"dual summary run set mismatch: {set(summary.get('runs', {}))}")
for run in ("ver2_3", "Batch33"):
    if int(summary["runs"][run].get("n") or 0) != 320:
        raise SystemExit(f"{run}: dual summary n != 320")

diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
if diagnostics.get("runs") != ["ver2_3", "Batch33"]:
    raise SystemExit(f"diagnostics run order/set mismatch: {diagnostics.get('runs')}")
print("[baseline-final-audit] PASS runs=2 cases=640 each_run=320 each_mode=160 all_dual_scores_finite=1")
PY
}

run_entrypoint() {
  mkdir -p "$RECORD_ROOT" "$EVAL_ROOT"
  exec > >(tee -a "$RECORD_ROOT/run.log") 2>&1
  echo "[baseline-dual] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "[baseline-dual] code_root=$CODE_ROOT"
  echo "[baseline-dual] record_root=$RECORD_ROOT"
  echo "[baseline-dual] eval_root=$EVAL_ROOT"
  validate_inputs

  local gpu_count
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d ' ')
  if [ "$gpu_count" != "8" ]; then
    echo "ERROR: expected exactly 8 GPUs in the MTTS whole node; got $gpu_count" >&2
    return 1
  fi
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Evq 'H200'; then
    echo "ERROR: one or more allocated GPUs are not H200" >&2
    return 1
  fi
  nvidia-smi || true

  start_gpu_keepalive
  trap stop_gpu_keepalive EXIT
  trap 'trap - EXIT INT TERM; stop_gpu_keepalive; exit 130' INT
  trap 'trap - EXIT INT TERM; stop_gpu_keepalive; exit 143' TERM
  prepare_run_views
  run_ver23_asr
  audit_run_views
  run_dual_encoder_and_diagnostics
  audit_final_outputs
  stop_gpu_keepalive
  trap - EXIT INT TERM
  echo "[baseline-dual] complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[baseline-dual] cases=$DUAL_CASES"
  echo "[baseline-dual] summary=$DUAL_SUMMARY_MD"
  echo "[baseline-dual] diagnostics=$DIAGNOSTICS_ROOT/baseline_dual_encoder.summary.md"
}

write_frozen_runner() {
  cp "$SELF_PATH" "$FROZEN_DRIVER"
  chmod +x "$FROZEN_DRIVER"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'export SEEDTTS_BASELINE_DUAL_ENCODER_ENTRYPOINT=%q\n' 1
    printf 'export RUN_TAG=%q\n' "$RUN_TAG"
    printf 'export FORCE=%q\n' "$FORCE"
    printf 'export KEEPALIVE_UNUSED_GPUS=%q\n' "$KEEPALIVE_UNUSED_GPUS"
    printf 'export ASR_NUM_SHARDS=%q\n' "$ASR_NUM_SHARDS"
    printf 'export PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
    printf 'export CODE_ROOT=%q\n' "$CODE_ROOT"
    printf 'export RECORD_ROOT=%q\n' "$RECORD_ROOT"
    printf 'export EVAL_ROOT=%q\n' "$EVAL_ROOT"
    printf 'export PYTHON=%q\n' "$PYTHON"
    printf 'export ASR_PYTHON=%q\n' "$ASR_PYTHON"
    printf 'export VALIDATION_JSONL=%q\n' "$VALIDATION_JSONL"
    printf 'export VER23_SOURCE_DIR=%q\n' "$VER23_SOURCE_DIR"
    printf 'export BATCH33_SOURCE_DIR=%q\n' "$BATCH33_SOURCE_DIR"
    printf 'export SPEAKER_SIM_ROOT=%q\n' "$SPEAKER_SIM_ROOT"
    printf 'export SPEECHBRAIN_ECAPA_MODEL_SOURCE=%q\n' "$SPEECHBRAIN_ECAPA_MODEL_SOURCE"
    printf 'export QWEN_ASR_MODEL=%q\n' "$QWEN_ASR_MODEL"
    printf 'export QWEN_ASR_DTYPE=%q\n' "$QWEN_ASR_DTYPE"
    printf 'export QWEN_ASR_MAX_BATCH_SIZE=%q\n' "$QWEN_ASR_MAX_BATCH_SIZE"
    printf 'export QWEN_ASR_MAX_NEW_TOKENS=%q\n' "$QWEN_ASR_MAX_NEW_TOKENS"
    printf 'exec bash %q\n' "$FROZEN_DRIVER"
  } > "$RUNNER"
  chmod +x "$RUNNER"
  bash -n "$FROZEN_DRIVER"
  bash -n "$RUNNER"
  {
    sha256sum "$FROZEN_DRIVER" "$RUNNER" "$RESOLVED_RUNS"
    sha256sum \
      "$ASR_SCRIPT" \
      "$BUILD_EVAL_SCRIPT" \
      "$SUMMARY_SCRIPT" \
      "$DUAL_ENCODER_SCRIPT" \
      "$DIAGNOSTICS_SCRIPT"
  } > "$RECORD_ROOT/sha256sums.txt"
}

if [ "$ENTRYPOINT" = "1" ]; then
  run_entrypoint
  exit 0
fi

validate_inputs
mkdir -p "$RECORD_ROOT" "$EVAL_ROOT" "$QZCLI_HOME"
if [ "$FORCE" != "1" ] && {
  [ -s "$RECORD_ROOT/submitted_jobs.tsv" ] ||
  [ -s "$DUAL_SUMMARY_JSON" ];
}; then
  echo "ERROR: existing submission or completed summary; use FORCE=1 only for an intentional rerun" >&2
  exit 1
fi
write_resolved_runs
write_frozen_runner

COMMAND="bash $RUNNER"
SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"

echo "QZ submit: SeedTTS baseline dual-encoder scoring"
echo "  JOB_NAME=$JOB_NAME"
echo "  RUNS=ver2_3,Batch33"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC"
echo "  CODE_ROOT=$CODE_ROOT"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  EVAL_ROOT=$EVAL_ROOT"
echo "  ASR_SHARDS=2 (matches 004072; Ver2.3 compatibility pass)"
echo "  SCORERS=cuda:0 WavLM-Large+ECAPA, cuda:1 SpeechBrain-ECAPA"
echo "  DRY_RUN=$DRY_RUN"
echo "  COMMAND=$COMMAND"

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
  echo "[baseline-dual] dry-run passed; no job submitted"
  exit 0
fi

job_id=$(printf '%s\n' "$output" | grep -Eo 'job-[0-9a-fA-F-]{36}' | tail -n 1 || true)
if [ -z "$job_id" ]; then
  echo "ERROR: QZ returned success but no job ID could be parsed; refusing to write a submission record" >&2
  exit 1
fi
{
  printf 'job_name\tjob_id\tcompute_group\tspec\tcode_root\trecord_root\teval_root\trunner\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$job_id" "$COMPUTE_GROUP" "$SPEC" "$CODE_ROOT" "$RECORD_ROOT" "$EVAL_ROOT" "$RUNNER"
} > "$RECORD_ROOT/submitted_jobs.tsv"
echo "[baseline-dual] submitted job_id=$job_id record=$RECORD_ROOT/submitted_jobs.tsv"
