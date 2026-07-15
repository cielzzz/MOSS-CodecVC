#!/usr/bin/env bash
# Submit the *only* allowed large-scale v2 data-cleaning arm: four-edge balanced.
#
# This submitter does not start training, Batch-41/42 work, or cross-episode U2.
# It uses one MTTS-3-2-0715 node with all eight H200 GPUs as actual shards.
# Default is dry generation.  A live submission needs both DRY_RUN=0 and the
# explicit ALLOW_V2_FOUR_EDGE_BALANCED_SUBMIT=1 guard.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PAIR_CONSTRUCTION_ROOT="${PAIR_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction}"
QZCLI="${QZCLI:-$PAIR_CONSTRUCTION_ROOT/scripts/qzcli_with_deps.sh}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"

WORKSPACE="${WORKSPACE:-CI-情境智能}"
PROJECT="${PROJECT:-CI-情境智能}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009" # one node x 8 H200
SPEC="${SPEC:-$ALLOWED_SPEC}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"
GPU_TYPE="${GPU_TYPE:-NVIDIA_H200_SXM_141G}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/trainset/ver2_9_prepared_v2_four_edge_balanced_20260715}"
NUM_SHARDS="${NUM_SHARDS:-8}"
WAVLM_BATCH_SIZE="${WAVLM_BATCH_SIZE:-8}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-1}"
BATCH_ID="${BATCH_ID:-v2_four_edge_balanced_$(date -u +%Y%m%dT%H%M%SZ)}"
JOB_NAME="${JOB_NAME:-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_v2_four_edge_balanced.sh"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
    die "only MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP) is allowed; got $COMPUTE_GROUP"
fi
if [ "$SPEC" != "$ALLOWED_SPEC" ]; then
    die "only registered 1x8-H200 spec $ALLOWED_SPEC is allowed; got $SPEC"
fi
if [ "$INSTANCES" != "1" ] || [ "$NUM_SHARDS" != "8" ]; then
    die "production cleaning is hard-locked to one instance and eight real shards"
fi
if [ "$OUTPUT_ROOT" != "$ROOT/trainset/ver2_9_prepared_v2_four_edge_balanced_20260715" ] && [ "${ALLOW_NONCANONICAL_FULL_OUTPUT:-0}" != "1" ]; then
    die "full output root must be canonical: $ROOT/trainset/ver2_9_prepared_v2_four_edge_balanced_20260715"
fi
if [ "$DRY_RUN" != "1" ] && [ "${ALLOW_V2_FOUR_EDGE_BALANCED_SUBMIT:-0}" != "1" ]; then
    die "live submit is guarded; set DRY_RUN=0 and ALLOW_V2_FOUR_EDGE_BALANCED_SUBMIT=1"
fi
if [ ! -x "$QZCLI" ]; then
    die "qzcli wrapper is not executable: $QZCLI"
fi
if [ ! -x "$PY" ]; then
    die "Python is not executable: $PY"
fi
for file in \
    "$ROOT/scripts/004134_clean_v2_four_edge_balanced.py" \
    "$ROOT/scripts/004135_run_v2_four_edge_balanced.sh" \
    "$ROOT/trainset/ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708/no_text.v2.train.jsonl" \
    "/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/VC_train/v2_real_target_no_text_300k_zh_en_balanced_20260707_seedvc_triples/source_seedvc_jobs.jsonl" \
    "/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/VC_train/v2_real_target_no_text_300k_zh_en_balanced_20260707_seedvc_triples/source_seedvc_results.jsonl"; do
    [ -s "$file" ] || die "required input/code is missing or empty: $file"
done
if [ -e "$OUTPUT_ROOT/COMPLETED.json" ] && [ "$FORCE" != "1" ]; then
    die "canonical output already has COMPLETED.json; refuse duplicate cleaning: $OUTPUT_ROOT"
fi

mkdir -p "$QZ_RECORD_ROOT/record_snapshot/scripts" "$QZ_RECORD_ROOT/record_snapshot/moss_codecvc/models"
cp -p "$ROOT/scripts/004134_clean_v2_four_edge_balanced.py" "$QZ_RECORD_ROOT/record_snapshot/scripts/"
cp -p "$ROOT/scripts/004135_run_v2_four_edge_balanced.sh" "$QZ_RECORD_ROOT/record_snapshot/scripts/"
cp -p "$ROOT/scripts/004136_submit_v2_four_edge_balanced_qz.sh" "$QZ_RECORD_ROOT/record_snapshot/scripts/" 2>/dev/null || true
cp -p "$ROOT/moss_codecvc/models/speaker_encoder.py" "$QZ_RECORD_ROOT/record_snapshot/moss_codecvc/models/"
cp -p "$ROOT/moss_codecvc/third_party.py" "$QZ_RECORD_ROOT/record_snapshot/moss_codecvc/"
sha256sum \
    "$QZ_RECORD_ROOT/record_snapshot/scripts/004134_clean_v2_four_edge_balanced.py" \
    "$QZ_RECORD_ROOT/record_snapshot/scripts/004135_run_v2_four_edge_balanced.sh" \
    "$QZ_RECORD_ROOT/record_snapshot/moss_codecvc/models/speaker_encoder.py" \
    "$QZ_RECORD_ROOT/record_snapshot/moss_codecvc/third_party.py" \
    >"$QZ_RECORD_ROOT/record_snapshot/SHA256SUMS"

cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

ROOT="$ROOT"
PY="$PY"
DOWNLOAD_ROOT="$DOWNLOAD_ROOT"
OUTPUT_ROOT="$OUTPUT_ROOT"
NUM_SHARDS=8
WAVLM_BATCH_SIZE="$WAVLM_BATCH_SIZE"
GPU_KEEPALIVE="$GPU_KEEPALIVE"
FORCE="$FORCE"
EXPECTED_GPU_TYPE="$GPU_TYPE"

export HF_HOME="\$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="\$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="\$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="\$DOWNLOAD_ROOT/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"

cd "\$ROOT"
mkdir -p "\$OUTPUT_ROOT/logs"
echo "[v2-four-edge-qz] started=\$(date -u +%Y-%m-%dT%H:%M:%SZ) host=\$(hostname)"
echo "[v2-four-edge-qz] output=\$OUTPUT_ROOT shards=\$NUM_SHARDS hf_cache=\$HUGGINGFACE_HUB_CACHE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"\$PY" - <<'PY'
from __future__ import annotations
import os
from pathlib import Path
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
from transformers.utils import logging as transformers_logging

expected = os.environ.get("EXPECTED_GPU_TYPE", "H200")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"require exactly 8 visible CUDA GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if not all("H200" in name.upper() for name in names):
    raise SystemExit(f"require eight H200 GPUs, got {names}")
model = "microsoft/wavlm-base-plus-sv"
transformers_logging.disable_progress_bar()
AutoFeatureExtractor.from_pretrained(model, local_files_only=True)
AutoModelForAudioXVector.from_pretrained(model, local_files_only=True)
ecapa = Path("/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb")
if not (ecapa / "embedding_model.ckpt").is_file():
    raise SystemExit(f"missing ECAPA checkpoint: {ecapa}")
print({"cuda_gpus": names, "wavlm_local_cache": "ok", "ecapa": str(ecapa)}, flush=True)
PY

MODE=full \\
ROOT="\$ROOT" \\
PY="\$PY" \\
OUTPUT_ROOT="\$OUTPUT_ROOT" \\
NUM_SHARDS="\$NUM_SHARDS" \\
GPUS="0 1 2 3 4 5 6 7" \\
DEVICE=cuda \\
WAVLM_BATCH_SIZE="\$WAVLM_BATCH_SIZE" \\
GPU_KEEPALIVE="\$GPU_KEEPALIVE" \\
FORCE="\$FORCE" \\
  bash "\$ROOT/scripts/004135_run_v2_four_edge_balanced.sh"
EOF
chmod +x "$RUNNER"

"$PY" - <<PY >"$QZ_RECORD_ROOT/preflight_local.json"
from __future__ import annotations
import hashlib
import json
from pathlib import Path

root = Path("$ROOT")
paths = {
    "input": root / "trainset/ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708/no_text.v2.train.jsonl",
    "cleaner": root / "scripts/004134_clean_v2_four_edge_balanced.py",
    "runner": root / "scripts/004135_run_v2_four_edge_balanced.sh",
}
payload = {}
for name, path in paths.items():
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if name != "input" else None
    payload[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
payload["contract"] = {
    "workspace": "$WORKSPACE",
    "project": "$PROJECT",
    "compute_group": "$COMPUTE_GROUP",
    "compute_group_name": "MTTS-3-2-0715",
    "spec": "$SPEC",
    "instances": int("$INSTANCES"),
    "shards": int("$NUM_SHARDS"),
    "gpu_type": "$GPU_TYPE",
    "output_root": "$OUTPUT_ROOT",
    "cross_episode_u2_changed": False,
    "batch41_42_resumed": False,
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "[v2-four-edge-submit] job_name=$JOB_NAME"
echo "[v2-four-edge-submit] resource=CI-情境智能 / MTTS-3-2-0715 / 1x8 H200"
echo "[v2-four-edge-submit] output=$OUTPUT_ROOT"
echo "[v2-four-edge-submit] runner=$RUNNER"
echo "[v2-four-edge-submit] record=$QZ_RECORD_ROOT"

if [ "$DRY_RUN" = "1" ]; then
    printf 'job_name\tcompute_group\tspec\tinstances\tshards\tgpu_type\toutput_root\trunner\n' >"$QZ_RECORD_ROOT/submission_plan.tsv"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$JOB_NAME" "$COMPUTE_GROUP" "$SPEC" "$INSTANCES" "$NUM_SHARDS" "$GPU_TYPE" "$OUTPUT_ROOT" "$RUNNER" \
        >>"$QZ_RECORD_ROOT/submission_plan.tsv"
    echo "[v2-four-edge-submit] dry-run only; no QZ job submitted"
    exit 0
fi

LOCK="$QZ_RECORD_ROOT/.live_submission.lock"
if ! mkdir "$LOCK"; then
    die "live submission lock exists: $LOCK"
fi
cleanup_lock() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup_lock EXIT

SUBMIT_OUTPUT="$QZ_RECORD_ROOT/submit_output.txt"
set +e
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    QZCLI_GPU_TYPE_OVERRIDE="$GPU_TYPE" \
    "$QZCLI" create-job \
    --name "$JOB_NAME" \
    --workspace "$WORKSPACE" \
    --project "$PROJECT" \
    --compute-group "$COMPUTE_GROUP" \
    --spec "$SPEC" \
    --framework "$FRAMEWORK" \
    --instances "$INSTANCES" \
    --shm "$SHM_GI" \
    --priority "$PRIORITY" \
    --image "$IMAGE" \
    --image-type "$IMAGE_TYPE" \
    --command "bash $RUNNER" >"$SUBMIT_OUTPUT" 2>&1
STATUS=$?
set -e
cat "$SUBMIT_OUTPUT"

if [ "$STATUS" -ne 0 ] && grep -q 'Cookie 已过期或无效' "$SUBMIT_OUTPUT"; then
    echo "[v2-four-edge-submit] QZ cookie expired; refreshing via the approved local wrapper"
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy "$QZCLI" login
    set +e
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
        QZCLI_GPU_TYPE_OVERRIDE="$GPU_TYPE" \
        "$QZCLI" create-job \
        --name "$JOB_NAME" \
        --workspace "$WORKSPACE" \
        --project "$PROJECT" \
        --compute-group "$COMPUTE_GROUP" \
        --spec "$SPEC" \
        --framework "$FRAMEWORK" \
        --instances "$INSTANCES" \
        --shm "$SHM_GI" \
        --priority "$PRIORITY" \
        --image "$IMAGE" \
        --image-type "$IMAGE_TYPE" \
        --command "bash $RUNNER" >"$SUBMIT_OUTPUT" 2>&1
    STATUS=$?
    set -e
    cat "$SUBMIT_OUTPUT"
fi
if [ "$STATUS" -ne 0 ]; then
    die "QZ submission failed; see $SUBMIT_OUTPUT"
fi

JOB_ID="$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$SUBMIT_OUTPUT" | tail -n 1 || true)"
if [ -z "$JOB_ID" ]; then
    JOB_UUID="$(grep -E '任务ID|job_id|Job ID' "$SUBMIT_OUTPUT" | grep -Eo '[0-9a-fA-F-]{36}' | tail -n 1 || true)"
    [ -z "$JOB_UUID" ] || JOB_ID="job-$JOB_UUID"
fi
printf 'job_name\tjob_id\tcompute_group\tspec\tinstances\tshards\tgpu_type\toutput_root\trunner\n' >"$QZ_RECORD_ROOT/submitted_jobs.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$JOB_ID" "$COMPUTE_GROUP" "$SPEC" "$INSTANCES" "$NUM_SHARDS" "$GPU_TYPE" "$OUTPUT_ROOT" "$RUNNER" \
    >>"$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "[v2-four-edge-submit] submitted job_id=${JOB_ID:-unparsed}"
