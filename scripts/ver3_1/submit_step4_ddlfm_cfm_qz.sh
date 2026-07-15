#!/usr/bin/env bash
# Batch-45 Step 4: ver3.1 DDLFM CFM 3k probe.
#
# The script is deliberately dry-run by default.  A live submission requires
# the explicit ALLOW_CODECVC_VER3_1_STEP4_SUBMIT=1 guard and is hard-bound to
# the only approved MTTS-3-2-0715 8xH200 resource contract.
set -Eeuo pipefail

ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
TORCHRUN="${TORCHRUN:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/torchrun}"
WORKSPACE="${WORKSPACE:-CI-情境智能}"
PROJECT="${PROJECT:-CI-情境智能}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
INSTANCES="${INSTANCES:-1}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"
GPU_TYPE="${GPU_TYPE:-NVIDIA_H200_SXM_141G}"
DRY_RUN="${DRY_RUN:-1}"

STEPS="${STEPS:-3000}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOG_STEPS="${LOG_STEPS:-20}"
MAX_FRAMES="${MAX_FRAMES:-256}"
PRECISION="${PRECISION:-bf16}"

BATCH_ID="${BATCH_ID:-codecVC-ver3-1-step4-ddlfm-cfm-3k-probe-20260715}"
JOB_NAME="${JOB_NAME:-$BATCH_ID}"
RECORD_ROOT="${RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
SNAPSHOT_ROOT="$RECORD_ROOT/record_snapshot"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/ver3_1_ddlfm_cfm_probe_20260715}"

INDEX="$ROOT/prepared/ddlfm_v1_index.jsonl"
INDEX_SUMMARY="$ROOT/prepared/ddlfm_v1_index.summary.json"
INDEX_SHA256="d114d222259fbf66cfb9f001929a216bd4cb7f3e2ac46a4c1dff6c56053c5804"
INDEX_ROWS="342839"
ZQ_SUMMARY="$ROOT/prepared/zq_targets_v1/COMPLETED.json"
SEMANTIC_SUMMARY="$ROOT/prepared/semantic_v1_v3_1_step3_no_text_20260715/REPORT.md"
SEMANTIC_COMPLETION="$ROOT/prepared/semantic_v1_v3_1_step3_no_text_20260715/COMPLETED.json"
TRAIN_SCRIPT="$ROOT/scripts/ver3_1/train_ddlfm_cfm.py"
DECODER_MODULE="$ROOT/moss_codecvc/models/ddlfm_decoder.py"
SOURCE_MEMORY_MODULE="$ROOT/moss_codecvc/models/source_semantic_memory.py"
DECODE_LATENTS_MODULE="$ROOT/moss_codecvc/audio/decode_latents.py"

die() { echo "ERROR: $*" >&2; exit 1; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

[ "$QZCLI" = "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh" ] || die "only approved qzcli wrapper allowed"
[ -x "$QZCLI" ] || die "missing qzcli: $QZCLI"
[ -x "$PY" ] || die "missing Python: $PY"
[ -x "$TORCHRUN" ] || die "missing torchrun: $TORCHRUN"
[ "$COMPUTE_GROUP" = "$ALLOWED_COMPUTE_GROUP" ] || die "only MTTS-3-2-0715 is allowed"
[ "$SPEC" = "$ALLOWED_SPEC" ] || die "only registered 8xH200 spec is allowed"
[ "$GPU_TYPE" = "NVIDIA_H200_SXM_141G" ] || die "only NVIDIA_H200_SXM_141G is allowed"
[ "$INSTANCES" = "1" ] || die "Step 4 requires one instance"
[[ "$JOB_NAME" == codecVC-* ]] || die "job name must start with codecVC-"
[[ "$BATCH_ID" == codecVC-* ]] || die "batch id must start with codecVC-"
[[ "$JOB_NAME" =~ ^codecVC-[A-Za-z0-9_.-]+$ ]] || die "unsafe job name"
[[ "$BATCH_ID" =~ ^codecVC-[A-Za-z0-9_.-]+$ ]] || die "unsafe batch id"
[ -s "$TRAIN_SCRIPT" ] || die "missing CFM train script"
[ -s "$DECODER_MODULE" ] || die "missing DDLFM decoder module"
[ -s "$SOURCE_MEMORY_MODULE" ] || die "missing source semantic memory module"
[ -s "$DECODE_LATENTS_MODULE" ] || die "missing decode_latents module"
[ -s "$INDEX" ] || die "missing complete DDLFM index"
[ -s "$INDEX_SUMMARY" ] || die "missing DDLFM index summary"
[ -s "$ZQ_SUMMARY" ] || die "missing zq completion marker"
[ -s "$SEMANTIC_SUMMARY" ] || die "missing no_text semantic report"
[ -s "$SEMANTIC_COMPLETION" ] || die "missing no_text semantic completion marker"
[ "$RECORD_ROOT" = "$ROOT/trainset/qz_jobs/$BATCH_ID" ] || die "record root must stay under project trainset/qz_jobs"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || die "STEPS must be a positive integer"
[[ "$PER_DEVICE_BATCH" =~ ^[1-9][0-9]*$ ]] || die "PER_DEVICE_BATCH must be a positive integer"
[[ "$GRAD_ACCUM_STEPS" =~ ^[1-9][0-9]*$ ]] || die "GRAD_ACCUM_STEPS must be a positive integer"
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || die "NUM_WORKERS must be a non-negative integer"
[[ "$WARMUP_STEPS" =~ ^[0-9]+$ ]] || die "WARMUP_STEPS must be a non-negative integer"
[[ "$SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || die "SAVE_STEPS must be a positive integer"
[[ "$LOG_STEPS" =~ ^[1-9][0-9]*$ ]] || die "LOG_STEPS must be a positive integer"
[[ "$MAX_FRAMES" =~ ^[1-9][0-9]*$ ]] || die "MAX_FRAMES must be a positive integer"
[ "$(wc -l < "$INDEX" | tr -d ' ')" = "$INDEX_ROWS" ] || die "DDLFM index row count changed"
[ "$(sha256_file "$INDEX")" = "$INDEX_SHA256" ] || die "DDLFM index SHA256 changed"
[ "$("$PY" - "$ZQ_SUMMARY" "$SEMANTIC_COMPLETION" "$INDEX_SUMMARY" <<'PY'
import json, sys
zq = json.loads(open(sys.argv[1], encoding="utf-8").read())
sem = json.loads(open(sys.argv[2], encoding="utf-8").read())
idx = json.loads(open(sys.argv[3], encoding="utf-8").read())
checks = [
    zq.get("status") == "completed",
    int(zq.get("errors", -1)) == 0,
    int(zq.get("total_utterances", -1)) == 342839,
    int(zq.get("latent_dim", -1)) == 768,
    float(zq.get("frame_rate_hz", -1)) == 12.5,
    sem.get("status") == "completed",
    int(sem.get("rows", -1)) == 310420,
    int(sem.get("semantic_dim", -1)) == 512,
    float(sem.get("rate_hz", -1)) == 12.5,
    int(sem.get("shards", -1)) == 8,
    idx.get("status") == "completed",
    int(idx.get("rows", -1)) == 342839,
    int(idx.get("missing_no_text_semantic", -1)) == 0,
]
if not all(checks):
    raise SystemExit(1)
print("ok")
PY
)" = "ok" ] || die "zq/semantic/index completion contracts failed"
[ ! -e "$OUTPUT_ROOT/COMPLETED.json" ] || die "CFM output already complete: $OUTPUT_ROOT"
if [ -d "$OUTPUT_ROOT" ] && [ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ] && [ "${ALLOW_NONEMPTY_OUTPUT:-0}" != "1" ]; then
  die "CFM output root is non-empty; choose a fresh OUTPUT_ROOT or set ALLOW_NONEMPTY_OUTPUT=1"
fi
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1" ;; esac
if [ "$DRY_RUN" = "0" ] && [ "${ALLOW_CODECVC_VER3_1_STEP4_SUBMIT:-0}" != "1" ]; then
  die "live submission guarded; set DRY_RUN=0 ALLOW_CODECVC_VER3_1_STEP4_SUBMIT=1"
fi

mkdir -p "$SNAPSHOT_ROOT/scripts/ver3_1" "$SNAPSHOT_ROOT/moss_codecvc" "$SNAPSHOT_ROOT/prepared"
cp -p "$TRAIN_SCRIPT" "$SNAPSHOT_ROOT/scripts/ver3_1/"
cp -a "$ROOT/moss_codecvc/." "$SNAPSHOT_ROOT/moss_codecvc/"
cp -p "$INDEX" "$SNAPSHOT_ROOT/prepared/ddlfm_v1_index.jsonl"
cp -p "$INDEX_SUMMARY" "$SNAPSHOT_ROOT/prepared/ddlfm_v1_index.summary.json"
SOURCE_GIT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
find "$SNAPSHOT_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum >"$SNAPSHOT_ROOT/SHA256SUMS"
printf '%s\n' "$SOURCE_GIT_SHA" >"$SNAPSHOT_ROOT/SOURCE_GIT_SHA"

RUNNER="$RECORD_ROOT/run_step4_ddlfm_cfm_qz.sh"
cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
export ROOT="$ROOT"
export PY="$PY"
export TORCHRUN="$TORCHRUN"
export INDEX="$SNAPSHOT_ROOT/prepared/ddlfm_v1_index.jsonl"
export INDEX_SUMMARY="$SNAPSHOT_ROOT/prepared/ddlfm_v1_index.summary.json"
export OUTPUT_ROOT="$OUTPUT_ROOT"
export SNAPSHOT_ROOT="$SNAPSHOT_ROOT"
export HF_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface"
export TRANSFORMERS_CACHE="\$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="\$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="\$SNAPSHOT_ROOT:\$ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
mkdir -p "\$OUTPUT_ROOT"
"\$PY" - "\$INDEX" "\$INDEX_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
index = Path(sys.argv[1])
summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if summary.get("status") != "completed" or int(summary.get("rows", -1)) != $INDEX_ROWS:
    raise SystemExit(f"invalid DDLFM index summary: {summary}")
if sum(1 for _ in index.open("r", encoding="utf-8")) != $INDEX_ROWS:
    raise SystemExit("DDLFM index row count mismatch at runtime")
print("[step4] runtime index preflight passed", flush=True)
PY
"\$TORCHRUN" --standalone --nproc_per_node=8 \\
  "\$SNAPSHOT_ROOT/scripts/ver3_1/train_ddlfm_cfm.py" \\
  --index "\$INDEX" \\
  --output-dir "\$OUTPUT_ROOT" \\
  --steps "$STEPS" \\
  --batch-size "$PER_DEVICE_BATCH" \\
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \\
  --num-workers "$NUM_WORKERS" \\
  --lr 1e-4 \\
  --warmup-steps "$WARMUP_STEPS" \\
  --save-every "$SAVE_STEPS" \\
  --log-every "$LOG_STEPS" \\
  --max-frames "$MAX_FRAMES" \\
  --precision "$PRECISION" \\
  --device cuda
"\$PY" - "\$OUTPUT_ROOT" <<'PY'
import json, sys, time
from pathlib import Path
root = Path(sys.argv[1])
last = root / "last.pt"
if not last.is_file():
    raise SystemExit("CFM training returned without last.pt")
payload = {
    "schema": "ver3_1_ddlfm_cfm_train_summary_v1",
    "status": "completed",
    "completed_at_unix": time.time(),
    "steps": $STEPS,
    "output_root": str(root),
    "last_checkpoint": str(last),
}
(root / "COMPLETED.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False), flush=True)
PY
EOF
chmod +x "$RUNNER"

cat >"$RECORD_ROOT/preflight.json" <<EOF
{
  "schema": "ver3_1_step4_ddlfm_cfm_submit_preflight_v1",
  "job_name": "$JOB_NAME",
  "source_git_sha": "$SOURCE_GIT_SHA",
  "dry_run": $DRY_RUN,
  "resource": {"compute_group": "$COMPUTE_GROUP", "compute_group_name": "MTTS-3-2-0715", "spec": "$SPEC", "instances": 1, "gpus": 8, "gpu_type": "$GPU_TYPE"},
  "input": {"index": "$INDEX", "index_sha256": "$INDEX_SHA256", "rows": $INDEX_ROWS, "zq_summary": "$ZQ_SUMMARY", "semantic_summary": "$SEMANTIC_COMPLETION", "semantic_report": "$SEMANTIC_SUMMARY"},
  "contract": {"target": "zq dequantized decoder latent [T,768] @ 12.5Hz", "no_text_semantic": "WavLM-base-plus layer9 adapter [T,512]", "text_semantic": "content_token_ids -> SourceTokenMemoryEncoder [L,512]", "speaker": "existing ECAPA 192-D sidecar", "steps": $STEPS, "per_device_batch": $PER_DEVICE_BATCH, "grad_accum_steps": $GRAD_ACCUM_STEPS, "global_batch": $((PER_DEVICE_BATCH * GRAD_ACCUM_STEPS * 8)), "lr": 0.0001, "warmup_steps": $WARMUP_STEPS, "save_every": $SAVE_STEPS, "max_frames": $MAX_FRAMES, "precision": "$PRECISION"},
  "record_root": "$RECORD_ROOT",
  "snapshot_root": "$SNAPSHOT_ROOT",
  "runner": "$RUNNER"
}
EOF

echo "[ver3.1-step4-submit] job=$JOB_NAME resource=MTTS-3-2-0715 8xH200"
echo "[ver3.1-step4-submit] index=$INDEX rows=$INDEX_ROWS sha256=$INDEX_SHA256"
echo "[ver3.1-step4-submit] output=$OUTPUT_ROOT"
echo "[ver3.1-step4-submit] contract=gbs32 steps=$STEPS lr=1e-4 warmup=$WARMUP_STEPS"
if [ "$DRY_RUN" = "1" ]; then
  printf 'job_name\tcompute_group\tspec\toutput_root\trunner\n%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "$COMPUTE_GROUP" "$SPEC" "$OUTPUT_ROOT" "$RUNNER" >"$RECORD_ROOT/submission_plan.tsv"
  echo "[ver3.1-step4-submit] DRY_RUN=1; no QZ job submitted"
  exit 0
fi

LOCK="$ROOT/trainset/qz_jobs/.codecVC-ver3-1-step4-ddlfm.live_submission.lock"
mkdir -p "$(dirname "$LOCK")"
mkdir "$LOCK" 2>/dev/null || die "submission lock exists: $LOCK"
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
LEDGER="$ROOT/trainset/qz_jobs/codecVC-ver3-1-step4-ddlfm.submitted_jobs.tsv"
[ ! -s "$LEDGER" ] || die "Step 4 submission ledger already exists"
SUBMIT_OUT="$RECORD_ROOT/submit_output.txt"
set +e
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \\
  QZCLI_GPU_TYPE_OVERRIDE="$GPU_TYPE" "$QZCLI" create-job \\
  --name "$JOB_NAME" --workspace "$WORKSPACE" --project "$PROJECT" \\
  --compute-group "$COMPUTE_GROUP" --spec "$SPEC" --framework "$FRAMEWORK" \\
  --instances 1 --shm "$SHM_GI" --priority "$PRIORITY" --image "$IMAGE" \\
  --image-type "$IMAGE_TYPE" --command "bash $RUNNER" >"$SUBMIT_OUT" 2>&1
STATUS=$?
set -e
cat "$SUBMIT_OUT"
[ "$STATUS" -eq 0 ] || die "QZ submission failed; see $SUBMIT_OUT"
JOB_ID="$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$SUBMIT_OUT" | tail -n1 || true)"
[ -n "$JOB_ID" ] || die "could not parse QZ job ID; preserve submit output"
printf 'job_name\tjob_id\tcompute_group\tspec\tinstances\tgpus\toutput_root\trunner\n%s\t%s\t%s\t%s\t1\t8\t%s\t%s\n' "$JOB_NAME" "$JOB_ID" "$COMPUTE_GROUP" "$SPEC" "$OUTPUT_ROOT" "$RUNNER" >"$RECORD_ROOT/submitted_jobs.tsv"
cp "$RECORD_ROOT/submitted_jobs.tsv" "$LEDGER"
echo "[ver3.1-step4-submit] submitted job_id=$JOB_ID"
