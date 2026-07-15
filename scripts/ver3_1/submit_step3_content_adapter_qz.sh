#!/usr/bin/env bash
# Batch-45 Step 3 adapter probe.  Dry-run by default; live submission is
# explicitly guarded and hard-bound to MTTS-3-2-0715 (one 8xH200 node).
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
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOG_STEPS="${LOG_STEPS:-25}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_ID="${BATCH_ID:-codecVC-ver3-1-step3-adapter-probe-$TIMESTAMP}"
JOB_NAME="${JOB_NAME:-$BATCH_ID}"
RECORD_ROOT="${RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
SNAPSHOT_ROOT="$RECORD_ROOT/record_snapshot"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/ver3_1_content_adapter_probe_20260715}"
MANIFEST="$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
MANIFEST_SHA256="c4b061f0a968e73710dc86d81478483a9195e8a053f510f09be7952d60c3d279"

TRAIN_SCRIPT="$ROOT/scripts/ver3_1/train_content_adapter.py"
ADAPTER_MODULE="$ROOT/moss_codecvc/models/content_adapter_v3_1.py"
CONTENT_ATTN="$ROOT/moss_codecvc/models/content_cross_attn.py"
CONFIG_MODULE="$ROOT/moss_codecvc/models/content_semantic_heads.py"
AUXILIARY_MODULE="$ROOT/moss_codecvc/models/auxiliary_losses.py"

die() { echo "ERROR: $*" >&2; exit 1; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

[ "$QZCLI" = "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh" ] || die "only approved qzcli wrapper allowed"
[ -x "$QZCLI" ] || die "missing qzcli: $QZCLI"
[ -x "$PY" ] || die "missing Python: $PY"
[ -x "$TORCHRUN" ] || die "missing torchrun: $TORCHRUN"
[ "$COMPUTE_GROUP" = "$ALLOWED_COMPUTE_GROUP" ] || die "only MTTS-3-2-0715 is allowed"
[ "$SPEC" = "$ALLOWED_SPEC" ] || die "only registered 8xH200 spec is allowed"
[ "$INSTANCES" = "1" ] || die "Step 3 requires one instance"
[[ "$JOB_NAME" == codecVC-* ]] || die "job name must start with codecVC-"
[[ "$BATCH_ID" == codecVC-* ]] || die "batch id must start with codecVC-"
[[ "$JOB_NAME" =~ ^codecVC-[A-Za-z0-9_.-]+$ ]] || die "unsafe job name"
[[ "$BATCH_ID" =~ ^codecVC-[A-Za-z0-9_.-]+$ ]] || die "unsafe batch id"
[ -s "$TRAIN_SCRIPT" ] || die "missing train script"
[ -s "$ADAPTER_MODULE" ] || die "missing adapter module"
[ -s "$CONTENT_ATTN" ] || die "missing conformer dependency"
[ -s "$CONFIG_MODULE" ] || die "missing classifier dependency"
[ -s "$AUXILIARY_MODULE" ] || die "missing auxiliary dependency"
[ -s "$MANIFEST" ] || die "missing v1 no_text manifest"
[ "$(stat -c '%s' "$MANIFEST")" = "18048211813" ] || die "v1 manifest size changed"
# The manifest is about 18 GiB.  Size plus the frozen Step-1 hash are recorded
# in preflight; avoid re-reading the whole file on every dry-run.  Set
# VERIFY_INPUT_SHA=1 when an explicit full re-hash is desired.
if [ "${VERIFY_INPUT_SHA:-0}" = "1" ]; then
  [ "$(sha256_file "$MANIFEST")" = "$MANIFEST_SHA256" ] || die "v1 manifest SHA256 changed"
fi
[ ! -e "$OUTPUT_ROOT/COMPLETED.json" ] || die "adapter output already complete"
if [ -d "$OUTPUT_ROOT" ] && [ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ] && [ "${ALLOW_NONEMPTY_OUTPUT:-0}" != "1" ]; then
  die "adapter output root is non-empty; choose a fresh OUTPUT_ROOT or set ALLOW_NONEMPTY_OUTPUT=1 explicitly"
fi
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1" ;; esac
if [ "$DRY_RUN" = "0" ] && [ "${ALLOW_CODECVC_VER3_1_STEP3_SUBMIT:-0}" != "1" ]; then
  die "live submission guarded; set DRY_RUN=0 ALLOW_CODECVC_VER3_1_STEP3_SUBMIT=1"
fi

mkdir -p "$SNAPSHOT_ROOT/scripts/ver3_1" "$SNAPSHOT_ROOT/moss_codecvc/models"
cp -p "$TRAIN_SCRIPT" "$SNAPSHOT_ROOT/scripts/ver3_1/"
# Copy the small Python package, not only the four immediate imports.  The
# package initializers export legacy modules, so a partial snapshot can still
# resolve code from the live checkout and defeat reproducibility.
cp -a "$ROOT/moss_codecvc/." "$SNAPSHOT_ROOT/moss_codecvc/"
SOURCE_GIT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
find "$SNAPSHOT_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum >"$SNAPSHOT_ROOT/SHA256SUMS"
printf '%s\n' "$SOURCE_GIT_SHA" >"$SNAPSHOT_ROOT/SOURCE_GIT_SHA"

RUNNER="$RECORD_ROOT/run_step3_content_adapter_qz.sh"
cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
export ROOT="$ROOT"
export PY="$PY"
export TORCHRUN="$TORCHRUN"
export MANIFEST="$MANIFEST"
export OUTPUT_ROOT="$OUTPUT_ROOT"
export SNAPSHOT_ROOT="$SNAPSHOT_ROOT"
export HF_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface"
export TRANSFORMERS_CACHE="\$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="\$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Put the immutable snapshot first; otherwise imports silently resolve to a
# dirty live checkout and the recorded SHA256SUMS is not the code being run.
export PYTHONPATH="$SNAPSHOT_ROOT:$ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
cd "\$ROOT"
\$TORCHRUN --standalone --nproc_per_node=8 \
  "\$SNAPSHOT_ROOT/scripts/ver3_1/train_content_adapter.py" \
  --input "no_text=\$MANIFEST" \
  --output-root "\$OUTPUT_ROOT" \
  --feature-key source_wavlm_bnf_features_path \
  --input-dim auto --semantic-dim 512 --num-layers 2 --num-heads 8 \
  --downsample-kernel 4 --downsample-stride 4 \
  --batch-size "$PER_DEVICE_BATCH" --max-steps "$STEPS" \
  --learning-rate 1e-4 --warmup-steps "$WARMUP_STEPS" \
  --save-steps "$SAVE_STEPS" --log-steps "$LOG_STEPS" \
  --device cuda --precision bf16
EOF
chmod +x "$RUNNER"

cat >"$RECORD_ROOT/preflight.json" <<EOF
{
  "schema": "ver3_1_step3_adapter_submit_preflight_v1",
  "job_name": "$JOB_NAME",
  "source_git_sha": "$SOURCE_GIT_SHA",
  "verify_input_sha": "${VERIFY_INPUT_SHA:-0}",
  "dry_run": $DRY_RUN,
  "resource": {"compute_group": "$COMPUTE_GROUP", "compute_group_name": "MTTS-3-2-0715", "spec": "$SPEC", "instances": 1, "gpus": 8},
  "input": {"path": "$MANIFEST", "sha256": "$MANIFEST_SHA256", "size_bytes": 18048211813, "mode": "no_text_only"},
  "contract": {"input_dim": "auto (v1 sidecar is 768)", "semantic_dim": 512, "layers": 2, "stride": 4, "steps": $STEPS, "per_device_batch": $PER_DEVICE_BATCH, "global_batch": $((PER_DEVICE_BATCH * 8)), "label_source": "content_token_ids pseudo-label; no MFA phoneme alignment"},
  "record_root": "$RECORD_ROOT",
  "snapshot_root": "$SNAPSHOT_ROOT",
  "runner": "$RUNNER"
}
EOF

echo "[ver3.1-step3-submit] job=$JOB_NAME resource=MTTS-3-2-0715 8xH200"
echo "[ver3.1-step3-submit] output=$OUTPUT_ROOT"
echo "[ver3.1-step3-submit] input=no_text only; text rows intentionally excluded from adapter pretraining"
if [ "$DRY_RUN" = "1" ]; then
  printf 'job_name\tcompute_group\tspec\toutput_root\trunner\n%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "$COMPUTE_GROUP" "$SPEC" "$OUTPUT_ROOT" "$RUNNER" >"$RECORD_ROOT/submission_plan.tsv"
  echo "[ver3.1-step3-submit] DRY_RUN=1; no QZ job submitted"
  exit 0
fi

LOCK="$ROOT/trainset/qz_jobs/.codecVC-ver3-1-step3-adapter.live_submission.lock"
mkdir -p "$(dirname "$LOCK")"
mkdir "$LOCK" 2>/dev/null || die "submission lock exists: $LOCK"
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
LEDGER="$ROOT/trainset/qz_jobs/codecVC-ver3-1-step3-adapter.submitted_jobs.tsv"
[ ! -s "$LEDGER" ] || die "Step 3 submission ledger already exists"
SUBMIT_OUT="$RECORD_ROOT/submit_output.txt"
set +e
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
  QZCLI_GPU_TYPE_OVERRIDE="$GPU_TYPE" "$QZCLI" create-job \
  --name "$JOB_NAME" --workspace "$WORKSPACE" --project "$PROJECT" \
  --compute-group "$COMPUTE_GROUP" --spec "$SPEC" --framework "$FRAMEWORK" \
  --instances 1 --shm "$SHM_GI" --priority "$PRIORITY" --image "$IMAGE" \
  --image-type "$IMAGE_TYPE" --command "bash $RUNNER" >"$SUBMIT_OUT" 2>&1
STATUS=$?
set -e
cat "$SUBMIT_OUT"
[ "$STATUS" -eq 0 ] || die "QZ submission failed; see $SUBMIT_OUT"
JOB_ID="$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$SUBMIT_OUT" | tail -n1 || true)"
[ -n "$JOB_ID" ] || die "could not parse QZ job ID; preserve submit output"
printf 'job_name\tjob_id\tcompute_group\tspec\tinstances\tgpus\toutput_root\trunner\n%s\t%s\t%s\t%s\t1\t8\t%s\t%s\n' "$JOB_NAME" "$JOB_ID" "$COMPUTE_GROUP" "$SPEC" "$OUTPUT_ROOT" "$RUNNER" >"$RECORD_ROOT/submitted_jobs.tsv"
cp "$RECORD_ROOT/submitted_jobs.tsv" "$LEDGER"
echo "[ver3.1-step3-submit] submitted job_id=$JOB_ID"
