#!/usr/bin/env bash
# Batch-45 Step 3 semantic pre-extraction (no_text Path-A source BNF only).
# Dry-run by default; one approved MTTS-3-2-0715 8xH200 node is used with one
# extractor process per GPU. Text rows are intentionally not materialised.
set -Eeuo pipefail

ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
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
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PROGRESS_EVERY="${PROGRESS_EVERY:-2000}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_ID="${BATCH_ID:-codecVC-ver3-1-step3-semantic-no-text-$TIMESTAMP}"
JOB_NAME="${JOB_NAME:-$BATCH_ID}"
RECORD_ROOT="${RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
SNAPSHOT_ROOT="$RECORD_ROOT/record_snapshot"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/prepared/semantic_v1_v3_1_step3_no_text_20260715}"
CHECKPOINT="${CHECKPOINT:-$ROOT/outputs/ver3_1_content_adapter_probe_20260715/step-003000}"
MANIFEST="$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
MANIFEST_SHA256="c4b061f0a968e73710dc86d81478483a9195e8a053f510f09be7952d60c3d279"
EXTRACT_SCRIPT="$ROOT/scripts/ver3_1/extract_semantic_v3_1.py"

die() { echo "ERROR: $*" >&2; exit 1; }
[ "$QZCLI" = "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh" ] || die "only approved qzcli wrapper allowed"
[ -x "$QZCLI" ] || die "missing qzcli: $QZCLI"
[ -x "$PY" ] || die "missing Python: $PY"
[ "$COMPUTE_GROUP" = "$ALLOWED_COMPUTE_GROUP" ] || die "only MTTS-3-2-0715 is allowed"
[ "$SPEC" = "$ALLOWED_SPEC" ] || die "only registered 8xH200 spec is allowed"
[ "$INSTANCES" = "1" ] || die "one instance required"
[[ "$JOB_NAME" == codecVC-* ]] || die "job name must start with codecVC-"
[[ "$BATCH_ID" == codecVC-* ]] || die "batch id must start with codecVC-"
[[ "$JOB_NAME" =~ ^codecVC-[A-Za-z0-9_.-]+$ ]] || die "unsafe job name"
[[ "$BATCH_ID" =~ ^codecVC-[A-Za-z0-9_.-]+$ ]] || die "unsafe batch id"
[ -s "$EXTRACT_SCRIPT" ] || die "missing extraction script"
[ -s "$MANIFEST" ] || die "missing no_text v1 manifest"
[ "$(stat -c '%s' "$MANIFEST")" = "18048211813" ] || die "manifest size changed"
[ -d "$CHECKPOINT" ] || die "missing adapter checkpoint: $CHECKPOINT"
[ -s "$CHECKPOINT/adapter.pt" ] || die "missing checkpoint adapter.pt"
[ ! -e "$OUTPUT_ROOT/COMPLETED.json" ] || die "semantic output already complete"
if [ -d "$OUTPUT_ROOT" ] && [ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ] && [ "${ALLOW_NONEMPTY_OUTPUT:-0}" != "1" ]; then
  die "semantic output root is non-empty; choose a fresh root or set ALLOW_NONEMPTY_OUTPUT=1"
fi
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1" ;; esac
if [ "$DRY_RUN" = "0" ] && [ "${ALLOW_CODECVC_VER3_1_STEP3_SEMANTIC_SUBMIT:-0}" != "1" ]; then
  die "live submission guarded; set DRY_RUN=0 ALLOW_CODECVC_VER3_1_STEP3_SEMANTIC_SUBMIT=1"
fi

mkdir -p "$SNAPSHOT_ROOT/scripts/ver3_1" "$SNAPSHOT_ROOT/moss_codecvc"
cp -p "$EXTRACT_SCRIPT" "$SNAPSHOT_ROOT/scripts/ver3_1/"
cp -a "$ROOT/moss_codecvc/." "$SNAPSHOT_ROOT/moss_codecvc/"
cp -p "$CHECKPOINT/adapter.pt" "$SNAPSHOT_ROOT/adapter.pt"
cp -p "$CHECKPOINT/adapter_config.json" "$SNAPSHOT_ROOT/adapter_config.json"
SOURCE_GIT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
find "$SNAPSHOT_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum >"$SNAPSHOT_ROOT/SHA256SUMS"
printf '%s\n' "$SOURCE_GIT_SHA" >"$SNAPSHOT_ROOT/SOURCE_GIT_SHA"

RUNNER="$RECORD_ROOT/run_step3_semantic_extract_qz.sh"
LOG_ROOT="$RECORD_ROOT/worker_logs"
cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
export ROOT="$ROOT"
export PY="$PY"
export MANIFEST="$MANIFEST"
export OUTPUT_ROOT="$OUTPUT_ROOT"
export CHECKPOINT="$SNAPSHOT_ROOT"
export SNAPSHOT_ROOT="$SNAPSHOT_ROOT"
export HF_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface"
export TRANSFORMERS_CACHE="\$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="\$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="\$SNAPSHOT_ROOT:\$ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
mkdir -p "$LOG_ROOT"
PIDS=()
for shard in \$(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="\$shard" "\$PY" "\$SNAPSHOT_ROOT/scripts/ver3_1/extract_semantic_v3_1.py" \\
    --input "no_text=\$MANIFEST" --checkpoint "\$CHECKPOINT" --output-root "\$OUTPUT_ROOT" \\
    --feature-key source_wavlm_bnf_features_path --mode-filter no_text \\
    --shard-index "\$shard" --num-shards "$NUM_SHARDS" --device cuda:0 \\
    --precision bf16 --save-dtype float16 --batch-size "$BATCH_SIZE" --progress-every "$PROGRESS_EVERY" \\
    >"$LOG_ROOT/shard-\$(printf '%03d' \"\$shard\").log" 2>&1 &
  PIDS+=("\$!")
done
status=0
for pid in "\${PIDS[@]}"; do
  if ! wait "\$pid"; then status=1; fi
done
[ "\$status" -eq 0 ] || { echo "one or more semantic shards failed" >&2; exit "\$status"; }
"\$PY" "\$SNAPSHOT_ROOT/scripts/ver3_1/extract_semantic_v3_1.py" \\
  --mode aggregate --input "no_text=\$MANIFEST" --checkpoint "\$CHECKPOINT" \\
  --output-root "\$OUTPUT_ROOT" --mode-filter no_text --device cpu --precision float32
EOF
chmod +x "$RUNNER"

cat >"$RECORD_ROOT/preflight.json" <<EOF
{
  "schema": "ver3_1_step3_semantic_extract_submit_preflight_v1",
  "job_name": "$JOB_NAME",
  "source_git_sha": "$SOURCE_GIT_SHA",
  "dry_run": $DRY_RUN,
  "resource": {"compute_group": "$COMPUTE_GROUP", "compute_group_name": "MTTS-3-2-0715", "spec": "$SPEC", "instances": 1, "gpus": 8},
  "input": {"path": "$MANIFEST", "sha256": "$MANIFEST_SHA256", "rows": 310420, "mode": "no_text"},
  "checkpoint": {"path": "$CHECKPOINT", "semantic_dim": 512, "rate_hz": 12.5, "provenance": "Step-3 3k proxy-label adapter; no MFA phoneme gate"},
  "output": {"root": "$OUTPUT_ROOT", "expected_split": "no_text", "num_shards": $NUM_SHARDS, "batch_size": $BATCH_SIZE},
  "record_root": "$RECORD_ROOT",
  "snapshot_root": "$SNAPSHOT_ROOT",
  "runner": "$RUNNER"
}
EOF

echo "[ver3.1-step3-semantic-submit] job=$JOB_NAME resource=MTTS-3-2-0715 8xH200"
echo "[ver3.1-step3-semantic-submit] checkpoint=$CHECKPOINT"
echo "[ver3.1-step3-semantic-submit] output=$OUTPUT_ROOT"
echo "[ver3.1-step3-semantic-submit] no_text only; text semantic remains intentionally absent"
if [ "$DRY_RUN" = "1" ]; then
  printf 'job_name\tcompute_group\tspec\toutput_root\trunner\n%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "$COMPUTE_GROUP" "$SPEC" "$OUTPUT_ROOT" "$RUNNER" >"$RECORD_ROOT/submission_plan.tsv"
  echo "[ver3.1-step3-semantic-submit] DRY_RUN=1; no QZ job submitted"
  exit 0
fi

LOCK="$ROOT/trainset/qz_jobs/.codecVC-ver3-1-step3-semantic.live_submission.lock"
mkdir -p "$(dirname "$LOCK")"
mkdir "$LOCK" 2>/dev/null || die "submission lock exists: $LOCK"
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
LEDGER="$ROOT/trainset/qz_jobs/codecVC-ver3-1-step3-semantic.submitted_jobs.tsv"
[ ! -s "$LEDGER" ] || die "semantic extraction ledger already exists"
SUBMIT_OUT="$RECORD_ROOT/submit_output.txt"
set +e
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY \
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
[ -n "$JOB_ID" ] || die "could not parse QZ job ID"
printf 'job_name\tjob_id\tcompute_group\tspec\tinstances\tgpus\toutput_root\trunner\n%s\t%s\t%s\t%s\t1\t8\t%s\t%s\n' "$JOB_NAME" "$JOB_ID" "$COMPUTE_GROUP" "$SPEC" "$OUTPUT_ROOT" "$RUNNER" >"$RECORD_ROOT/submitted_jobs.tsv"
cp "$RECORD_ROOT/submitted_jobs.tsv" "$LEDGER"
echo "[ver3.1-step3-semantic-submit] submitted job_id=$JOB_ID"
