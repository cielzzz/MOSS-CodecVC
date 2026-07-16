#!/usr/bin/env bash
# Batch-47: aggressive endpoint-conditioned ver3.1 DDLFM no_text-only probe.
#
# Safety contract:
#   * dry-run is the default and never calls qzcli;
#   * live submission additionally requires a clean, tagged branch and an
#     explicit ALLOW_CODECVC_BATCH47_SUBMIT=1 guard;
#   * critical model/data/resource settings are constants, not casual CLI
#     overrides, so the submitted experiment remains the preregistered arm.
set -Eeuo pipefail

ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
APPROVED_QZCLI="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh"
QZCLI="${QZCLI:-$APPROVED_QZCLI}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
TORCHRUN="${TORCHRUN:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/torchrun}"

WORKSPACE="${WORKSPACE:-CI-情境智能}"
PROJECT="${PROJECT:-CI-情境智能}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"             # 1x8 H200
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
INSTANCES="${INSTANCES:-1}"
GPU_TYPE="${GPU_TYPE:-NVIDIA_H200_SXM_141G}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"
DRY_RUN="${DRY_RUN:-1}"

# The exact Batch-47 training arm.  These are intentionally not environment
# overrides: changing one creates a different experiment and needs a new
# wrapper/name.
STEPS=3000
PER_DEVICE_BATCH=2
GRAD_ACCUM_STEPS=2
WORLD_SIZE=8
GLOBAL_BATCH=32
NUM_WORKERS=2
LEARNING_RATE="1e-4"
WARMUP_STEPS=3000
SAVE_STEPS=500
LOG_STEPS=20
MAX_FRAMES=256
PRECISION="bf16"
SEED=20260715
LATENT_DIM=768
SEMANTIC_DIM=512
SPEAKER_DIM=192
HIDDEN_SIZE=768
NUM_LAYERS=12
NUM_HEADS=12
FFN_SIZE=3072
TRAIN_MODE="no_text"
T_SAMPLING="shift_low"
T_LOGIT_MU="0.0"
T_LOGIT_SIGMA="1.0"
T_SHIFT_POWER="4.0"
T_MODE_SHIFT_M="3.0"
LOSS_WEIGHTING="low_t"
LOSS_WEIGHT_EPS="0.02"
LOSS_WEIGHT_CAP="25.0"
SPEAKER_DROPOUT="0.25"
SEMANTIC_DROPOUT="0.15"
AUX_LOSS_WEIGHT="1.0"
AUX_WARMUP_STEPS=500
EMA_DECAY="0.9999"
EMA_WARMUP=1
CROSS_GATE_INIT="0.05"
GATE_WARMUP_STEPS=500
GATE_WARMUP_START="0.05"
EVAL_SPEAKER_CFG_SCALE="2.5"
EVAL_SEMANTIC_CFG_SCALE="2.0"
EVAL_USE_EMA=1
NUM_SPEAKER_PROMPT_TOKENS=4
SPEAKER_CONDITION_SCALE="4.0"
SPEAKER_INPUT_SCALE="1.0"

# Evaluation is normally every saved checkpoint for Batch-47.  Newer
# batches may deliberately reserve audio evaluation for a sparse set of
# checkpoints while keeping the identifiability watcher at every 500 steps.
# Keep the historical default, but make the contract explicit and
# machine-readable for wrappers such as Batch-48.
LOCAL_QUICK20_STEPS="${LOCAL_QUICK20_STEPS:-500,1000,1500,2000,2500,3000}"
LOCAL_FULL_VALIDATION_AT="${LOCAL_FULL_VALIDATION_AT:-3000}"

REQUIRED_BRANCH="${REQUIRED_BRANCH:-feat/ver3_1_batch47_endpoint_rescue}"
REQUIRED_READY_TAG="${REQUIRED_READY_TAG:-ver3_1_batch47_fixes_ready}"
BATCH_ID="${BATCH_ID:-codecVC-ver3-1-batch47-ddlfm-no-text-3k-probe-20260716}"
JOB_NAME="${JOB_NAME:-$BATCH_ID}"
RECORD_ROOT="${RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
SNAPSHOT_ROOT="$RECORD_ROOT/record_snapshot"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/ver3_1_batch47_ddlfm_no_text_probe_20260716}"

INDEX="$ROOT/prepared/ddlfm_v1_index.jsonl"
INDEX_SUMMARY="$ROOT/prepared/ddlfm_v1_index.summary.json"
INDEX_SHA256="d114d222259fbf66cfb9f001929a216bd4cb7f3e2ac46a4c1dff6c56053c5804"
INDEX_ROWS=342839
NO_TEXT_ROWS=310420
ZQ_SUMMARY="$ROOT/prepared/zq_targets_v1/COMPLETED.json"
ZQ_MANIFEST_SHA256="1d5e7dcd11b23bb91e4a0be27c91fb2f964f8173bdc57f753ed5abffdbdfa8b6"
CHANNEL_STATS="$ROOT/prepared/zq_targets_v1/channel_stats.pt"
CHANNEL_STATS_AUDIT="$ROOT/prepared/zq_targets_v1/channel_stats.json"
EXPECTED_ZQ_FRAMES=35098460
SEMANTIC_COMPLETION="$ROOT/prepared/semantic_v1_v3_1_step3_no_text_20260715/COMPLETED.json"
NORMALIZATION_SANITY_REPORT="$ROOT/testset/outputs/ver3_1_batch46_zq_normalization_sanity_20260716/report.json"
TINY_GATE_REPORT="${TINY_GATE_REPORT:-$ROOT/testset/outputs/ver3_1_batch47_endpoint_gate_20260716/report.json}"
TRAIN_SCRIPT="$ROOT/scripts/ver3_1/train_ddlfm_cfm.py"
INFER_SCRIPT="$ROOT/scripts/ver3_1/infer_ddlfm_cfm.py"
EVAL_SCRIPT="$ROOT/scripts/ver3_1/evaluate_ddlfm_validation.py"
DECODER_MODULE="$ROOT/moss_codecvc/models/ddlfm_decoder.py"
NORMALIZATION_MODULE="$ROOT/moss_codecvc/audio/zq_normalization.py"

# This escape hatch exists only so the wrapper can receive a static dry-run
# review while the one-pass canonical statistics job is still running.  It
# never marks the arm launch-ready and is rejected for DRY_RUN=0.
ALLOW_PENDING_CHANNEL_STATS_DRY_RUN="${ALLOW_PENDING_CHANNEL_STATS_DRY_RUN:-0}"
ALLOW_FAILED_TINY_GATE="${ALLOW_FAILED_TINY_GATE:-0}"

die() { echo "ERROR: $*" >&2; exit 1; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1" ;; esac
case "$ALLOW_PENDING_CHANNEL_STATS_DRY_RUN" in 0|1) ;; *) die "ALLOW_PENDING_CHANNEL_STATS_DRY_RUN must be 0 or 1" ;; esac
case "$ALLOW_FAILED_TINY_GATE" in 0|1) ;; *) die "ALLOW_FAILED_TINY_GATE must be 0 or 1" ;; esac
[ -n "$LOCAL_QUICK20_STEPS" ] || die "LOCAL_QUICK20_STEPS must not be empty"
[[ "$LOCAL_QUICK20_STEPS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "LOCAL_QUICK20_STEPS must be comma-separated integers"
[ "$LOCAL_FULL_VALIDATION_AT" -gt 0 ] 2>/dev/null || die "LOCAL_FULL_VALIDATION_AT must be a positive integer"
[ "$QZCLI" = "$APPROVED_QZCLI" ] || die "only the approved project qzcli wrapper may be used"
[ -x "$QZCLI" ] || die "approved qzcli wrapper is missing or not executable: $QZCLI"
[ -x "$PY" ] || die "missing Python: $PY"
[ -x "$TORCHRUN" ] || die "missing torchrun: $TORCHRUN"
[ "$COMPUTE_GROUP" = "$ALLOWED_COMPUTE_GROUP" ] || die "only MTTS-3-2-0715 is allowed"
[ "$SPEC" = "$ALLOWED_SPEC" ] || die "only the registered 1x8 H200 spec is allowed"
[ "$INSTANCES" = "1" ] || die "Batch-47 requires exactly one instance"
[ "$GPU_TYPE" = "NVIDIA_H200_SXM_141G" ] || die "Batch-47 requires NVIDIA_H200_SXM_141G"
[ "$GLOBAL_BATCH" -eq $((PER_DEVICE_BATCH * GRAD_ACCUM_STEPS * WORLD_SIZE)) ] || die "global batch contract is inconsistent"
[[ "$JOB_NAME" == codecVC-* ]] || die "job name must start with codecVC-"
[[ "$BATCH_ID" == codecVC-* ]] || die "batch id must start with codecVC-"
[[ "$JOB_NAME" =~ ^codecVC-[A-Za-z0-9_.-]+$ ]] || die "unsafe job name"
[[ "$BATCH_ID" =~ ^codecVC-[A-Za-z0-9_.-]+$ ]] || die "unsafe batch id"
[ "$RECORD_ROOT" = "$ROOT/trainset/qz_jobs/$BATCH_ID" ] || die "record root must equal trainset/qz_jobs/BATCH_ID"

for required in \
  "$INDEX" "$INDEX_SUMMARY" "$ZQ_SUMMARY" "$SEMANTIC_COMPLETION" \
  "$TRAIN_SCRIPT" "$INFER_SCRIPT" "$EVAL_SCRIPT" "$DECODER_MODULE" "$NORMALIZATION_MODULE"; do
  [ -s "$required" ] || die "missing required input: $required"
done
[ "$(wc -l < "$INDEX" | tr -d ' ')" = "$INDEX_ROWS" ] || die "DDLFM index row count changed"
[ "$(sha256_file "$INDEX")" = "$INDEX_SHA256" ] || die "DDLFM index SHA256 changed"

# Verify that the current training entry point actually exposes every option
# this wrapper promises to pass.  This catches a stale snapshot before QZ.
TRAIN_HELP="$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$TRAIN_SCRIPT" --help)"
for option in \
  --mode --t-sampling --t-logit-mu --t-logit-sigma --t-shift-power \
  --t-mode-shift-m --loss-weighting --loss-weight-eps --loss-weight-cap \
  --speaker-dropout --semantic-dropout --aux-loss-weight --aux-warmup-steps \
  --speaker-cfg-scale --semantic-cfg-scale --num-speaker-prompt-tokens \
  --speaker-condition-scale --speaker-input-scale \
  --cross-gate-init --gate-warmup-steps --gate-warmup-start --ema-decay \
  --ema-warmup --zq-channel-stats; do
  grep -q -- "$option" <<<"$TRAIN_HELP" || die "training entry point is missing $option"
done

"$PY" - "$ZQ_SUMMARY" "$SEMANTIC_COMPLETION" "$INDEX_SUMMARY" <<'PY'
import json
import sys

zq = json.load(open(sys.argv[1], encoding="utf-8"))
semantic = json.load(open(sys.argv[2], encoding="utf-8"))
index = json.load(open(sys.argv[3], encoding="utf-8"))
checks = {
    "zq completed": zq.get("status") == "completed",
    "zq errors=0": int(zq.get("errors", -1)) == 0,
    "zq rows": int(zq.get("total_utterances", -1)) == 342839,
    "zq frames": int(zq.get("total_frames", -1)) == 35098460,
    "zq latent dim": int(zq.get("latent_dim", -1)) == 768,
    "zq frame rate": float(zq.get("frame_rate_hz", -1)) == 12.5,
    "semantic completed": semantic.get("status") == "completed",
    "semantic no_text rows": int(semantic.get("rows", -1)) == 310420,
    "semantic dim": int(semantic.get("semantic_dim", -1)) == 512,
    "semantic rate": float(semantic.get("rate_hz", -1)) == 12.5,
    "index completed": index.get("status") == "completed",
    "index rows": int(index.get("rows", -1)) == 342839,
    "index no_text rows": int((index.get("mode_counts") or {}).get("no_text", -1)) == 310420,
    "index no missing no_text semantic": int(index.get("missing_no_text_semantic", -1)) == 0,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("data completion contract failed: " + ", ".join(failed))
PY

SOURCE_BRANCH="$(git -C "$ROOT" branch --show-current)"
SOURCE_GIT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
WORKTREE_CLEAN=0
[ -z "$(git -C "$ROOT" status --porcelain)" ] && WORKTREE_CLEAN=1
READY_TAG_AT_HEAD=0
if git -C "$ROOT" rev-parse -q --verify "refs/tags/$REQUIRED_READY_TAG" >/dev/null; then
  [ "$(git -C "$ROOT" rev-list -n1 "$REQUIRED_READY_TAG")" = "$SOURCE_GIT_SHA" ] && READY_TAG_AT_HEAD=1
fi

CHANNEL_STATS_READY=0
CHANNEL_STATS_SHA256="pending"
if [ -s "$CHANNEL_STATS" ]; then
  CHANNEL_STATS_SHA256="$(sha256_file "$CHANNEL_STATS")"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" - \
    "$CHANNEL_STATS" "$CHANNEL_STATS_AUDIT" "$CHANNEL_STATS_SHA256" "$ZQ_MANIFEST_SHA256" \
    "$INDEX_ROWS" "$EXPECTED_ZQ_FRAMES" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

import torch

stats_path = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
actual_sha = sys.argv[3]
manifest_sha = sys.argv[4]
expected_rows = int(sys.argv[5])
expected_frames = int(sys.argv[6])
stats = torch.load(stats_path, map_location="cpu", weights_only=False)
checks = {
    "schema": stats.get("schema") == "ver3_1_zq_channel_stats_v1",
    "status": stats.get("status") == "completed",
    "not partial": not bool(stats.get("partial", True)),
    "latent dim": int(stats.get("latent_dim", -1)) == 768,
    "row count": int(stats.get("row_count", -1)) == expected_rows,
    "frame count": int(stats.get("frame_count", -1)) == expected_frames,
    "frame rate": float(stats.get("frame_rate_hz", -1)) == 12.5,
    "manifest sha": stats.get("manifest_sha256") == manifest_sha,
}
for name in ("mean", "std"):
    value = torch.as_tensor(stats.get(name))
    checks[f"{name} shape"] = tuple(value.shape) == (768,)
    checks[f"{name} finite"] = bool(torch.isfinite(value).all())
checks["std positive"] = bool(torch.all(torch.as_tensor(stats.get("std")) > 0))
if audit_path.is_file():
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks["audit sha"] = audit.get("channel_stats_sha256") == actual_sha
else:
    checks["audit exists"] = False
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("canonical channel-stats contract failed: " + ", ".join(failed))
PY
  CHANNEL_STATS_READY=1
elif [ "$DRY_RUN" = "1" ] && [ "$ALLOW_PENDING_CHANNEL_STATS_DRY_RUN" = "1" ]; then
  echo "[batch47-submit] WARNING: canonical channel_stats.pt is pending; static dry-run only" >&2
else
  die "canonical channel stats are not ready: $CHANNEL_STATS"
fi

NORMALIZATION_SANITY_READY=0
TINY_GATE_READY=0
if [ "$CHANNEL_STATS_READY" = "1" ]; then
  [ -s "$NORMALIZATION_SANITY_REPORT" ] || die "missing normalization sanity report: $NORMALIZATION_SANITY_REPORT"
  [ -s "$TINY_GATE_REPORT" ] || die "missing tiny identifiability report: $TINY_GATE_REPORT"
"$PY" - "$NORMALIZATION_SANITY_REPORT" "$TINY_GATE_REPORT" "$CHANNEL_STATS_SHA256" <<'PY'
import json
import math
import os
import sys

normalization = json.load(open(sys.argv[1], encoding="utf-8"))
tiny = json.load(open(sys.argv[2], encoding="utf-8"))
stats_sha = sys.argv[3]
normalization_checks = {
    "status": normalization.get("status") == "passed",
    "five cases": int(normalization.get("num_cases", -1)) == 5,
    "five passed": int(normalization.get("passed_cases", -1)) == 5,
    "stats sha": normalization.get("stats_sha256") == stats_sha,
    "latent roundtrip": float(normalization.get("max_latent_abs_diff", math.inf)) < 1.0e-6,
    "waveform finite tolerance": float(normalization.get("max_waveform_abs_diff", math.inf)) <= 1.0e-5,
}
tiny_gates = tiny.get("gates") or {}
tiny_checks = {
    "status": tiny.get("status") == "passed",
    "steps": int(tiny.get("steps", -1)) in (800, 3000),
    "speaker_cfg": float(tiny.get("speaker_cfg_scale", -1)) == 2.5,
    "semantic_cfg": float(tiny.get("semantic_cfg_scale", -1)) == 2.0,
    "stats sha": tiny.get("zq_channel_stats_sha256") == stats_sha,
    "all gates": bool(tiny_gates) and all(bool(value) for value in tiny_gates.values()),
}
failed = [f"normalization:{name}" for name, ok in normalization_checks.items() if not ok]
tiny_failed = [f"tiny:{name}" for name, ok in tiny_checks.items() if not ok]
allow_failed_tiny = os.environ.get("ALLOW_FAILED_TINY_GATE", "0") == "1"
if tiny_failed and allow_failed_tiny:
    print("WARNING: accepting failed tiny identifiability gate under explicit Batch-48 override: " + ", ".join(tiny_failed))
else:
    failed += tiny_failed
if failed:
    raise SystemExit("Batch local sanity contract failed: " + ", ".join(failed))
PY
  NORMALIZATION_SANITY_READY=1
  TINY_GATE_READY=1
fi

if [ "$DRY_RUN" = "0" ]; then
  [ "$ALLOW_PENDING_CHANNEL_STATS_DRY_RUN" = "0" ] || die "pending-stats escape hatch is forbidden for live submission"
  [ "$CHANNEL_STATS_READY" = "1" ] || die "live submission requires canonical channel stats"
  [ "$NORMALIZATION_SANITY_READY" = "1" ] || die "live submission requires normalization sanity"
  [ "$TINY_GATE_READY" = "1" ] || die "live submission requires tiny identifiability gate"
  [ "$SOURCE_BRANCH" = "$REQUIRED_BRANCH" ] || die "live submission must stay on $REQUIRED_BRANCH"
  [ "$WORKTREE_CLEAN" = "1" ] || die "live submission requires a clean worktree"
  [ "$READY_TAG_AT_HEAD" = "1" ] || die "live submission requires $REQUIRED_READY_TAG at HEAD"
  SUBMIT_GUARD="${ALLOW_CODECVC_BATCH48_SUBMIT:-${ALLOW_CODECVC_BATCH47_SUBMIT:-0}}"
  [ "$SUBMIT_GUARD" = "1" ] || die "live submission guarded; set the explicit Batch submit guard"
fi

[ ! -e "$OUTPUT_ROOT/COMPLETED.json" ] || die "Batch-47 output is already complete: $OUTPUT_ROOT"
if [ -d "$OUTPUT_ROOT" ] && [ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  die "Batch-47 output root is non-empty; use a fresh BATCH_ID/OUTPUT_ROOT"
fi

mkdir -p "$SNAPSHOT_ROOT/scripts/ver3_1" "$SNAPSHOT_ROOT/moss_codecvc" \
  "$SNAPSHOT_ROOT/prepared/zq_targets_v1" "$SNAPSHOT_ROOT/sanity"
cp -a "$ROOT/moss_codecvc/." "$SNAPSHOT_ROOT/moss_codecvc/"
cp -p "$TRAIN_SCRIPT" "$INFER_SCRIPT" "$EVAL_SCRIPT" "$SNAPSHOT_ROOT/scripts/ver3_1/"
cp -p "$INDEX_SUMMARY" "$SNAPSHOT_ROOT/prepared/ddlfm_v1_index.summary.json"
if [ "$CHANNEL_STATS_READY" = "1" ]; then
  cp -p "$CHANNEL_STATS" "$SNAPSHOT_ROOT/prepared/zq_targets_v1/channel_stats.pt"
  cp -p "$CHANNEL_STATS_AUDIT" "$SNAPSHOT_ROOT/prepared/zq_targets_v1/channel_stats.json"
  cp -p "$NORMALIZATION_SANITY_REPORT" "$SNAPSHOT_ROOT/sanity/zq_normalization_report.json"
  cp -p "$TINY_GATE_REPORT" "$SNAPSHOT_ROOT/sanity/tiny_identifiability_report.json"
fi
printf '%s\n' "$SOURCE_GIT_SHA" >"$SNAPSHOT_ROOT/SOURCE_GIT_SHA"
printf '%s\n' "$SOURCE_BRANCH" >"$SNAPSHOT_ROOT/SOURCE_BRANCH"
find "$SNAPSHOT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$SNAPSHOT_ROOT/SHA256SUMS"

SNAPSHOT_STATS="$SNAPSHOT_ROOT/prepared/zq_targets_v1/channel_stats.pt"
RUNNER="$RECORD_ROOT/run_batch47_ddlfm_probe_qz.sh"
cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
export ROOT="$ROOT"
export PY="$PY"
export TORCHRUN="$TORCHRUN"
export INDEX="$INDEX"
export OUTPUT_ROOT="$OUTPUT_ROOT"
export SNAPSHOT_ROOT="$SNAPSHOT_ROOT"
export CHANNEL_STATS="$SNAPSHOT_STATS"
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

[ -s "\$CHANNEL_STATS" ] || { echo "missing canonical channel stats snapshot: \$CHANNEL_STATS" >&2; exit 1; }
[ "\$(sha256sum "\$INDEX" | awk '{print \$1}')" = "$INDEX_SHA256" ] || { echo "runtime index SHA mismatch" >&2; exit 1; }
[ "\$(sha256sum "\$CHANNEL_STATS" | awk '{print \$1}')" = "$CHANNEL_STATS_SHA256" ] || { echo "runtime channel stats SHA mismatch" >&2; exit 1; }
mkdir -p "\$OUTPUT_ROOT"

"\$TORCHRUN" --standalone --nproc_per_node=$WORLD_SIZE \\
  "\$SNAPSHOT_ROOT/scripts/ver3_1/train_ddlfm_cfm.py" \\
  --index "\$INDEX" \\
  --output-dir "\$OUTPUT_ROOT" \\
  --mode "$TRAIN_MODE" \\
  --steps "$STEPS" \\
  --batch-size "$PER_DEVICE_BATCH" \\
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \\
  --num-workers "$NUM_WORKERS" \\
  --lr "$LEARNING_RATE" \\
  --warmup-steps "$WARMUP_STEPS" \\
  --save-every "$SAVE_STEPS" \\
  --log-every "$LOG_STEPS" \\
  --max-frames "$MAX_FRAMES" \\
  --precision "$PRECISION" \\
  --seed "$SEED" \\
  --latent-dim "$LATENT_DIM" \\
  --semantic-dim "$SEMANTIC_DIM" \\
  --speaker-dim "$SPEAKER_DIM" \\
  --hidden-size "$HIDDEN_SIZE" \\
  --num-layers "$NUM_LAYERS" \\
  --num-heads "$NUM_HEADS" \\
  --ffn-size "$FFN_SIZE" \\
  --num-speaker-prompt-tokens "$NUM_SPEAKER_PROMPT_TOKENS" \\
  --speaker-condition-scale "$SPEAKER_CONDITION_SCALE" \\
  --speaker-input-scale "$SPEAKER_INPUT_SCALE" \\
  --t-sampling "$T_SAMPLING" \\
  --t-logit-mu "$T_LOGIT_MU" \\
  --t-logit-sigma "$T_LOGIT_SIGMA" \\
  --t-shift-power "$T_SHIFT_POWER" \\
  --t-mode-shift-m "$T_MODE_SHIFT_M" \\
  --loss-weighting "$LOSS_WEIGHTING" \\
  --loss-weight-eps "$LOSS_WEIGHT_EPS" \\
  --loss-weight-cap "$LOSS_WEIGHT_CAP" \\
  --speaker-dropout "$SPEAKER_DROPOUT" \\
  --semantic-dropout "$SEMANTIC_DROPOUT" \\
  --aux-loss-weight "$AUX_LOSS_WEIGHT" \\
  --aux-warmup-steps "$AUX_WARMUP_STEPS" \\
  --speaker-cfg-scale "$EVAL_SPEAKER_CFG_SCALE" \\
  --semantic-cfg-scale "$EVAL_SEMANTIC_CFG_SCALE" \\
  --ema-decay "$EMA_DECAY" \\
  --ema-warmup \\
  --cross-gate-init "$CROSS_GATE_INIT" \\
  --gate-warmup-steps "$GATE_WARMUP_STEPS" \\
  --gate-warmup-start "$GATE_WARMUP_START" \\
  --zq-channel-stats "\$CHANNEL_STATS" \\
  --device cuda

"\$PY" - "\$OUTPUT_ROOT" <<PY
import hashlib
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
last = root / "last.pt"
config = root / "config.json"
ready_path = root / "step-003000.ready.json"
if not last.is_file() or not config.is_file() or not ready_path.is_file():
    raise SystemExit("training returned without last.pt/config.json/step-003000.ready.json")
ready = json.loads(ready_path.read_text(encoding="utf-8"))
if ready.get("status") != "ready" or int(ready.get("step", -1)) != 3000:
    raise SystemExit("final checkpoint readiness marker is invalid")
inference_checkpoint = Path(str(ready.get("inference_checkpoint") or ""))
if not inference_checkpoint.is_file():
    raise SystemExit(f"final inference checkpoint is missing: {inference_checkpoint}")
payload = {
    "schema": "ver3_1_ddlfm_probe_completion_v2",
    "batch_id": "$BATCH_ID",
    "status": "completed",
    "completed_at_unix": time.time(),
    "steps": 3000,
    "mode": "no_text",
    "last_checkpoint": str(last),
    "last_checkpoint_sha256": hashlib.sha256(last.read_bytes()).hexdigest(),
    "inference_checkpoint": str(inference_checkpoint),
    "checkpoint_ready_marker": str(ready_path),
    "local_evaluation": {
        "device": "local RTX4090",
        "speaker_cfg_scale": 2.5,
        "semantic_cfg_scale": 2.0,
        "checkpoints": [int(item) for item in "$LOCAL_QUICK20_STEPS".split(",") if item],
        "quick20_steps": [int(item) for item in "$LOCAL_QUICK20_STEPS".split(",") if item],
        "full_validation_step": int("$LOCAL_FULL_VALIDATION_AT"),
        "primary": {"weights": "ema", "use_ema": True},
        "diagnostic": {"weights": "raw", "use_ema": False},
        "final_scope": "no_text validation cases only",
    },
}
(root / "COMPLETED.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False), flush=True)
PY
EOF
chmod +x "$RUNNER"

LAUNCH_READY=0
if [ "$CHANNEL_STATS_READY" = "1" ] && [ "$NORMALIZATION_SANITY_READY" = "1" ] && \
   [ "$TINY_GATE_READY" = "1" ] && [ "$SOURCE_BRANCH" = "$REQUIRED_BRANCH" ] && \
   [ "$WORKTREE_CLEAN" = "1" ] && [ "$READY_TAG_AT_HEAD" = "1" ]; then
  LAUNCH_READY=1
fi

"$PY" - "$RECORD_ROOT/preflight.json" <<PY
import json
import pathlib

quick20_steps = [int(item) for item in "$LOCAL_QUICK20_STEPS".split(",") if item]
if not quick20_steps or any(step <= 0 for step in quick20_steps):
    raise SystemExit("invalid LOCAL_QUICK20_STEPS")

payload = {
    "schema": "ver3_1_batch47_ddlfm_qz_preflight_v1",
    "job_name": "$JOB_NAME",
    "batch_id": "$BATCH_ID",
    "dry_run": bool(int("$DRY_RUN")),
    "launch_ready": bool(int("$LAUNCH_READY")),
    "source": {
        "branch": "$SOURCE_BRANCH",
        "git_sha": "$SOURCE_GIT_SHA",
        "worktree_clean": bool(int("$WORKTREE_CLEAN")),
        "required_branch": "$REQUIRED_BRANCH",
        "required_ready_tag": "$REQUIRED_READY_TAG",
        "ready_tag_at_head": bool(int("$READY_TAG_AT_HEAD")),
    },
    "resource": {
        "compute_group": "$COMPUTE_GROUP",
        "compute_group_name": "MTTS-3-2-0715",
        "spec": "$SPEC",
        "instances": 1,
        "gpus": 8,
        "gpu_type": "$GPU_TYPE",
    },
    "data": {
        "version": "v1",
        "mode": "no_text",
        "index": "$INDEX",
        "index_sha256": "$INDEX_SHA256",
        "index_rows_all": $INDEX_ROWS,
        "train_rows_no_text": $NO_TEXT_ROWS,
        "zq_manifest_sha256": "$ZQ_MANIFEST_SHA256",
        "channel_stats": "$CHANNEL_STATS",
        "channel_stats_ready": bool(int("$CHANNEL_STATS_READY")),
        "channel_stats_sha256": "$CHANNEL_STATS_SHA256",
        "channel_stats_expected_rows": $INDEX_ROWS,
        "channel_stats_expected_frames": $EXPECTED_ZQ_FRAMES,
        "normalization_sanity_report": "$NORMALIZATION_SANITY_REPORT",
        "normalization_sanity_ready": bool(int("$NORMALIZATION_SANITY_READY")),
        "tiny_identifiability_report": "$TINY_GATE_REPORT",
        "tiny_identifiability_ready": bool(int("$TINY_GATE_READY")),
        "allow_failed_tiny_gate": bool(int("$ALLOW_FAILED_TINY_GATE")),
    },
    "training": {
        "steps": $STEPS,
        "per_device_batch": $PER_DEVICE_BATCH,
        "gradient_accumulation": $GRAD_ACCUM_STEPS,
        "world_size": $WORLD_SIZE,
        "global_batch": $GLOBAL_BATCH,
        "lr": 1e-4,
        "warmup_steps": $WARMUP_STEPS,
        "save_every": $SAVE_STEPS,
        "seed": $SEED,
        "latent_dim": $LATENT_DIM,
        "semantic_dim": $SEMANTIC_DIM,
        "speaker_dim": $SPEAKER_DIM,
        "hidden_size": $HIDDEN_SIZE,
        "num_layers": $NUM_LAYERS,
        "num_heads": $NUM_HEADS,
        "ffn_size": $FFN_SIZE,
        "num_speaker_prompt_tokens": $NUM_SPEAKER_PROMPT_TOKENS,
        "speaker_condition_scale": $SPEAKER_CONDITION_SCALE,
        "speaker_input_scale": $SPEAKER_INPUT_SCALE,
        "t_sampling": "$T_SAMPLING",
        "t_logit_mu": $T_LOGIT_MU,
        "t_logit_sigma": $T_LOGIT_SIGMA,
        "t_shift_power": $T_SHIFT_POWER,
        "t_mode_shift_m": $T_MODE_SHIFT_M,
        "loss_weighting": "$LOSS_WEIGHTING",
        "loss_weight_eps": $LOSS_WEIGHT_EPS,
        "loss_weight_cap": $LOSS_WEIGHT_CAP,
        "speaker_dropout": $SPEAKER_DROPOUT,
        "semantic_dropout": $SEMANTIC_DROPOUT,
        "aux_loss_weight": $AUX_LOSS_WEIGHT,
        "aux_warmup_steps": $AUX_WARMUP_STEPS,
        "speaker_cfg_scale": $EVAL_SPEAKER_CFG_SCALE,
        "semantic_cfg_scale": $EVAL_SEMANTIC_CFG_SCALE,
        "ema_decay": $EMA_DECAY,
        "ema_warmup": bool(int("$EMA_WARMUP")),
        "cross_gate_init": $CROSS_GATE_INIT,
        "gate_warmup_steps": $GATE_WARMUP_STEPS,
        "gate_warmup_start": $GATE_WARMUP_START,
    },
    "local_evaluation": {
        "speaker_cfg_scale": $EVAL_SPEAKER_CFG_SCALE,
        "semantic_cfg_scale": $EVAL_SEMANTIC_CFG_SCALE,
        "mode": "no_text",
        "quick20_checkpoints": quick20_steps,
        "primary": {
            "weights": "ema",
            "use_ema": bool(int("$EVAL_USE_EMA")),
            "required_every_checkpoint": True,
        },
        "diagnostic": {
            "weights": "raw",
            "use_ema": False,
            "required_every_checkpoint": True,
            "purpose": "detect short-probe EMA lag before judging model failure",
        },
        "full_validation_at": int("$LOCAL_FULL_VALIDATION_AT"),
        "execution_device": "local RTX4090; never auto-submit evaluation to QZ",
    },
    "output_root": "$OUTPUT_ROOT",
    "record_root": "$RECORD_ROOT",
    "snapshot_root": "$SNAPSHOT_ROOT",
    "runner": "$RUNNER",
}
path = pathlib.Path("$RECORD_ROOT/preflight.json")
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

cat >"$RECORD_ROOT/local_eval_contract.json" <<EOF
{
  "schema": "ver3_1_local_eval_contract_v2",
  "checkpoint_root": "$OUTPUT_ROOT",
  "mode": "no_text",
  "speaker_cfg_scale": $EVAL_SPEAKER_CFG_SCALE,
  "semantic_cfg_scale": $EVAL_SEMANTIC_CFG_SCALE,
  "sampling_steps": 20,
  "quick20_steps": [$(printf '%s' "$LOCAL_QUICK20_STEPS" | sed 's/,/, /g')],
  "trigger": "wait for step-XXXXXX.ready.json, then load its inference_checkpoint",
  "primary": {
    "weights": "ema",
    "use_ema": true,
    "required_every_checkpoint": true
  },
  "diagnostic": {
    "weights": "raw",
    "use_ema": false,
    "required_every_checkpoint": true,
    "purpose": "detect short-probe EMA lag before judging model failure"
  },
  "full_validation_step": $LOCAL_FULL_VALIDATION_AT,
  "execution_device": "local RTX4090"
}
EOF

echo "[batch47-submit] job=$JOB_NAME"
echo "[batch47-submit] resource=MTTS-3-2-0715 spec=$SPEC instances=1 gpus=8"
echo "[batch47-submit] data=v1/no_text rows=$NO_TEXT_ROWS channel_stats=$CHANNEL_STATS_READY"
echo "[batch47-submit] train=gbs$GLOBAL_BATCH steps=$STEPS warmup=$WARMUP_STEPS save=$SAVE_STEPS"
echo "[batch47-submit] cfm=$T_SAMPLING(power=$T_SHIFT_POWER)/$LOSS_WEIGHTING eps=$LOSS_WEIGHT_EPS cap=$LOSS_WEIGHT_CAP"
echo "[batch47-submit] dropout=speaker$SPEAKER_DROPOUT semantic$SEMANTIC_DROPOUT aux=$AUX_LOSS_WEIGHT/$AUX_WARMUP_STEPS"
echo "[batch47-submit] speaker_prompts=$NUM_SPEAKER_PROMPT_TOKENS speaker_condition_scale=$SPEAKER_CONDITION_SCALE speaker_input_scale=$SPEAKER_INPUT_SCALE cfg=speaker$EVAL_SPEAKER_CFG_SCALE semantic$EVAL_SEMANTIC_CFG_SCALE"
echo "[batch47-submit] launch_ready=$LAUNCH_READY preflight=$RECORD_ROOT/preflight.json"

if [ "$DRY_RUN" = "1" ]; then
  printf 'job_name\tcompute_group\tspec\tinstances\tgpus\toutput_root\trunner\tlaunch_ready\n%s\t%s\t%s\t1\t8\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$COMPUTE_GROUP" "$SPEC" "$OUTPUT_ROOT" "$RUNNER" "$LAUNCH_READY" \
    >"$RECORD_ROOT/submission_plan.tsv"
  echo "[batch47-submit] DRY_RUN=1; qzcli was not called and no job was created"
  exit 0
fi

LOCK="$ROOT/trainset/qz_jobs/.${BATCH_ID}.live_submission.lock"
mkdir "$LOCK" 2>/dev/null || die "live submission lock exists: $LOCK"
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
LEDGER="$ROOT/trainset/qz_jobs/${BATCH_ID}.submitted_jobs.tsv"
[ ! -s "$LEDGER" ] || die "submission ledger already exists: $LEDGER"
SUBMIT_OUT="$RECORD_ROOT/submit_output.txt"
set +e
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
  QZCLI_GPU_TYPE_OVERRIDE="$GPU_TYPE" "$QZCLI" create-job \
  --name "$JOB_NAME" \
  --workspace "$WORKSPACE" \
  --project "$PROJECT" \
  --compute-group "$COMPUTE_GROUP" \
  --spec "$SPEC" \
  --framework "$FRAMEWORK" \
  --instances 1 \
  --shm "$SHM_GI" \
  --priority "$PRIORITY" \
  --image "$IMAGE" \
  --image-type "$IMAGE_TYPE" \
  --command "bash $RUNNER" >"$SUBMIT_OUT" 2>&1
SUBMIT_STATUS=$?
set -e
cat "$SUBMIT_OUT"
[ "$SUBMIT_STATUS" -eq 0 ] || die "QZ submission failed; see $SUBMIT_OUT"
JOB_ID="$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$SUBMIT_OUT" | tail -n1 || true)"
[ -n "$JOB_ID" ] || die "QZ accepted submission but no job ID could be parsed"
printf 'job_name\tjob_id\tcompute_group\tspec\tinstances\tgpus\toutput_root\trunner\n%s\t%s\t%s\t%s\t1\t8\t%s\t%s\n' \
  "$JOB_NAME" "$JOB_ID" "$COMPUTE_GROUP" "$SPEC" "$OUTPUT_ROOT" "$RUNNER" \
  >"$LEDGER"
cp -p "$LEDGER" "$RECORD_ROOT/submitted_jobs.tsv"
echo "[batch47-submit] submitted job_id=$JOB_ID"
