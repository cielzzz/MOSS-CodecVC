#!/usr/bin/env bash
# Resumable local / single-node 8-way runner for v2 four-edge balanced cleaning.
#
# Default MODE=smoke is intentionally small and writes a separate root.  Use
# MODE=full only after the smoke completion marker has been reviewed.  This
# script changes no training variable and never invokes Batch-41/42 machinery.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
CLEANER="${CLEANER:-$ROOT/scripts/004134_clean_v2_four_edge_balanced.py}"
MODE="${MODE:-smoke}"
SMOKE_ROWS="${SMOKE_ROWS:-32}"
NUM_SHARDS="${NUM_SHARDS:-1}"
GPUS="${GPUS:-0}"
DEVICE="${DEVICE:-cuda}"
ECAPA_LOG_EVERY="${ECAPA_LOG_EVERY:-1000}"
WAVLM_BATCH_SIZE="${WAVLM_BATCH_SIZE:-8}"
AUDIT_SIZE="${AUDIT_SIZE:-32}"
AUDIT_SEED="${AUDIT_SEED:-20260715}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-0}"
FORCE="${FORCE:-0}"

CANONICAL_OUTPUT="$ROOT/trainset/ver2_9_prepared_v2_four_edge_balanced_20260715"
SMOKE_OUTPUT="$ROOT/trainset/ver2_9_prepared_v2_four_edge_balanced_20260715_smoke${SMOKE_ROWS}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

if [ ! -x "$PY" ]; then
    die "Python is not executable: $PY"
fi
if [ ! -f "$CLEANER" ]; then
    die "cleaner is missing: $CLEANER"
fi
if ! [[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
    die "NUM_SHARDS must be a positive integer, got $NUM_SHARDS"
fi

case "$MODE" in
    smoke)
        MAX_ROWS="$SMOKE_ROWS"
        OUTPUT_ROOT="${OUTPUT_ROOT:-$SMOKE_OUTPUT}"
        VERIFY_ARGS=(--no-verify-input-sha256)
        COMPLETION_NAME="SMOKE_COMPLETED.json"
        ;;
    full)
        MAX_ROWS=0
        OUTPUT_ROOT="${OUTPUT_ROOT:-$CANONICAL_OUTPUT}"
        VERIFY_ARGS=(--verify-input-sha256)
        COMPLETION_NAME="COMPLETED.json"
        ;;
    *) die "MODE must be smoke or full, got $MODE" ;;
esac

if [ "$MODE" = "full" ] && [ "$NUM_SHARDS" -ne 8 ]; then
    die "full production cleaning requires NUM_SHARDS=8 for real 8-way processing"
fi
if [ "$MODE" = "full" ] && [ "$OUTPUT_ROOT" != "$CANONICAL_OUTPUT" ] && [ "${ALLOW_NONCANONICAL_FULL_OUTPUT:-0}" != "1" ]; then
    die "full output must remain $CANONICAL_OUTPUT unless ALLOW_NONCANONICAL_FULL_OUTPUT=1"
fi

read -r -a GPU_LIST <<<"$GPUS"
if [ "${#GPU_LIST[@]}" -lt "$NUM_SHARDS" ]; then
    die "need at least NUM_SHARDS GPU ids: GPUS=$GPUS NUM_SHARDS=$NUM_SHARDS"
fi

if [ ! -d "$ROOT" ]; then
    die "ROOT does not exist: $ROOT"
fi

mkdir -p "$OUTPUT_ROOT/logs"
LOG_ROOT="$OUTPUT_ROOT/logs/runner_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_ROOT"

FORCE_ARGS=()
if [ "$FORCE" = "1" ]; then
    FORCE_ARGS=(--force)
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[four-edge-runner] mode=$MODE output=$OUTPUT_ROOT shards=$NUM_SHARDS gpus=$GPUS"
echo "[four-edge-runner] cleaner=$CLEANER python=$PY log_root=$LOG_ROOT"

cd "$ROOT"

"$PY" "$CLEANER" --stage prepare \
    --max-rows "$MAX_ROWS" \
    --output-root "$OUTPUT_ROOT" \
    "${VERIFY_ARGS[@]}" \
    "${FORCE_ARGS[@]}" \
    --log-every "$ECAPA_LOG_EVERY" |& tee "$LOG_ROOT/prepare.log"

KEEPALIVE_PIDS=()
stop_keepalive() {
    for pid in "${KEEPALIVE_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${KEEPALIVE_PIDS[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap stop_keepalive EXIT INT TERM

if [ "$GPU_KEEPALIVE" = "1" ]; then
    for ((index=0; index<NUM_SHARDS; index++)); do
        gpu="${GPU_LIST[$index]}"
        (
            CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -c '
import time
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable for keepalive")
while True:
    a = torch.randn((512, 512), device="cuda")
    _ = a @ a
    torch.cuda.synchronize()
    time.sleep(20)
'
        ) >"$LOG_ROOT/keepalive_gpu${gpu}.log" 2>&1 &
        KEEPALIVE_PIDS+=("$!")
    done
fi

run_sharded() {
    local stage="$1"
    shift
    local pids=()
    local status=0
    local index gpu log_path
    for ((index=0; index<NUM_SHARDS; index++)); do
        gpu="${GPU_LIST[$index]}"
        log_path="$LOG_ROOT/${stage}.shard$(printf '%02d' "$index").log"
        echo "[four-edge-runner] stage=$stage shard=$index gpu=$gpu log=$log_path"
        (
            CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$CLEANER" --stage "$stage" \
                --max-rows "$MAX_ROWS" \
                --output-root "$OUTPUT_ROOT" \
                --shard-index "$index" --shard-count "$NUM_SHARDS" \
                --device "$DEVICE" --log-every "$ECAPA_LOG_EVERY" \
                "${FORCE_ARGS[@]}" "$@"
        ) >"$log_path" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    if [ "$status" -ne 0 ]; then
        die "stage=$stage has failed shard(s); inspect $LOG_ROOT"
    fi
}

run_sharded prompt-ecapa
run_sharded ecapa-score

"$PY" -u "$CLEANER" --stage wavlm-plan \
    --max-rows "$MAX_ROWS" --output-root "$OUTPUT_ROOT" \
    --shard-count "$NUM_SHARDS" "${FORCE_ARGS[@]}" |& tee "$LOG_ROOT/wavlm-plan.log"
run_sharded wavlm-cache --batch-size "$WAVLM_BATCH_SIZE"
run_sharded wavlm-score

"$PY" -u "$CLEANER" --stage finalize \
    --max-rows "$MAX_ROWS" --output-root "$OUTPUT_ROOT" \
    --shard-count "$NUM_SHARDS" --log-every "$ECAPA_LOG_EVERY" \
    "${FORCE_ARGS[@]}" |& tee "$LOG_ROOT/finalize.log"

# Reserve GPU 0 for the final fresh-cache audit; its cache audit marker is the
# only route that is allowed to create COMPLETED.json / SMOKE_COMPLETED.json.
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PY" -u "$CLEANER" --stage fresh-audit \
    --max-rows "$MAX_ROWS" --output-root "$OUTPUT_ROOT" --device "$DEVICE" \
    --audit-size "$AUDIT_SIZE" --audit-seed "$AUDIT_SEED" \
    "${FORCE_ARGS[@]}" |& tee "$LOG_ROOT/fresh-audit.log"

"$PY" - "$OUTPUT_ROOT/$COMPLETION_NAME" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"completion marker missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "complete":
    raise SystemExit(f"completion marker not complete: {payload}")
contract = payload.get("completion_contract", {})
if not all(contract.get(key) is True for key in ("count_identity", "all_kept_dual_encoder_balanced")):
    raise SystemExit(f"completion contract failed: {contract}")
print(json.dumps({"status": "complete", "completion": str(path), "contract": contract}, ensure_ascii=False))
PY

echo "[four-edge-runner] completed mode=$MODE output=$OUTPUT_ROOT"
