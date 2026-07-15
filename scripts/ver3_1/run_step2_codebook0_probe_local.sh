#!/usr/bin/env bash
# Local RTX4090-only runner for Batch-45 Step 2.  This is an evaluation probe,
# not a long-running training job; it intentionally never calls qzcli.
set -Eeuo pipefail

ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
SCRIPT="$ROOT/scripts/ver3_1/run_step2_codebook0_probe.py"
OUTPUT_ROOT="$ROOT/testset/outputs/ver3_1_step2_codebook0_probe_20260715"
DATA_ROOT="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/datasets/seed-tts-eval/seedtts_testset"
CODEC_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-Audio-Tokenizer"
LOG_ROOT="$OUTPUT_ROOT/logs"

[ -x "$PY" ] || { echo "missing Python: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "missing probe script: $SCRIPT" >&2; exit 2; }
[ -d "$DATA_ROOT" ] || { echo "missing official Seed-TTS-Eval root: $DATA_ROOT" >&2; exit 2; }
[ -d "$CODEC_ROOT" ] || { echo "missing codec checkpoint: $CODEC_ROOT" >&2; exit 2; }

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

if [ -e "$OUTPUT_ROOT/COMPLETED.json" ]; then
    echo "Step 2 is already completed; refusing to overwrite $OUTPUT_ROOT" >&2
    exit 3
fi
if [ -d "$OUTPUT_ROOT/workers" ] && find "$OUTPUT_ROOT/workers" -type f -name '*.jsonl' -print -quit | grep -q .; then
    echo "partial worker outputs exist under $OUTPUT_ROOT; archive them before retry" >&2
    exit 3
fi

"$PY" "$SCRIPT" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
    --codec-root "$CODEC_ROOT" --prepare-only
mkdir -p "$LOG_ROOT"

PIDS=()
for worker in 0 1; do
    device="cuda:${worker}"
    "$PY" "$SCRIPT" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
        --codec-root "$CODEC_ROOT" --encode --worker-id "$worker" --num-workers 2 \
        --device "$device" >"$LOG_ROOT/worker-${worker}.log" 2>&1 &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
    echo "one or more Step 2 encoding workers failed; inspect $LOG_ROOT" >&2
    exit "$status"
fi

"$PY" "$SCRIPT" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
    --codec-root "$CODEC_ROOT" | tee "$LOG_ROOT/aggregate.json"

echo "Step 2 completed: $OUTPUT_ROOT"
