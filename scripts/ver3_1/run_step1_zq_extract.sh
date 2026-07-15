#!/usr/bin/env bash
# Production-only, resumable 8-GPU runner for Batch-45 Step 1 zq extraction.
#
# The input identity and output contract are intentionally immutable.  A QZ
# submission wrapper supplies the SHA256 identities of the snapshotted code.
# This runner then gives one byte-range shard to each of the eight H200 GPUs,
# finalizes only after every worker succeeds, and independently verifies the
# dataset-level COMPLETED.json contract.

set -Eeuo pipefail

ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
EXTRACTOR="${EXTRACTOR:-$ROOT/scripts/ver3_1/extract_zq_targets.py}"
CONFIG="${CONFIG:-$ROOT/configs/remote_full.yaml}"
OUTPUT_ROOT="$ROOT/prepared/zq_targets_v1"
OUTPUT_PARENT="$ROOT/prepared"
RUN_CONTRACT_PATH="$OUTPUT_PARENT/zq_targets_v1.RUN_CONTRACT.json"
NO_TEXT_MANIFEST="$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
TEXT_MANIFEST="$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
CODEC_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-Audio-Tokenizer"

# Authoritative v1 identities.  These are the Path-X sequence manifests, not
# the v2 source-perturbed data and not repeat-expanded sampling specifications.
NO_TEXT_SHA256="c4b061f0a968e73710dc86d81478483a9195e8a053f510f09be7952d60c3d279"
NO_TEXT_BYTES=18048211813
NO_TEXT_ROWS=310420
NO_TEXT_FRAMES=31089741
TEXT_SHA256="c6632888d08e79382001909a65951d6ce7bab80d7fb585cf7729e0a9188a9a80"
TEXT_BYTES=2196087856
TEXT_ROWS=32419
TEXT_FRAMES=4008719
EXPECTED_UTTERANCES=342839
EXPECTED_FRAMES=35098460

# The production topology and tensor contract are hard locks.  Only the decode
# micro-batch is tunable because it does not change output semantics.
NUM_SHARDS=8
NUM_QUANTIZERS=32
LATENT_DIM=768
OUTPUT_DTYPE=float32
CODEC_DTYPE=float32
CODES_SOURCE=manifest
ZQ_BATCH_SIZE="${ZQ_BATCH_SIZE:-32}"
LOG_EVERY="${LOG_EVERY:-250}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"

# Set by submit_step1_zq_extract_qz.sh from its immutable record snapshot.
EXPECTED_RUNNER_SHA256="${EXPECTED_RUNNER_SHA256:-}"
EXPECTED_EXTRACTOR_SHA256="${EXPECTED_EXTRACTOR_SHA256:-}"
EXPECTED_CONFIG_SHA256="${EXPECTED_CONFIG_SHA256:-}"
EXPECTED_MOSS_CODEC_SHA256="${EXPECTED_MOSS_CODEC_SHA256:-}"
EXPECTED_CONFIG_MODULE_SHA256="${EXPECTED_CONFIG_MODULE_SHA256:-}"
SOURCE_GIT_SHA="${SOURCE_GIT_SHA:-unknown}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

is_sha256() {
    [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

require_exact_sha() {
    local label="$1" path="$2" expected="$3" actual
    is_sha256 "$expected" || die "$label expected SHA256 was not supplied by the submission snapshot"
    [ -s "$path" ] || die "$label is missing or empty: $path"
    actual="$(sha256_file "$path")"
    [ "$actual" = "$expected" ] || die "$label SHA256 mismatch: expected=$expected actual=$actual path=$path"
}

[[ "$ZQ_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "ZQ_BATCH_SIZE must be a positive integer"
[[ "$LOG_EVERY" =~ ^[1-9][0-9]*$ ]] || die "LOG_EVERY must be a positive integer"
[[ "$HEARTBEAT_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "HEARTBEAT_SECONDS must be a positive integer"
[ -x "$PY" ] || die "Python is not executable: $PY"
[ -d "$ROOT" ] || die "project root is missing: $ROOT"
[ "$OUTPUT_ROOT" = "$ROOT/prepared/zq_targets_v1" ] || die "non-canonical output root is forbidden"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
EXTRACTOR="$(readlink -f "$EXTRACTOR")"
CONFIG="$(readlink -f "$CONFIG")"
EXTRACTOR_PROJECT_ROOT="$(cd "$(dirname "$EXTRACTOR")/../.." && pwd -P)"
MOSS_CODEC_MODULE="$EXTRACTOR_PROJECT_ROOT/moss_codecvc/moss_codec.py"
CONFIG_MODULE="$EXTRACTOR_PROJECT_ROOT/moss_codecvc/config.py"

require_exact_sha "runner" "$SCRIPT_PATH" "$EXPECTED_RUNNER_SHA256"
require_exact_sha "extractor" "$EXTRACTOR" "$EXPECTED_EXTRACTOR_SHA256"
require_exact_sha "config" "$CONFIG" "$EXPECTED_CONFIG_SHA256"
require_exact_sha "moss codec module" "$MOSS_CODEC_MODULE" "$EXPECTED_MOSS_CODEC_SHA256"
require_exact_sha "config module" "$CONFIG_MODULE" "$EXPECTED_CONFIG_MODULE_SHA256"

# The extractor and all remote-code/model loads must remain offline.
export PYTHONPATH="$EXTRACTOR_PROJECT_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/torch"
export XDG_CACHE_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

if [ -e "$OUTPUT_ROOT/COMPLETED.json" ] || [ -e "$OUTPUT_ROOT/VERIFIED_COMPLETED.json" ]; then
    die "canonical zq dataset is already completed; refusing a duplicate production run: $OUTPUT_ROOT"
fi

mkdir -p "$OUTPUT_PARENT" "$OUTPUT_ROOT"
RUN_LOCK="$OUTPUT_PARENT/.zq_targets_v1.RUNNING.lock"
if ! mkdir "$RUN_LOCK" 2>/dev/null; then
    die "another extraction (or an unaudited stale lock) owns $RUN_LOCK"
fi
printf 'host=%s\npid=%s\nstarted_utc=%s\n' "$(hostname)" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$RUN_LOCK/owner.txt"

WORKER_PIDS=()
KEEPALIVE_PIDS=()
HEARTBEAT_PID=""
HEARTBEAT_STOP="$RUN_LOCK/stop-heartbeat"
declare -A ACTIVE_WORKERS=()

cleanup() {
    local pid
    touch "$HEARTBEAT_STOP" 2>/dev/null || true
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    for pid in "${WORKER_PIDS[@]:-}" "${KEEPALIVE_PIDS[@]:-}"; do
        [ -n "$pid" ] || continue
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${WORKER_PIDS[@]:-}" "${KEEPALIVE_PIDS[@]:-}"; do
        [ -n "$pid" ] || continue
        wait "$pid" 2>/dev/null || true
    done
    rm -rf "$RUN_LOCK"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# A contractless non-empty directory could contain outputs from an ad-hoc or
# differently sharded extraction.  It is never safe to adopt those files.
CONTRACT_PATH="$RUN_CONTRACT_PATH"
EXTRACTOR_CONTRACT_PATH="$OUTPUT_ROOT/CONTRACT.json"
if [ ! -f "$EXTRACTOR_CONTRACT_PATH" ]; then
    shopt -s nullglob dotglob
    for entry in "$OUTPUT_ROOT"/*; do
        [ "$entry" = "$RUN_LOCK" ] && continue
        die "output root is non-empty but has no RUN_CONTRACT.json: $entry"
    done
    shopt -u nullglob dotglob
fi
if [ -f "$EXTRACTOR_CONTRACT_PATH" ] && [ ! -f "$CONTRACT_PATH" ]; then
    die "refusing to adopt preexisting extractor output without the matching Step 1 RUN_CONTRACT.json: $EXTRACTOR_CONTRACT_PATH"
fi

# Verify the one-node 8xH200 assumption and the complete local tokenizer
# checkpoint before spending time scanning the 20 GB manifests.
"$PY" - "$CONFIG" "$CODEC_ROOT" <<'PY'
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch

from moss_codecvc.config import deep_get, load_config

config_path = Path(sys.argv[1]).resolve()
codec_root = Path(sys.argv[2]).resolve()
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"Step 1 requires exactly 8 visible CUDA GPUs, got {torch.cuda.device_count()}")
gpu_names = [torch.cuda.get_device_name(index) for index in range(8)]
if any("H200" not in name.upper() for name in gpu_names):
    raise SystemExit(f"Step 1 is locked to eight H200 GPUs, got {gpu_names}")

config = load_config(config_path)
configured_codec = Path(str(deep_get(config, "moss.codec_path"))).resolve()
if configured_codec != codec_root:
    raise SystemExit(f"codec checkpoint mismatch: config={configured_codec} expected={codec_root}")
if int(deep_get(config, "moss.default_n_vq", -1)) != 32:
    raise SystemExit("remote_full.yaml must retain moss.default_n_vq=32")

required_sizes = {
    "config.json": 6601,
    "model.safetensors.index.json": 148113,
    "model-00001-of-00002.safetensors": 4998259168,
    "model-00002-of-00002.safetensors": 2100202560,
    "configuration_moss_audio_tokenizer.py": 12933,
    "modeling_moss_audio_tokenizer.py": 70906,
}
for relative, expected_size in required_sizes.items():
    path = codec_root / relative
    if not path.is_file() or path.stat().st_size != expected_size:
        actual = path.stat().st_size if path.exists() else None
        raise SystemExit(f"incomplete codec checkpoint: {path} expected_bytes={expected_size} actual={actual}")

index = json.loads((codec_root / "model.safetensors.index.json").read_text(encoding="utf-8"))
referenced = set(index.get("weight_map", {}).values())
expected_shards = {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
if referenced != expected_shards:
    raise SystemExit(f"codec index shard contract mismatch: {sorted(referenced)}")
for shard in sorted(referenced):
    with (codec_root / shard).open("rb") as handle:
        header_bytes = handle.read(8)
        if len(header_bytes) != 8:
            raise SystemExit(f"truncated safetensors header: {shard}")
        header_length = struct.unpack("<Q", header_bytes)[0]
        if not (0 < header_length < 64 * 1024 * 1024):
            raise SystemExit(f"invalid safetensors header length for {shard}: {header_length}")
        json.loads(handle.read(header_length))

print(json.dumps({"cuda_devices": gpu_names, "codec_checkpoint": str(codec_root), "status": "ok"}))
PY

# One streaming pass simultaneously proves SHA256, row count, and target-frame
# totals for both fixed manifests.  No cached metadata is trusted here.
MANIFEST_AUDIT_JSON="$({
    "$PY" - \
        "$NO_TEXT_MANIFEST" "$NO_TEXT_SHA256" "$NO_TEXT_BYTES" "$NO_TEXT_ROWS" "$NO_TEXT_FRAMES" \
        "$TEXT_MANIFEST" "$TEXT_SHA256" "$TEXT_BYTES" "$TEXT_ROWS" "$TEXT_FRAMES" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

frame_pattern = re.compile(br'"target_codec_frames"\s*:\s*([0-9]+)')
values = sys.argv[1:]
audits = []
for offset in range(0, len(values), 5):
    path = Path(values[offset]).resolve()
    expected_sha = values[offset + 1]
    expected_bytes = int(values[offset + 2])
    expected_rows = int(values[offset + 3])
    expected_frames = int(values[offset + 4])
    if not path.is_file() or path.stat().st_size != expected_bytes:
        actual = path.stat().st_size if path.exists() else None
        raise SystemExit(f"manifest size mismatch: {path} expected={expected_bytes} actual={actual}")
    digest = hashlib.sha256()
    rows = 0
    frames = 0
    with path.open("rb", buffering=16 * 1024 * 1024) as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            rows += 1
            matches = frame_pattern.findall(raw_line)
            if len(matches) != 1:
                raise SystemExit(f"expected one target_codec_frames at {path}:{rows}, got {len(matches)}")
            frames += int(matches[0])
    actual_sha = digest.hexdigest()
    if (actual_sha, rows, frames) != (expected_sha, expected_rows, expected_frames):
        raise SystemExit(
            f"manifest identity mismatch: {path} "
            f"sha={actual_sha}/{expected_sha} rows={rows}/{expected_rows} frames={frames}/{expected_frames}"
        )
    audits.append({
        "path": str(path), "sha256": actual_sha, "bytes": expected_bytes,
        "rows": rows, "frames": frames,
    })
print(json.dumps(audits, sort_keys=True))
PY
} )"

export MANIFEST_AUDIT_JSON
export CONTRACT_PATH OUTPUT_ROOT EXTRACTOR CONFIG SCRIPT_PATH MOSS_CODEC_MODULE CONFIG_MODULE
export EXPECTED_RUNNER_SHA256 EXPECTED_EXTRACTOR_SHA256 EXPECTED_CONFIG_SHA256
export EXPECTED_MOSS_CODEC_SHA256 EXPECTED_CONFIG_MODULE_SHA256 SOURCE_GIT_SHA
export ZQ_BATCH_SIZE LOG_EVERY HEARTBEAT_SECONDS
"$PY" - <<'PY'
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

path = Path(os.environ["CONTRACT_PATH"])
contract = {
    "schema_version": 1,
    "batch": "Batch-45 Step 1",
    "architecture": "ver3.1 DDLFM",
    "output_root": os.environ["OUTPUT_ROOT"],
    "inputs": json.loads(os.environ["MANIFEST_AUDIT_JSON"]),
    "expected_utterances": 342839,
    "expected_frames": 35098460,
    "num_shards": 8,
    "gpu_contract": "one MTTS-3-2-0715 node; 8xH200; one worker per GPU",
    "codes_source": "manifest",
    "num_quantizers": 32,
    "latent_dim": 768,
    "frame_rate_hz": 12.5,
    "codec_dtype": "float32",
    "output_dtype": "float32",
    "batch_size": int(os.environ["ZQ_BATCH_SIZE"]),
    "code_identity": {
        "source_git_sha": os.environ["SOURCE_GIT_SHA"],
        "runner_sha256": os.environ["EXPECTED_RUNNER_SHA256"],
        "extractor_sha256": os.environ["EXPECTED_EXTRACTOR_SHA256"],
        "config_sha256": os.environ["EXPECTED_CONFIG_SHA256"],
        "moss_codec_sha256": os.environ["EXPECTED_MOSS_CODEC_SHA256"],
        "config_module_sha256": os.environ["EXPECTED_CONFIG_MODULE_SHA256"],
    },
}
if path.exists():
    current = json.loads(path.read_text(encoding="utf-8"))
    if current != contract:
        raise SystemExit(f"mixed extraction contract at {path}: current={current!r} expected={contract!r}")
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
print(json.dumps(contract, ensure_ascii=False, sort_keys=True))
PY

# The extractor owns CONTRACT.json inside the canonical output.  If this is a
# resume, validate that its immutable low-level contract agrees with Step 1;
# on a fresh root the first worker will publish it under its own file lock.
"$PY" - "$EXTRACTOR_CONTRACT_PATH" "$NO_TEXT_MANIFEST" "$TEXT_MANIFEST" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(json.dumps({"status": "pending", "path": str(path)}))
    raise SystemExit(0)
contract = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(contract, dict):
    raise SystemExit(f"extractor contract is not an object: {path}")
inputs = contract.get("inputs")
if not isinstance(inputs, list):
    raise SystemExit(f"extractor contract has no inputs: {path}")
expected = {str(Path(sys.argv[2]).resolve()): "no_text", str(Path(sys.argv[3]).resolve()): "text"}
actual = {str(Path(item.get("manifest", "")).resolve()): str(item.get("split", "")) for item in inputs if isinstance(item, dict)}
if actual != expected:
    raise SystemExit(f"extractor input contract mismatch: {actual} != {expected}")
checks = {
    "codes_source": "manifest", "n_vq": 32, "output_dtype": "float32",
    "expected_dim": 768, "frame_rate_hz": 12.5, "num_shards": 8,
    "max_rows_per_input_per_shard": 0, "partial": False,
}
for key, value in checks.items():
    if contract.get(key) != value:
        raise SystemExit(f"extractor contract mismatch for {key}: {contract.get(key)!r} != {value!r}")
provenance = contract.get("codec_provenance") or {}
if int(provenance.get("max_quantizers", -1)) < 32 or int(provenance.get("latent_dim", -1)) != 768:
    raise SystemExit(f"extractor codec provenance mismatch: {provenance}")
print(json.dumps({"status": "validated", "path": str(path), "contract_sha256": contract.get("contract_sha256")}))
PY

# Reject shard markers from any topology other than the production 8-way one.
if [ -d "$OUTPUT_ROOT/_shards" ]; then
    while IFS= read -r marker; do
        [[ "$(basename "$marker")" == shard-?????-of-00008.COMPLETED.json ]] || \
            die "foreign shard contract found: $marker"
    done < <(find "$OUTPUT_ROOT/_shards" -maxdepth 1 -type f -name 'shard-*-of-*.COMPLETED.json' -print)
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
# Keep runner logs outside the extractor-owned output root.  On a fresh run the
# extractor must see an empty root so it can atomically create CONTRACT.json.
LOG_ROOT="$OUTPUT_PARENT/zq_targets_v1_logs/$RUN_ID"
mkdir -p "$LOG_ROOT"
cp -p "$CONTRACT_PATH" "$LOG_ROOT/RUN_CONTRACT.json"
printf '%s\n' "$MANIFEST_AUDIT_JSON" >"$LOG_ROOT/manifest_audit.json"

echo "[ver3.1-step1] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
echo "[ver3.1-step1] output=$OUTPUT_ROOT shards=8 gpus=8 batch=$ZQ_BATCH_SIZE"
echo "[ver3.1-step1] expected_utterances=$EXPECTED_UTTERANCES expected_frames=$EXPECTED_FRAMES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Keep all eight allocated GPUs visibly alive during JSONL I/O and finalization.
# The 512x512 multiply every 20 seconds is intentionally tiny relative to H200.
for gpu in 0 1 2 3 4 5 6 7; do
    (
        CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u - <<'PY'
import time
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("keepalive requires exactly one visible GPU")
left = torch.randn((512, 512), device="cuda", dtype=torch.float16)
right = torch.randn((512, 512), device="cuda", dtype=torch.float16)
while True:
    torch.mm(left, right)
    torch.cuda.synchronize()
    time.sleep(20)
PY
    ) >"$LOG_ROOT/keepalive.gpu${gpu}.log" 2>&1 &
    KEEPALIVE_PIDS+=("$!")
done

# A heartbeat reports completed shards plus the latest extractor progress from
# every worker without repeatedly walking hundreds of thousands of .npy files.
(
    exec > >(tee -a "$LOG_ROOT/heartbeat.log") 2>&1
    while [ ! -e "$HEARTBEAT_STOP" ]; do
        sleep "$HEARTBEAT_SECONDS"
        [ ! -e "$HEARTBEAT_STOP" ] || break
        completed=0
        [ ! -d "$OUTPUT_ROOT/_shards" ] || \
            completed="$(find "$OUTPUT_ROOT/_shards" -maxdepth 1 -type f -name 'shard-?????-of-00008.COMPLETED.json' | wc -l)"
        echo "[ver3.1-step1-heartbeat] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) completed_shards=$completed/8"
        for shard in 0 1 2 3 4 5 6 7; do
            log="$LOG_ROOT/extract.shard$(printf '%02d' "$shard").log"
            [ -f "$log" ] || continue
            latest="$(grep -E '\[zq-extract\]' "$log" | tail -n 1 || true)"
            [ -z "$latest" ] || echo "[ver3.1-step1-heartbeat] $latest"
        done
        df -h "$OUTPUT_ROOT" | tail -n 1 | sed 's/^/[ver3.1-step1-heartbeat] disk /'
    done
) &
HEARTBEAT_PID="$!"

COMMON_ARGS=(
    extract
    --input "no_text=$NO_TEXT_MANIFEST"
    --input "text=$TEXT_MANIFEST"
    --output-root "$OUTPUT_ROOT"
    --config "$CONFIG"
    --codes-source "$CODES_SOURCE"
    --codec-dtype "$CODEC_DTYPE"
    --output-dtype "$OUTPUT_DTYPE"
    --n-vq "$NUM_QUANTIZERS"
    --expected-dim "$LATENT_DIM"
    --num-shards "$NUM_SHARDS"
    --batch-size "$ZQ_BATCH_SIZE"
    --log-every "$LOG_EVERY"
    --strict
)

for shard in 0 1 2 3 4 5 6 7; do
    log="$LOG_ROOT/extract.shard$(printf '%02d' "$shard").log"
    echo "[ver3.1-step1] launch shard=$shard gpu=$shard log=$log"
    (
        CUDA_VISIBLE_DEVICES="$shard" "$PY" -u "$EXTRACTOR" \
            "${COMMON_ARGS[@]}" --device cuda:0 --shard-id "$shard"
    ) >"$log" 2>&1 &
    worker_pid="$!"
    WORKER_PIDS+=("$worker_pid")
    ACTIVE_WORKERS["$worker_pid"]="$shard"
done

# wait -n fails fast: an early worker error stops the other seven through the
# EXIT trap instead of burning a whole node until the first sequential wait.
while [ "${#ACTIVE_WORKERS[@]}" -gt 0 ]; do
    finished_pid=""
    set +e
    wait -n -p finished_pid "${!ACTIVE_WORKERS[@]}"
    status=$?
    set -e
    finished_pid="${finished_pid:-}"
    [ -n "$finished_pid" ] || die "wait -n returned without a worker pid; inspect $LOG_ROOT"
    finished_shard="${ACTIVE_WORKERS[$finished_pid]:-unknown}"
    unset 'ACTIVE_WORKERS[$finished_pid]'
    if [ "$status" -ne 0 ]; then
        die "zq extraction worker failed: shard=$finished_shard pid=$finished_pid; inspect $LOG_ROOT"
    fi
    echo "[ver3.1-step1] worker complete shard=$finished_shard pid=$finished_pid remaining=${#ACTIVE_WORKERS[@]}"
done
WORKER_PIDS=()

for shard in 0 1 2 3 4 5 6 7; do
    marker="$OUTPUT_ROOT/_shards/shard-$(printf '%05d' "$shard")-of-00008.COMPLETED.json"
    [ -s "$marker" ] || die "worker exited without shard completion marker: $marker"
done

echo "[ver3.1-step1] all shards completed; finalizing"
"$PY" -u "$EXTRACTOR" finalize \
    --output-root "$OUTPUT_ROOT" --num-shards 8 \
    --expected-total-utterances "$EXPECTED_UTTERANCES" \
    --expected-total-frames "$EXPECTED_FRAMES" |& tee "$LOG_ROOT/finalize.log"

# The extractor owns COMPLETED.json creation; this independent verifier enforces
# the stronger production totals and 32-quantizer per-record contract.
if ! "$PY" - "$OUTPUT_ROOT" "$NO_TEXT_MANIFEST" "$TEXT_MANIFEST" <<'PY' |& tee "$LOG_ROOT/verify_completed.log"; then
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected_inputs = {str(Path(sys.argv[2]).resolve()), str(Path(sys.argv[3]).resolve())}
completion_path = root / "COMPLETED.json"
if not completion_path.is_file():
    raise SystemExit(f"missing completion marker: {completion_path}")
completion = json.loads(completion_path.read_text(encoding="utf-8"))
expected_scalars = {
    "status": "completed", "num_shards": 8, "codes_source": "manifest",
    "latent_dim": 768, "dtype": "float32", "total_utterances": 342839,
    "total_frames": 35098460, "errors": 0,
}
for key, expected in expected_scalars.items():
    if completion.get(key) != expected:
        raise SystemExit(f"COMPLETED.json mismatch for {key}: {completion.get(key)!r} != {expected!r}")
if float(completion.get("frame_rate_hz", -1)) != 12.5:
    raise SystemExit("COMPLETED.json frame rate must be 12.5 Hz")
input_paths = {str(Path(item["manifest"]).resolve()) for item in completion.get("inputs", [])}
if input_paths != expected_inputs:
    raise SystemExit(f"COMPLETED.json inputs mismatch: {input_paths} != {expected_inputs}")

counts: Counter[str] = Counter()
frames: defaultdict[str, int] = defaultdict(int)
quantizers: Counter[int] = Counter()
records_path = root / "manifest.jsonl"
with records_path.open("r", encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        split = str(row["split"])
        counts[split] += 1
        frames[split] += int(row["num_frames"])
        quantizers[int(row["num_quantizers"])] += 1
        output = Path(row["output_path"]).resolve()
        if root not in output.parents:
            raise SystemExit(f"record escapes canonical output root at line {line_no}: {output}")

expected_counts = {"no_text": 310420, "text": 32419}
expected_frames = {"no_text": 31089741, "text": 4008719}
if dict(counts) != expected_counts or dict(frames) != expected_frames:
    raise SystemExit(
        f"split totals mismatch: counts={dict(counts)}/{expected_counts} "
        f"frames={dict(frames)}/{expected_frames}"
    )
if dict(quantizers) != {32: 342839}:
    raise SystemExit(f"all records must use 32 quantizers, got {dict(quantizers)}")

verified = {
    "schema_version": 1,
    "status": "verified_completed",
    "completion_path": str(completion_path),
    "completion_sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
    "total_utterances": 342839,
    "total_frames": 35098460,
    "split_utterances": expected_counts,
    "split_frames": expected_frames,
    "num_quantizers": 32,
    "latent_dim": 768,
    "dtype": "float32",
    "frame_rate_hz": 12.5,
}
target = root / "VERIFIED_COMPLETED.json"
fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=root)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        json.dump(verified, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
finally:
    Path(temporary).unlink(missing_ok=True)
print(json.dumps(verified, ensure_ascii=False, sort_keys=True))
PY
    invalid="$OUTPUT_ROOT/INVALID_COMPLETED.$(date -u +%Y%m%dT%H%M%SZ).json"
    mv "$OUTPUT_ROOT/COMPLETED.json" "$invalid" 2>/dev/null || true
    die "dataset-level completion verification failed; marker quarantined as $invalid"
fi

echo "[ver3.1-step1] COMPLETED and VERIFIED_COMPLETED accepted at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
