#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
cd "$ROOT"

PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"

SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/VC_train/v2_real_target_no_text_300k_zh_en_balanced_20260707_seedvc_triples}"
TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-$SOURCE_DATA_DIR/no_text.train.refdecorr.train_minus_valid.manifest.jsonl}"
VALID_ROOT="${VALID_ROOT:-$SOURCE_DATA_DIR/valid_ref_channel_heldout_2k_20260708}"
VALID_SAME_INPUT_JSONL="${VALID_SAME_INPUT_JSONL:-$VALID_ROOT/same_episode_near_original_valid.manifest.jsonl}"
VALID_CROSS_INPUT_JSONL="${VALID_CROSS_INPUT_JSONL:-$VALID_ROOT/heldout_refdecorr_cross_channel_valid.manifest.jsonl}"

WORK_ROOT="${WORK_ROOT:-$ROOT/trainset/v2_real_target_no_text_refdecorr_20260708}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_v2_data_20260708}"
TEXT_PREPARED_DIR="${TEXT_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_20260707}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"

SPM_MODEL="${SPM_MODEL:-$ROOT/trainset/shared_content_tokenizers/spm_multilingual_byte_fallback_v1.model}"
SPM_META="${SPM_META:-$ROOT/trainset/shared_content_tokenizers/spm_multilingual_byte_fallback_v1.json}"
TOKENIZER_ID="${TOKENIZER_ID:-spm_multilingual_byte_fallback_v1}"

N_VQ="${N_VQ:-32}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CODEC_GPU_IDS="${CODEC_GPU_IDS:-$GPU_IDS}"
SPEAKER_GPU_IDS="${SPEAKER_GPU_IDS:-$GPU_IDS}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
ECAPA_SHARD_COUNT="${ECAPA_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
WAVLM_BNF_SHARD_COUNT="${WAVLM_BNF_SHARD_COUNT:-8}"
WAVLM_SV_SHARD_COUNT="${WAVLM_SV_SHARD_COUNT:-8}"
WAVLM_DEVICES="${WAVLM_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
WAVLM_SV_DEVICES="${WAVLM_SV_DEVICES:-$WAVLM_DEVICES}"

WAVLM_CACHE_DIR="${WAVLM_CACHE_DIR:-$DOWNLOAD_ROOT/huggingface}"
DEFAULT_WAVLM_MODEL="$(find "$WAVLM_CACHE_DIR/hub/models--microsoft--wavlm-base-plus/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | grep -m1 . || true)"
if [ -z "$DEFAULT_WAVLM_MODEL" ]; then
  DEFAULT_WAVLM_MODEL="$(find "$WAVLM_CACHE_DIR/models--microsoft--wavlm-base-plus/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | grep -m1 . || true)"
fi
if [ -z "$DEFAULT_WAVLM_MODEL" ] || [ ! -f "$DEFAULT_WAVLM_MODEL/config.json" ]; then
  DEFAULT_WAVLM_MODEL="microsoft/wavlm-base-plus"
fi
WAVLM_MODEL="${WAVLM_MODEL:-$DEFAULT_WAVLM_MODEL}"
WAVLM_LAYER="${WAVLM_LAYER:-9}"
WAVLM_DTYPE="${WAVLM_DTYPE:-auto}"
WAVLM_SAVE_DTYPE="${WAVLM_SAVE_DTYPE:-float16}"
WAVLM_DOWNSAMPLE_STRIDE="${WAVLM_DOWNSAMPLE_STRIDE:-1}"
WAVLM_LOCAL_FILES_ONLY="${WAVLM_LOCAL_FILES_ONLY:-1}"

WAVLM_SV_MODEL="${WAVLM_SV_MODEL:-microsoft/wavlm-base-plus-sv}"
WAVLM_SV_BATCH_SIZE="${WAVLM_SV_BATCH_SIZE:-32}"
WAVLM_SV_LOCAL_FILES_ONLY="${WAVLM_SV_LOCAL_FILES_ONLY:-1}"

FORCE="${FORCE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"
WAIT_HEARTBEAT_SECS="${WAIT_HEARTBEAT_SECS:-60}"

export HF_HOME="$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="$DOWNLOAD_ROOT/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="$DOWNLOAD_ROOT/cache"
export TOKENIZERS_PARALLELISM=false
export DISABLE_SAFETENSORS_CONVERSION=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if ! truthy "$DRY_RUN"; then
    "$@"
  fi
}

csv_item() {
  local csv="$1"
  local index="$2"
  IFS=',' read -r -a items <<< "$csv"
  if [ "${#items[@]}" -eq 0 ]; then
    echo "0"
    return 0
  fi
  echo "${items[$((index % ${#items[@]}))]}"
}

stage_done() {
  local path="$1"
  if truthy "$SKIP_EXISTING" && [ "$FORCE" != "1" ] && [ -s "$path" ]; then
    return 0
  fi
  return 1
}

wait_for_pids() {
  local failed=0
  local heartbeat_pid=0
  local heartbeat_label="${1:-parallel}"
  shift || true
  if [ "$WAIT_HEARTBEAT_SECS" -gt 0 ] && ! truthy "$DRY_RUN"; then
    (
      while true; do
        sleep "$WAIT_HEARTBEAT_SECS" || exit 0
        printf '[heartbeat] %s %s pids=' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$heartbeat_label"
        printf ' %s' "$@"
        printf '\n'
      done
    ) &
    heartbeat_pid=$!
  fi
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$heartbeat_pid" -ne 0 ]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  if [ "$failed" -ne 0 ]; then
    echo "ERROR: one or more $heartbeat_label workers failed" >&2
    exit 1
  fi
}

merge_modulo_shards() {
  local output_jsonl="$1"
  local shard_dir="$2"
  local num_shards="$3"
  local tmp_path="$output_jsonl.tmp"
  : > "$tmp_path"
  local shard
  for shard in $(seq 0 $((num_shards - 1))); do
    local part
    part=$(printf "%s/shard_%05d.jsonl" "$shard_dir" "$shard")
    if [ ! -s "$part" ]; then
      echo "ERROR: missing shard output: $part" >&2
      rm -f "$tmp_path"
      exit 1
    fi
    cat "$part" >> "$tmp_path"
  done
  mv "$tmp_path" "$output_jsonl"
}

run_wavlm_bnf_source() {
  local label="$1"
  local input_jsonl="$2"
  local output_jsonl="$3"
  local feature_root="$4"
  if stage_done "$output_jsonl"; then
    echo "[skip] WavLM-BNF source exists label=$label output=$output_jsonl"
    return 0
  fi
  if [ "$OVERWRITE" = "1" ]; then
    rm -rf "$output_jsonl.shards" "$output_jsonl" "$output_jsonl.tmp" "$output_jsonl.summary.json" "$output_jsonl.done.json"
  fi
  mkdir -p "$output_jsonl.shards" "$output_jsonl.shards/logs" "$feature_root"
  pids=()
  for shard in $(seq 0 $((WAVLM_BNF_SHARD_COUNT - 1))); do
    shard_out=$(printf "%s.shards/shard_%05d.jsonl" "$output_jsonl" "$shard")
    shard_log=$(printf "%s.shards/logs/shard_%05d.log" "$output_jsonl" "$shard")
    shard_device="$(csv_item "$WAVLM_DEVICES" "$shard")"
    echo "[wavlm-bnf] label=$label shard=$shard/$WAVLM_BNF_SHARD_COUNT device=$shard_device out=$shard_out"
    (
      "$PY" scripts/001020_extract_hubert_semantic_features.py \
        --input-jsonl "$input_jsonl" \
        --output-jsonl "$shard_out" \
        --feature-root "$feature_root" \
        --extractor wavlm \
        --model-name-or-path "$WAVLM_MODEL" \
        --cache-dir "$WAVLM_CACHE_DIR" \
        --source source \
        --layer "$WAVLM_LAYER" \
        --dtype "$WAVLM_DTYPE" \
        --save-dtype "$WAVLM_SAVE_DTYPE" \
        --downsample-stride "$WAVLM_DOWNSAMPLE_STRIDE" \
        --use-safetensors false \
        --progress-every "$PROGRESS_EVERY" \
        --reuse-existing-features \
        --device "$shard_device" \
        --num-shards "$WAVLM_BNF_SHARD_COUNT" \
        --shard-index "$shard" \
        $( [ "$WAVLM_LOCAL_FILES_ONLY" = "1" ] && printf '%s' '--local-files-only' ) \
        $( [ "$OVERWRITE" = "1" ] && printf '%s' '--overwrite' )
    ) > "$shard_log" 2>&1 &
    pids+=("$!")
  done
  wait_for_pids "wavlm-bnf-$label" "${pids[@]}"
  merge_modulo_shards "$output_jsonl" "$output_jsonl.shards" "$WAVLM_BNF_SHARD_COUNT"
  "$PY" - "$output_jsonl" "$WAVLM_BNF_SHARD_COUNT" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
num_shards = int(sys.argv[2])
rows = sum(1 for _ in path.open("r", encoding="utf-8"))
payload = {"status": "complete", "output_jsonl": str(path), "rows": rows, "num_shards": num_shards}
path.with_suffix(path.suffix + ".summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
path.with_name(path.name + ".done.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[wavlm-bnf] merged {path} rows={rows}", flush=True)
PY
}

run_wavlm_sv_speaker_vec() {
  local label="$1"
  local input_jsonl="$2"
  local output_jsonl="$3"
  local speaker_vec_dir="$4"
  if stage_done "$output_jsonl"; then
    echo "[skip] WavLM-SV speaker vec exists label=$label output=$output_jsonl"
    return 0
  fi
  if [ "$OVERWRITE" = "1" ]; then
    rm -rf "$output_jsonl.shards" "$output_jsonl" "$output_jsonl.tmp" "$output_jsonl.summary.json" "$output_jsonl.done.json"
  fi
  mkdir -p "$output_jsonl.shards" "$output_jsonl.shards/logs" "$speaker_vec_dir"
  pids=()
  for shard in $(seq 0 $((WAVLM_SV_SHARD_COUNT - 1))); do
    shard_out=$(printf "%s.shards/shard_%05d.jsonl" "$output_jsonl" "$shard")
    shard_log=$(printf "%s.shards/logs/shard_%05d.log" "$output_jsonl" "$shard")
    shard_device="$(csv_item "$WAVLM_SV_DEVICES" "$shard")"
    echo "[speaker-vec] label=$label shard=$shard/$WAVLM_SV_SHARD_COUNT device=$shard_device out=$shard_out"
    (
      "$PY" scripts/002022_precompute_ver2_9_speaker_vecs.py \
        --input-jsonl "$input_jsonl" \
        --output-jsonl "$shard_out" \
        --speaker-vec-dir "$speaker_vec_dir" \
        --audio-key timbre_ref_audio \
        --speaker-encoder-path "$WAVLM_SV_MODEL" \
        --speaker-embedding-dim 512 \
        --device "$shard_device" \
        --batch-size "$WAVLM_SV_BATCH_SIZE" \
        --num-shards "$WAVLM_SV_SHARD_COUNT" \
        --shard-index "$shard" \
        $( [ "$WAVLM_SV_LOCAL_FILES_ONLY" = "1" ] && printf '%s' '--local-files-only' ) \
        $( [ "$OVERWRITE" = "1" ] && printf '%s' '--overwrite' )
    ) > "$shard_log" 2>&1 &
    pids+=("$!")
  done
  wait_for_pids "speaker-vec-$label" "${pids[@]}"
  merge_modulo_shards "$output_jsonl" "$output_jsonl.shards" "$WAVLM_SV_SHARD_COUNT"
  "$PY" - "$output_jsonl" "$WAVLM_SV_SHARD_COUNT" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
num_shards = int(sys.argv[2])
rows = 0
missing = 0
for line in path.open("r", encoding="utf-8"):
    if not line.strip():
        continue
    rows += 1
    row = json.loads(line)
    vec = row.get("speaker_vec_path")
    if not vec or not Path(str(vec)).exists():
        missing += 1
payload = {"status": "complete", "output_jsonl": str(path), "rows": rows, "missing_speaker_vec": missing, "num_shards": num_shards}
path.with_suffix(path.suffix + ".summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
path.with_name(path.name + ".done.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[speaker-vec] merged {path} rows={rows} missing={missing}", flush=True)
if missing:
    raise SystemExit(f"missing speaker_vec_path for {missing} rows")
PY
}

filter_ref_content_leakage() {
  local label="$1"
  local input_jsonl="$2"
  local output_jsonl="$3"
  local dropped_jsonl="$4"
  if [ ! -s "$input_jsonl" ]; then
    echo "ERROR: missing input for ref-content leakage filter label=$label input=$input_jsonl" >&2
    exit 1
  fi
  run_cmd "$PY" scripts/002039_filter_v2_real_no_text_ref_content_leak.py \
    --input-jsonl "$input_jsonl" \
    --output-jsonl "$output_jsonl" \
    --dropped-jsonl "$dropped_jsonl" \
    --summary-json "$output_jsonl.summary.json" \
    --overwrite
}

process_split() {
  local label="$1"
  local input_jsonl="$2"
  local dataset_name="$3"
  local final_name="$4"
  local dataset_root="$WORK_ROOT/$dataset_name"
  local train_ready_jsonl="$dataset_root/sft/moss_codecvc_sft.$dataset_name.with_light_ecapa_spk.with_prosody.jsonl"
  local known_jsonl="$dataset_root/sft/moss_codecvc_sft.$dataset_name.with_light_ecapa_spk.with_prosody.with_known_content.jsonl"
  local content_jsonl="$dataset_root/sft/moss_codecvc_sft.$dataset_name.with_light_ecapa_spk.with_prosody.with_known_content.with_spm_content_tokens.jsonl"
  local wavlm_jsonl="$dataset_root/sft/moss_codecvc_sft.$dataset_name.with_light_ecapa_spk.with_prosody.with_known_content.with_spm_content_tokens.with_wavlm_bnf_source.jsonl"
  local prepared_jsonl="$PREPARED_DIR/$final_name"

  echo "=========================================="
  echo "[v2-real-no-text] process label=$label"
  echo "  input=$input_jsonl"
  echo "  dataset_root=$dataset_root"
  echo "  prepared_jsonl=$prepared_jsonl"
  echo "=========================================="
  if [ ! -s "$input_jsonl" ]; then
    echo "ERROR: missing input JSONL for $label: $input_jsonl" >&2
    exit 1
  fi

  run_cmd env \
    PY="$PY" \
    DATASET_NAME="$dataset_name" \
    DATASET_ROOT="$dataset_root" \
    INPUT_JSONL="$input_jsonl" \
    EMIT_MODES=no_text \
    N_VQ="$N_VQ" \
    GPU_IDS="$GPU_IDS" \
    CODEC_GPU_IDS="$CODEC_GPU_IDS" \
    SPEAKER_GPU_IDS="$SPEAKER_GPU_IDS" \
    CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT" \
    SPEAKER_SHARD_COUNT="$ECAPA_SHARD_COUNT" \
    PROSODY_SHARD_COUNT="$PROSODY_SHARD_COUNT" \
    RUN_PROSODY_FEATURES=1 \
    INCLUDE_TARGET_PROSODY=1 \
    SKIP_EXISTING="$SKIP_EXISTING" \
    FORCE="$FORCE" \
    WAIT_HEARTBEAT_SECS="$WAIT_HEARTBEAT_SECS" \
    WRITE_TRAIN_COMMAND=0 \
    bash scripts/001025_run_train_ready_no_text_ver2.sh

  run_cmd "$PY" scripts/002033_mark_v2_real_no_text_known_content.py \
    --input-jsonl "$train_ready_jsonl" \
    --output-jsonl "$known_jsonl" \
    --progress-every "$PROGRESS_EVERY" \
    --overwrite

  run_cmd "$PY" scripts/001046_extract_multilingual_content_tokens.py \
    --input-jsonl "$known_jsonl" \
    --output-jsonl "$content_jsonl" \
    --spm-model "$SPM_MODEL" \
    --tokenizer-meta "$SPM_META" \
    --tokenizer-id "$TOKENIZER_ID" \
    --text-keys content_ref_text,source_text,target_text \
    --progress-every "$PROGRESS_EVERY" \
    --overwrite

  if ! truthy "$DRY_RUN"; then
    run_wavlm_bnf_source "$label" "$content_jsonl" "$wavlm_jsonl" "$dataset_root/semantic_features/wavlm_bnf"
    run_wavlm_sv_speaker_vec "$label" "$wavlm_jsonl" "$prepared_jsonl" "$PREPARED_DIR/speaker_vecs/$label"
    case "$label" in
      train)
        filter_ref_content_leakage \
          "$label" \
          "$prepared_jsonl" \
          "$PREPARED_DIR/no_text.v2.train.filtered.jsonl" \
          "$PREPARED_DIR/no_text.v2.train.ref_content_leak_dropped.jsonl"
        ;;
      same_episode_valid)
        filter_ref_content_leakage \
          "$label" \
          "$prepared_jsonl" \
          "$PREPARED_DIR/no_text.same_episode_near_original_valid.filtered.jsonl" \
          "$PREPARED_DIR/no_text.same_episode_near_original_valid.ref_content_leak_dropped.jsonl"
        ;;
      heldout_cross_valid)
        filter_ref_content_leakage \
          "$label" \
          "$prepared_jsonl" \
          "$PREPARED_DIR/no_text.heldout_refdecorr_cross_channel_valid.filtered.jsonl" \
          "$PREPARED_DIR/no_text.heldout_refdecorr_cross_channel_valid.ref_content_leak_dropped.jsonl"
        ;;
    esac
  fi
}

make_prepared_links() {
  mkdir -p "$PREPARED_DIR"
  no_text_train_target="$PREPARED_DIR/no_text.v2.train.filtered.jsonl"
  no_text_valid_target="$PREPARED_DIR/no_text.heldout_refdecorr_cross_channel_valid.filtered.jsonl"
  no_text_seen_target="$PREPARED_DIR/no_text.same_episode_near_original_valid.filtered.jsonl"
  if [ ! -e "$no_text_train_target" ]; then
    no_text_train_target="$PREPARED_DIR/no_text.v2.train.jsonl"
  fi
  if [ ! -e "$no_text_valid_target" ]; then
    no_text_valid_target="$PREPARED_DIR/no_text.heldout_refdecorr_cross_channel_valid.jsonl"
  fi
  if [ ! -e "$no_text_seen_target" ]; then
    no_text_seen_target="$PREPARED_DIR/no_text.same_episode_near_original_valid.jsonl"
  fi
  ln -sfn "$no_text_train_target" "$PREPARED_DIR/no_text.train.jsonl"
  ln -sfn "$no_text_valid_target" "$PREPARED_DIR/no_text.valid.jsonl"
  ln -sfn "$no_text_seen_target" "$PREPARED_DIR/no_text.seen_valid.jsonl"
  ln -sfn "$no_text_valid_target" "$PREPARED_DIR/no_text.unseen_valid.jsonl"
  for split in text.train.jsonl text.valid.jsonl text.seen_valid.jsonl text.unseen_valid.jsonl; do
    if [ -e "$TEXT_PREPARED_DIR/$split" ]; then
      ln -sfn "$TEXT_PREPARED_DIR/$split" "$PREPARED_DIR/$split"
    fi
  done
  {
    printf '%s::repeat=1\n' "$PREPARED_DIR/no_text.train.jsonl"
    printf '%s::repeat=%s\n' "$PREPARED_DIR/text.train.jsonl" "$TEXT_REPEAT"
  } > "$PREPARED_DIR/mixed.train.spec.txt"
  {
    printf '%s::repeat=1\n' "$PREPARED_DIR/no_text.valid.jsonl"
    printf '%s::repeat=1\n' "$PREPARED_DIR/text.valid.jsonl"
  } > "$PREPARED_DIR/mixed.valid.spec.txt"
  {
    printf '%s::repeat=1\n' "$PREPARED_DIR/no_text.seen_valid.jsonl"
    printf '%s::repeat=1\n' "$PREPARED_DIR/text.seen_valid.jsonl"
  } > "$PREPARED_DIR/mixed.valid_seen.spec.txt"
  {
    printf '%s::repeat=1\n' "$PREPARED_DIR/no_text.unseen_valid.jsonl"
    printf '%s::repeat=1\n' "$PREPARED_DIR/text.unseen_valid.jsonl"
  } > "$PREPARED_DIR/mixed.valid_unseen.spec.txt"
}

verify_prepared() {
  "$PY" - "$PREPARED_DIR" "$TEXT_PREPARED_DIR" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
prepared = Path(sys.argv[1])
text_prepared = Path(sys.argv[2])
checks = [
    "no_text.v2.train.jsonl",
    "no_text.v2.train.filtered.jsonl",
    "no_text.same_episode_near_original_valid.jsonl",
    "no_text.same_episode_near_original_valid.filtered.jsonl",
    "no_text.heldout_refdecorr_cross_channel_valid.jsonl",
    "no_text.heldout_refdecorr_cross_channel_valid.filtered.jsonl",
    "no_text.train.jsonl",
    "no_text.valid.jsonl",
    "no_text.seen_valid.jsonl",
    "no_text.unseen_valid.jsonl",
    "text.train.jsonl",
]
summary = {"prepared_dir": str(prepared), "text_prepared_dir": str(text_prepared), "splits": {}}
sample_sets = {}

def norm_text(text):
    out = []
    for ch in str(text or "").lower():
        code = ord(ch)
        if ch.isalnum() or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            out.append(ch)
    return "".join(out)

for name in checks:
    path = prepared / name
    if not path.exists():
        raise SystemExit(f"missing prepared split: {path}")
    rows = 0
    missing = {"speaker_vec_path": 0, "content_token_ids": 0}
    ref_content_leaks = 0
    sample_ids = set()
    first = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            first = first or row
            sid = row.get("sample_id")
            if sid:
                sample_ids.add(str(sid))
            for key in missing:
                if key == "content_token_ids":
                    if not row.get(key):
                        missing[key] += 1
                elif not row.get(key):
                    missing[key] += 1
            if name.startswith("no_text"):
                ref_text = row.get("timbre_ref_text") or (row.get("moss_codecvc_meta") or {}).get("timbre_ref_text")
                target_text = row.get("target_text") or (row.get("moss_codecvc_meta") or {}).get("target_text")
                if norm_text(ref_text) and norm_text(ref_text) == norm_text(target_text):
                    ref_content_leaks += 1
    summary["splits"][name] = {
        "rows": rows,
        "missing": missing,
        "ref_content_leaks": ref_content_leaks,
        "first_sample_id": None if first is None else first.get("sample_id"),
    }
    if name.endswith(".filtered.jsonl") and ref_content_leaks:
        raise SystemExit(f"filtered split still has ref-content leaks: {name} leaks={ref_content_leaks}")
    sample_sets[name] = sample_ids
train = sample_sets.get("no_text.v2.train.jsonl", set())
for name in ("no_text.same_episode_near_original_valid.jsonl", "no_text.heldout_refdecorr_cross_channel_valid.jsonl"):
    overlap = len(train & sample_sets.get(name, set()))
    summary["splits"][name]["train_sample_id_overlap"] = overlap
    if overlap:
        raise SystemExit(f"train/valid sample_id overlap for {name}: {overlap}")
summary_path = prepared / "v2_real_no_text_prepare_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
}

echo "=========================================="
echo "Prepare v2 real-target no-text data for ver2.9"
echo "  ROOT=$ROOT"
echo "  TRAIN_INPUT_JSONL=$TRAIN_INPUT_JSONL"
echo "  VALID_SAME_INPUT_JSONL=$VALID_SAME_INPUT_JSONL"
echo "  VALID_CROSS_INPUT_JSONL=$VALID_CROSS_INPUT_JSONL"
echo "  WORK_ROOT=$WORK_ROOT"
echo "  PREPARED_DIR=$PREPARED_DIR"
echo "  TEXT_PREPARED_DIR=$TEXT_PREPARED_DIR"
echo "  WAVLM_MODEL=$WAVLM_MODEL"
echo "  WAVLM_SV_MODEL=$WAVLM_SV_MODEL"
echo "  GPU_IDS=$GPU_IDS"
echo "  DRY_RUN=$DRY_RUN SKIP_EXISTING=$SKIP_EXISTING FORCE=$FORCE OVERWRITE=$OVERWRITE"
echo "=========================================="

run_cmd "$PY" -m py_compile \
  scripts/001002_encode_codec_tokens.py \
  scripts/001003_build_moss_sft_jsonl.py \
  scripts/001007_build_speaker_embedding_plan.py \
  scripts/001010_attach_speaker_embeddings.py \
  scripts/001011_extract_speaker_embeddings.py \
  scripts/001015_extract_prosody_content_features.py \
  scripts/001020_extract_hubert_semantic_features.py \
  scripts/001046_extract_multilingual_content_tokens.py \
  scripts/002022_precompute_ver2_9_speaker_vecs.py \
  scripts/002033_mark_v2_real_no_text_known_content.py

mkdir -p "$WORK_ROOT" "$PREPARED_DIR"

process_split "train" "$TRAIN_INPUT_JSONL" "v2_real_no_text_refdecorr_train_minus_valid" "no_text.v2.train.jsonl"
process_split "same_episode_valid" "$VALID_SAME_INPUT_JSONL" "v2_real_no_text_refdecorr_valid_same_episode" "no_text.same_episode_near_original_valid.jsonl"
process_split "heldout_cross_valid" "$VALID_CROSS_INPUT_JSONL" "v2_real_no_text_refdecorr_valid_cross_channel" "no_text.heldout_refdecorr_cross_channel_valid.jsonl"

if ! truthy "$DRY_RUN"; then
  make_prepared_links
  verify_prepared
fi

echo "=========================================="
echo "v2 real-target no-text ver2.9 preparation finished"
echo "  prepared_dir=$PREPARED_DIR"
echo "  train=$PREPARED_DIR/no_text.v2.train.jsonl"
echo "  same_valid=$PREPARED_DIR/no_text.same_episode_near_original_valid.jsonl"
echo "  cross_valid=$PREPARED_DIR/no_text.heldout_refdecorr_cross_channel_valid.jsonl"
echo "=========================================="
