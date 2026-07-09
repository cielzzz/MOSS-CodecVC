#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel


ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text[:180] or "row"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc


def record_value(record: dict[str, Any], key: str) -> Any | None:
    if record.get(key) is not None:
        return record[key]
    meta = record.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def record_id(record: dict[str, Any], fallback: str) -> str:
    for key in ("pair_id", "sample_id", "case_id", "utt_id", "id"):
        value = record_value(record, key)
        if value not in (None, ""):
            return safe_id(str(value))
    return safe_id(fallback)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Precompute WavLM sequence speaker features for ver2.9 Fix 6.")
    ap.add_argument("--input-jsonl", required=True, help="Input train-ready JSONL manifest.")
    ap.add_argument("--output-jsonl", required=True, help="Output JSONL with speaker_seq_path added.")
    ap.add_argument(
        "--speaker-seq-dir",
        default="",
        help="Directory for .npy sequence features. Defaults to output parent/speaker_seq_features.",
    )
    ap.add_argument("--audio-key", default="timbre_ref_audio")
    ap.add_argument("--model-name-or-path", default="microsoft/wavlm-base-plus")
    ap.add_argument("--layer", type=int, default=9, help="Hidden-state layer index. 9 matches the Vevo-style probe.")
    ap.add_argument(
        "--downsample-stride",
        type=int,
        default=2,
        help="Keep every Nth WavLM frame. 2 is about 25 Hz for WavLM; 4 is about 12.5 Hz.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-index", type=int, default=-1, help="0-based shard index. Negative disables sharding.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--summary-json", default="")
    return ap.parse_args()


def load_audio(path: str | Path, target_sr: int) -> np.ndarray:
    try:
        wav, sr = torchaudio.load(str(path))
        wav = wav.float()
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != int(target_sr):
            wav = torchaudio.functional.resample(wav, int(sr), int(target_sr))
        return wav.squeeze(0).cpu().numpy()
    except Exception:
        import soundfile as sf
        from scipy.signal import resample_poly

        wav_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav_np = wav_np.mean(axis=1)
        if int(sr) != int(target_sr):
            gcd = math.gcd(int(sr), int(target_sr))
            wav_np = resample_poly(wav_np, int(target_sr) // gcd, int(sr) // gcd).astype(np.float32, copy=False)
        return np.asarray(wav_np, dtype=np.float32)


def feature_attention_mask(model: torch.nn.Module, hidden_len: int, sample_mask: torch.Tensor | None) -> torch.Tensor | None:
    if sample_mask is None:
        return None
    base = model.module if hasattr(model, "module") else model
    if hasattr(base, "_get_feature_vector_attention_mask"):
        return base._get_feature_vector_attention_mask(hidden_len, sample_mask)
    lengths = sample_mask.long().sum(dim=1)
    approx = torch.div(lengths + 319, 320, rounding_mode="floor").clamp(min=1, max=hidden_len)
    idx = torch.arange(hidden_len, device=sample_mask.device).view(1, -1)
    return idx < approx.view(-1, 1)


def main() -> int:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser()
    output_jsonl = Path(args.output_jsonl).expanduser()
    speaker_seq_dir = (
        Path(args.speaker_seq_dir).expanduser()
        if args.speaker_seq_dir
        else output_jsonl.parent / "speaker_seq_features"
    )
    batch_size = max(1, int(args.batch_size))
    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index >= num_shards:
        raise ValueError(f"shard-index must be < num-shards, got {shard_index} >= {num_shards}")
    stride = max(1, int(args.downsample_stride))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    speaker_seq_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name_or_path,
        local_files_only=bool(args.local_files_only),
    )
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        local_files_only=bool(args.local_files_only),
    )
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    target_sr = int(getattr(feature_extractor, "sampling_rate", 16000) or 16000)

    compute_dtype = torch.float32
    if args.dtype == "float16":
        model = model.half()
        compute_dtype = torch.float16
        save_dtype = np.float16
    elif args.dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
        compute_dtype = torch.bfloat16
        save_dtype = np.float16
    else:
        save_dtype = np.float32

    rows = 0
    scanned = 0
    written = 0
    reused = 0
    missing = 0
    feature_lengths: list[int] = []
    feature_dims: list[int] = []

    @torch.inference_mode()
    def flush_batch(out, batch: list[dict[str, Any]]) -> None:
        nonlocal written, reused, missing
        encode_items: list[tuple[dict[str, Any], Path, str]] = []
        wavs: list[np.ndarray] = []
        for item in batch:
            record = item["record"]
            audio_path = item["audio_path"]
            seq_path = item["seq_path"]
            if not audio_path:
                missing += 1
                record["speaker_seq_path"] = None
                continue
            record["speaker_seq_path"] = str(seq_path)
            if args.overwrite or not seq_path.exists():
                wavs.append(load_audio(audio_path, target_sr))
                encode_items.append((record, seq_path, str(audio_path)))
            else:
                reused += 1
        if encode_items:
            inputs = feature_extractor(
                wavs,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            inputs = {
                key: value.to(device=device, dtype=compute_dtype) if torch.is_floating_point(value) else value.to(device)
                for key, value in inputs.items()
            }
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("WavLM model did not return hidden_states")
            layer = int(args.layer)
            if layer < 0:
                layer = len(hidden_states) + layer
            if layer < 0 or layer >= len(hidden_states):
                raise ValueError(f"layer index {args.layer} outside hidden_states length {len(hidden_states)}")
            hidden = hidden_states[layer].detach().float()
            mask = feature_attention_mask(model, int(hidden.shape[1]), inputs.get("attention_mask"))
            if mask is None:
                mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            for row_idx, (_record, seq_path, audio_path) in enumerate(encode_items):
                valid_len = int(mask[row_idx].long().sum().item())
                if valid_len <= 0:
                    raise RuntimeError(f"failed to compute speaker sequence for {audio_path}")
                feat = hidden[row_idx, :valid_len]
                if stride > 1:
                    feat = feat[::stride]
                if int(feat.shape[0]) <= 0:
                    raise RuntimeError(f"empty speaker sequence after downsampling for {audio_path}")
                feature_lengths.append(int(feat.shape[0]))
                feature_dims.append(int(feat.shape[-1]))
                tmp_path = seq_path.with_name(f"{seq_path.name}.tmp{os.getpid()}.npy")
                np.save(tmp_path, feat.cpu().numpy().astype(save_dtype, copy=False))
                tmp_path.replace(seq_path)
                written += 1
        for item in batch:
            out.write(json.dumps(item["record"], ensure_ascii=False) + "\n")

    batch: list[dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as out:
        for line_no, record in iter_jsonl(input_jsonl):
            scanned += 1
            if shard_index >= 0 and (line_no - 1) % num_shards != shard_index:
                continue
            if args.max_rows > 0 and rows >= args.max_rows:
                break
            rows += 1
            audio_path = record_value(record, args.audio_key)
            pair_id = record_id(record, f"{input_jsonl.stem}_{line_no}")
            seq_path = speaker_seq_dir / f"{pair_id}.npy"
            batch.append({"record": record, "audio_path": audio_path, "seq_path": seq_path})
            if len(batch) >= batch_size:
                flush_batch(out, batch)
                batch.clear()
            if rows % 1000 == 0:
                print(
                    f"[speaker-seq] rows={rows} scanned={scanned} written={written} "
                    f"reused={reused} missing={missing}",
                    flush=True,
                )
        if batch:
            flush_batch(out, batch)
            batch.clear()

    summary = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "speaker_seq_dir": str(speaker_seq_dir),
        "rows": rows,
        "scanned": scanned,
        "written": written,
        "reused": reused,
        "missing_audio": missing,
        "model_name_or_path": str(args.model_name_or_path),
        "layer": int(args.layer),
        "downsample_stride": stride,
        "feature_dim": int(feature_dims[0]) if feature_dims else None,
        "feature_len_mean": float(np.mean(feature_lengths)) if feature_lengths else None,
        "feature_len_min": int(np.min(feature_lengths)) if feature_lengths else None,
        "feature_len_max": int(np.max(feature_lengths)) if feature_lengths else None,
    }
    summary_json = Path(args.summary_json).expanduser() if args.summary_json else output_jsonl.with_suffix(".speaker_seq_summary.json")
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"[speaker-seq] done input={input_jsonl} output={output_jsonl} rows={rows} scanned={scanned} "
        f"written={written} reused={reused} missing_audio={missing} speaker_seq_dir={speaker_seq_dir} "
        f"batch_size={batch_size} shard_index={shard_index} num_shards={num_shards} summary={summary_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
