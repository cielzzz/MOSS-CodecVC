#!/usr/bin/env python
"""Pre-extract ver3.1 12.5-Hz semantic conditions.

The preferred path reuses the v1 manifest's frozen WavLM layer-9 sidecar.  In
the current v1 data this is ``microsoft/wavlm-base-plus`` (768 dimensions),
not a 1024-D WavLM-Large sidecar.  Text rows intentionally have no source
semantic sidecar: their source audio is only a prosody/style reference and its
BNF does not match the target text labels.  Therefore text rows are skipped by
default; ``--allow-text-source-bnf`` is an explicit diagnostic escape hatch,
not a production semantic condition.  No target-side feature is silently
substituted because that would leak target speaker information.

Use ``--mode extract`` once per shard, then ``--mode aggregate`` to validate
and merge the shard manifests.  The script is intentionally independent of
the DDLFM decoder; it only materialises adapter outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.models.content_adapter_v3_1 import ContentAdapterV31


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("extract", "aggregate"), default="extract")
    ap.add_argument("--input", action="append", default=[], metavar="SPLIT=JSONL")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--feature-key", default="source_wavlm_bnf_features_path")
    ap.add_argument(
        "--mode-filter",
        default="no_text",
        help="Comma-separated moss_codecvc_mode values; default no_text is speaker-safe.",
    )
    ap.add_argument("--extract-missing-source", action="store_true")
    ap.add_argument("--allow-text-source-bnf", action="store_true")
    ap.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--cache-dir", default="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", choices=("float32", "bf16", "fp16"), default="bf16")
    ap.add_argument("--save-dtype", choices=("float16", "float32"), default="float16")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--full-manifest",
        action="store_true",
        help="Keep the complete source JSON row in shard manifests. The default compact form avoids duplicating audio_codes.",
    )
    return ap.parse_args()


def parse_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--input must be SPLIT=JSONL, got {value!r}")
        split, path = value.split("=", 1)
        specs.append((split.strip(), Path(path).expanduser().resolve()))
    if not specs:
        raise ValueError("at least one --input SPLIT=JSONL is required")
    return specs


def nested_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def row_mode(row: dict[str, Any], split: str) -> str:
    value = row.get("moss_codecvc_mode")
    if value in (None, ""):
        meta = row.get("moss_codecvc_meta")
        if isinstance(meta, dict):
            value = meta.get("moss_codecvc_mode") or meta.get("mode")
    return str(value or split or "unknown")


def parse_mode_filter(value: str) -> set[str] | None:
    values = {part.strip() for part in str(value).split(",") if part.strip()}
    if not values or "all" in values:
        return None
    return values


def iter_rows(specs: list[tuple[str, Path]]):
    for split, path in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                yield split, line_no, json.loads(line)


def load_feature(path: str | Path) -> torch.Tensor:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    value = None
    if isinstance(payload, dict):
        for key in ("wavlm_bnf_features", "wavlm_features", "semantic_features"):
            candidate = payload.get(key)
            if torch.is_tensor(candidate):
                value = candidate
                break
    elif torch.is_tensor(payload):
        value = payload
    if value is None:
        raise ValueError(f"no feature tensor in {path}")
    value = value.detach().float().cpu()
    if value.dim() != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"feature must be non-empty [T,D], got {tuple(value.shape)}: {path}")
    return value


def read_audio(path: str, target_sr: int) -> torch.Tensor:
    import soundfile as sf

    values, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(values).mean(dim=1)
    if int(sr) == int(target_sr):
        return audio
    try:
        import torchaudio.functional as AF

        return AF.resample(audio, int(sr), int(target_sr))
    except Exception:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(sr), int(target_sr))
        return torch.from_numpy(
            resample_poly(audio.numpy(), int(target_sr) // gcd, int(sr) // gcd).astype("float32")
        )


def load_wavlm(args: argparse.Namespace):
    from transformers import AutoFeatureExtractor, AutoModel

    common = {
        "cache_dir": str(Path(args.cache_dir).expanduser()),
        "local_files_only": bool(args.local_files_only),
    }
    extractor = AutoFeatureExtractor.from_pretrained(args.wavlm_model, **common)
    model = AutoModel.from_pretrained(args.wavlm_model, **common)
    device = torch.device(args.device)
    model.eval().to(device)
    if args.precision == "bf16" and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    elif args.precision == "fp16" and device.type == "cuda":
        model = model.half()
    return extractor, model, device


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


@torch.no_grad()
def extract_source_feature(audio_path: str, extractor: Any, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    target_sr = int(getattr(extractor, "sampling_rate", 16000) or 16000)
    audio = read_audio(audio_path, target_sr)
    inputs = extractor(audio.numpy(), sampling_rate=target_sr, return_tensors="pt", padding=True, return_attention_mask=True)
    inputs = {
        key: value.to(device=device, dtype=next(model.parameters()).dtype)
        if torch.is_tensor(value) and torch.is_floating_point(value)
        else value.to(device=device)
        if torch.is_tensor(value)
        else value
        for key, value in inputs.items()
    }
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) <= 9:
        raise RuntimeError("WavLM checkpoint did not return layer-9 hidden states")
    hidden = hidden_states[9].detach().float()
    mask = feature_attention_mask(model, int(hidden.shape[1]), inputs.get("attention_mask"))
    valid = int(mask[0].long().sum().item()) if mask is not None else int(hidden.shape[1])
    if valid <= 0:
        raise RuntimeError(f"empty WavLM feature for {audio_path}")
    return hidden[0, :valid].cpu()


def checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    if checkpoint.is_dir():
        candidate = checkpoint / "adapter_config.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
        checkpoint = checkpoint / "adapter.pt"
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        return dict(payload["config"])
    raise ValueError(f"checkpoint has no adapter config: {checkpoint}")


def load_adapter(checkpoint_path: Path, device: torch.device) -> tuple[ContentAdapterV31, dict[str, Any]]:
    config = checkpoint_config(checkpoint_path)
    model = ContentAdapterV31(
        input_dim=int(config["input_dim"]),
        semantic_dim=int(config.get("semantic_dim", 512)),
        num_layers=int(config.get("num_layers", 2)),
        num_heads=int(config.get("num_heads", 8)),
        dropout=float(config.get("dropout", 0.0)),
        downsample_kernel_size=int(config.get("downsample_kernel", 4)),
        downsample_stride=int(config.get("downsample_stride", 4)),
        vocab_size=int(config.get("vocab_size", 0)),
    )
    state_path = checkpoint_path / "adapter.pt" if checkpoint_path.is_dir() else checkpoint_path
    payload = torch.load(str(state_path), map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, config


def safe_id(split: str, row: dict[str, Any], line_no: int) -> str:
    raw = str(row.get("sample_id") or row.get("utt_id") or f"line-{line_no}")
    digest = hashlib.sha256(f"{split}\0{raw}\0{line_no}".encode()).hexdigest()[:16]
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)[:100] or "row"
    return f"{stem}_{digest}"


def atomic_npy(path: Path, value: np.ndarray) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, value)
    tmp.replace(path)
    return int(path.stat().st_size)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_semantic_batch(
    pending: list[tuple[str, int, dict[str, Any], torch.Tensor, str]],
    *,
    adapter: ContentAdapterV31,
    device: torch.device,
    output_root: Path,
    checkpoint: Path,
    save_dtype: str,
    full_manifest: bool,
    stats: dict[str, Any],
    out_handle: Any,
) -> None:
    """Run one padded adapter forward and write each row's semantic tensor."""
    if not pending:
        return
    max_frames = max(int(item[3].shape[0]) for item in pending)
    feature_dim = int(pending[0][3].shape[1])
    batch_features = torch.zeros(
        (len(pending), max_frames, feature_dim), dtype=torch.float32, device=device
    )
    batch_mask = torch.zeros((len(pending), max_frames), dtype=torch.bool, device=device)
    for index, (_split, _line_no, _row, feature, _kind) in enumerate(pending):
        frames = int(feature.shape[0])
        batch_features[index, :frames] = feature.to(device=device)
        batch_mask[index, :frames] = True
    with torch.no_grad():
        output = adapter(batch_features, batch_mask)
    for index, (split, line_no, row, _feature, feature_kind) in enumerate(pending):
        semantic = output.semantic[index, output.semantic_mask[index]].detach().float().cpu().numpy()
        if semantic.ndim != 2 or semantic.shape[0] <= 0:
            raise ValueError(f"empty adapter output for {row.get('sample_id')}")
        if save_dtype == "float16":
            semantic = semantic.astype(np.float16, copy=False)
        else:
            semantic = semantic.astype(np.float32, copy=False)
        uid = safe_id(split, row, line_no)
        destination = output_root / split / f"{uid}.npy"
        if destination.exists() and not getattr(_write_semantic_batch, "overwrite", False):
            stats["reused"] += 1
        else:
            atomic_npy(destination, semantic)
            stats["written"] += 1
        stats["frames"] += int(semantic.shape[0])
        mode = row_mode(row, split)
        record = dict(row) if full_manifest else {
            "sample_id": str(row.get("sample_id") or row.get("utt_id") or f"line-{line_no}"),
            "utt_id": row.get("utt_id"),
            "moss_codecvc_mode": mode,
            "language": row.get("language"),
            "split": split,
        }
        record["semantic_v3_1_path"] = str(destination)
        record["semantic_v3_1_frames"] = int(semantic.shape[0])
        record["semantic_v3_1_dim"] = int(semantic.shape[1])
        record["semantic_v3_1_rate_hz"] = 12.5
        record["semantic_v3_1_feature_kind"] = feature_kind
        record["semantic_v3_1_adapter_checkpoint"] = str(checkpoint)
        record["semantic_v3_1_sha256"] = sha256_file(destination)
        out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    pending.clear()


def extract(args: argparse.Namespace) -> int:
    specs = parse_specs(args.input)
    # A torchrun wrapper can launch one extractor per GPU without duplicating
    # CLI arguments.  Explicit CLI shard values still take precedence for
    # local/manual runs.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and int(args.num_shards) == 1 and int(args.shard_index) == 0:
        args.num_shards = world_size
        args.shard_index = rank
    mode_filter = parse_mode_filter(args.mode_filter)
    if int(args.num_shards) <= 0 or not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("invalid shard index/num-shards")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    device = torch.device(args.device)
    adapter, adapter_cfg = load_adapter(checkpoint, device)
    wavlm_extractor = wavlm_model = wavlm_device = None
    if args.extract_missing_source:
        wavlm_extractor, wavlm_model, wavlm_device = load_wavlm(args)
    shard_path = output_root / "shards" / f"semantic-{int(args.shard_index):03d}-of-{int(args.num_shards):03d}.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "status": "running",
        "started_at_utc": utc_now(),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "scanned": 0,
        "written": 0,
        "reused": 0,
        "missing_feature": 0,
        "missing_audio": 0,
        "text_skipped_no_source_semantic": 0,
        "mode_filter": sorted(mode_filter) if mode_filter is not None else ["all"],
        "errors": 0,
        "frames": 0,
        "input_dim": int(adapter_cfg["input_dim"]),
        "semantic_dim": int(adapter_cfg["semantic_dim"]),
        "checkpoint": str(checkpoint),
        "label_source": adapter_cfg.get("label_source_probe"),
    }
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    _write_semantic_batch.overwrite = bool(args.overwrite)
    pending: list[tuple[str, int, dict[str, Any], torch.Tensor, str]] = []
    with shard_path.open("w", encoding="utf-8") as out:
        for global_line, (split, line_no, row) in enumerate(iter_rows(specs)):
            if global_line % int(args.num_shards) != int(args.shard_index):
                continue
            if int(args.max_rows) > 0 and stats["scanned"] >= int(args.max_rows):
                break
            stats["scanned"] += 1
            mode = row_mode(row, split)
            if mode_filter is not None and mode not in mode_filter:
                stats["mode_filtered"] = int(stats.get("mode_filtered", 0)) + 1
                continue
            feature_path = nested_value(row, args.feature_key)
            feature_kind = "manifest_sidecar"
            try:
                if feature_path not in (None, "") and Path(str(feature_path)).is_file():
                    feature = load_feature(str(feature_path))
                elif args.extract_missing_source:
                    mode = row_mode(row, split).strip().lower()
                    if mode == "text" and not args.allow_text_source_bnf:
                        stats["text_skipped_no_source_semantic"] += 1
                        continue
                    audio_path = nested_value(row, "source_audio")
                    if audio_path in (None, "") or not Path(str(audio_path)).is_file():
                        stats["missing_audio"] += 1
                        continue
                    feature = extract_source_feature(str(audio_path), wavlm_extractor, wavlm_model, wavlm_device)
                    feature_kind = "wavlm_layer9_source_audio"
                else:
                    stats["missing_feature"] += 1
                    continue
                if int(feature.shape[-1]) != int(adapter_cfg["input_dim"]):
                    raise ValueError(
                        f"feature dim={feature.shape[-1]} != adapter input={adapter_cfg['input_dim']}"
                    )
                pending.append((split, line_no, row, feature, feature_kind))
                if len(pending) >= int(args.batch_size):
                    _write_semantic_batch(
                        pending,
                        adapter=adapter,
                        device=device,
                        output_root=output_root,
                        checkpoint=checkpoint,
                        save_dtype=args.save_dtype,
                        full_manifest=bool(args.full_manifest),
                        stats=stats,
                        out_handle=out,
                    )
            except Exception as exc:
                stats["errors"] += 1
                print(f"[semantic-v3.1] error split={split} line={line_no}: {exc}", file=sys.stderr, flush=True)
            if stats["scanned"] and stats["scanned"] % max(1, int(args.progress_every)) == 0:
                print(json.dumps(stats, ensure_ascii=False), flush=True)
        _write_semantic_batch(
            pending,
            adapter=adapter,
            device=device,
            output_root=output_root,
            checkpoint=checkpoint,
            save_dtype=args.save_dtype,
            full_manifest=bool(args.full_manifest),
            stats=stats,
            out_handle=out,
        )
    stats["status"] = "completed" if stats["errors"] == 0 else "completed_with_errors"
    stats["finished_at_utc"] = utc_now()
    marker = shard_path.with_suffix(".stats.json")
    marker.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0 if stats["errors"] == 0 else 1


def aggregate(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    shard_dir = output_root / "shards"
    shard_files = sorted(shard_dir.glob("semantic-*-of-*.jsonl"))
    stats_files = sorted(shard_dir.glob("semantic-*.stats.json"))
    if not shard_files or len(stats_files) != len(shard_files):
        raise RuntimeError(f"missing semantic shards/stats under {shard_dir}")
    expected_indices = list(range(len(shard_files)))
    actual_indices = []
    for path in shard_files:
        try:
            actual_indices.append(int(path.name.split("semantic-", 1)[1].split("-of-", 1)[0]))
        except Exception as exc:
            raise RuntimeError(f"invalid semantic shard filename: {path.name}") from exc
    if actual_indices != expected_indices:
        raise RuntimeError(f"semantic shards are not complete/contiguous: {actual_indices} != {expected_indices}")
    stats = [json.loads(path.read_text(encoding="utf-8")) for path in stats_files]
    if any(item.get("status") != "completed" for item in stats):
        raise RuntimeError("one or more semantic shards did not complete without errors")
    merged = output_root / "manifest.jsonl"
    rows = 0
    frames = 0
    dims: set[int] = set()
    with merged.open("w", encoding="utf-8") as out:
        for shard in shard_files:
            with shard.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    path = Path(row["semantic_v3_1_path"])
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    arr = np.load(path, mmap_mode="r")
                    if arr.ndim != 2 or arr.shape[0] <= 0:
                        raise ValueError(f"invalid semantic tensor {path}: {arr.shape}")
                    dims.add(int(arr.shape[1]))
                    expected_hash = row.get("semantic_v3_1_sha256")
                    if expected_hash and sha256_file(path) != expected_hash:
                        raise ValueError(f"semantic hash mismatch: {path}")
                    rows += 1
                    frames += int(arr.shape[0])
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if len(dims) != 1:
        raise RuntimeError(f"mixed semantic dimensions: {sorted(dims)}")
    completion = {
        "schema": "ver3_1_semantic_v1_completion_v1",
        "status": "completed",
        "generated_at_utc": utc_now(),
        "rows": rows,
        "frames": frames,
        "semantic_dim": next(iter(dims)),
        "rate_hz": 12.5,
        "shards": len(shard_files),
        "output_root": str(output_root),
        "manifest": str(merged),
        "shard_stats": stats,
    }
    (output_root / "COMPLETED.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "aggregate":
        return aggregate(args)
    return extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
