#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl, safe_stem, stable_id


os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

DOWNLOAD_ROOT = Path("/inspire/ssd/project/embodied-multimodality/public/xyzhang/download")
DEFAULT_CACHE_DIR = DOWNLOAD_ROOT / "huggingface"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Attach frozen HuBERT/WavLM continuous semantic features for Ver2.1 semantic loss."
    )
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--extractor", choices=("hubert", "wavlm"), default="hubert")
    ap.add_argument("--model-name-or-path", default="facebook/hubert-base-ls960")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument(
        "--use-safetensors",
        choices=("auto", "true", "false"),
        default="false",
        help="Default false avoids transformers background safetensors auto-conversion hanging after extraction.",
    )
    ap.add_argument("--source", choices=("source", "target", "both"), default="source")
    ap.add_argument("--layer", type=int, default=9, help="Hidden-state layer index. Use -1 for last_hidden_state.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    ap.add_argument("--save-dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    ap.add_argument("--downsample-stride", type=int, default=1)
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1, help="Total number of modulo shards for parallel feature extraction.")
    ap.add_argument("--shard-index", type=int, default=0, help="Current modulo shard index, in [0, num_shards).")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--reuse-existing-features",
        action="store_true",
        help="When rewriting the output manifest, reuse existing feature .pt files instead of recomputing them.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Validate JSONL/audio paths and write no features.")
    return ap.parse_args()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def nested_get(row: dict[str, Any], key: str) -> Any | None:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def resolve_audio(row: dict[str, Any], split: str, audio_root: Path | None) -> str | None:
    if split == "source":
        keys = ("source_audio", "source_wav", "prosody_ref_audio", "audio", "wav", "path")
    else:
        keys = ("target_audio", "teacher_audio", "converted_audio")
    for key in keys:
        value = nested_get(row, key)
        if value in (None, ""):
            continue
        path = Path(str(value))
        if not path.is_absolute() and audio_root is not None:
            path = audio_root / path
        return str(path)
    return None


def torch_dtype(name: str, device: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if str(device).startswith("cuda"):
        return torch.float16
    return torch.float32


def save_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def read_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(wav).mean(dim=1)
    if int(sr) == int(target_sr):
        return audio
    try:
        import torchaudio.functional as AF

        return AF.resample(audio, orig_freq=int(sr), new_freq=int(target_sr))
    except Exception:
        try:
            from scipy.signal import resample_poly

            gcd = math.gcd(int(sr), int(target_sr))
            up = int(target_sr) // gcd
            down = int(sr) // gcd
            return torch.from_numpy(resample_poly(audio.numpy(), up, down).astype("float32"))
        except Exception as exc:
            raise RuntimeError(
                f"audio sample rate mismatch {sr}->{target_sr}; install torchaudio or scipy for resampling"
            ) from exc


def feature_path(root: Path, split: str, sample_id: str, audio_path: str, extractor: str, layer: int) -> Path:
    name = f"{safe_stem(sample_id)}_{stable_id(split, audio_path, extractor, layer, length=12)}.pt"
    return root / split / name


def load_model(args: argparse.Namespace):
    from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "cache_dir": str(cache_dir),
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": True,
    }
    try:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, **common)
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path, **common)
    dtype = torch_dtype(args.dtype, args.device)
    model_kwargs = dict(common)
    use_safetensors = str(args.use_safetensors).strip().lower()
    if use_safetensors != "auto":
        model_kwargs["use_safetensors"] = use_safetensors == "true"
    model = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=dtype, **model_kwargs)
    model.eval().to(args.device)
    for param in model.parameters():
        param.requires_grad = False
    return processor, model, dtype


def processor_sample_rate(processor: Any) -> int:
    value = getattr(processor, "sampling_rate", None)
    if value:
        return int(value)
    feature_extractor = getattr(processor, "feature_extractor", None)
    value = getattr(feature_extractor, "sampling_rate", None)
    if value:
        return int(value)
    return 16000


@torch.no_grad()
def extract_features(
    *,
    audio_path: str,
    processor: Any,
    model: torch.nn.Module,
    device: str,
    dtype: torch.dtype,
    layer: int,
    downsample_stride: int,
) -> torch.Tensor:
    sr = processor_sample_rate(processor)
    audio = read_audio_mono(audio_path, sr)
    inputs = processor(audio.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    model_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if key == "input_values":
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device=device)
        model_inputs[key] = value
    outputs = model(**model_inputs, output_hidden_states=True)
    if int(layer) == -1 or getattr(outputs, "hidden_states", None) is None:
        features = outputs.last_hidden_state
    else:
        hidden_states = outputs.hidden_states
        idx = int(layer)
        if idx < 0:
            idx = len(hidden_states) + idx
        if idx < 0 or idx >= len(hidden_states):
            raise ValueError(f"layer index {layer} outside available hidden_states={len(hidden_states)}")
        features = hidden_states[idx]
    features = features.squeeze(0).detach().float().cpu()
    stride = max(1, int(downsample_stride))
    if stride > 1:
        features = features[::stride].contiguous()
    if features.dim() != 2 or features.numel() == 0:
        raise RuntimeError(f"empty semantic features for {audio_path}: shape={tuple(features.shape)}")
    return features


def save_features(
    path: Path,
    *,
    features: torch.Tensor,
    audio_path: str,
    split: str,
    args: argparse.Namespace,
    sample_rate: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = features.to(dtype=save_dtype(args.save_dtype)).cpu()
    key = f"{args.extractor}_features"
    payload = {
        "semantic_features": tensor,
        key: tensor,
        "semantic_mask": torch.ones((tensor.shape[0],), dtype=torch.bool),
        "extractor": args.extractor,
        "model_name_or_path": args.model_name_or_path,
        "layer": int(args.layer),
        "feature_dim": int(tensor.shape[-1]),
        "frames": int(tensor.shape[0]),
        "sample_rate": int(sample_rate),
        "audio_path": audio_path,
        "split": split,
    }
    if args.extractor == "wavlm":
        payload["wavlm_bnf_features"] = tensor
    torch.save(payload, path)


def load_saved_feature_dim(path: Path) -> int | None:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if isinstance(payload, dict):
        value = payload.get("feature_dim")
        if value is not None:
            return int(value)
        for key in ("semantic_features", "hubert_features", "wavlm_features", "wavlm_bnf_features"):
            tensor = payload.get(key)
            if torch.is_tensor(tensor) and tensor.dim() >= 2:
                return int(tensor.shape[-1])
    return None


def main() -> int:
    args = parse_args()
    if args.extractor == "wavlm" and args.model_name_or_path == "facebook/hubert-base-ls960":
        args.model_name_or_path = "microsoft/wavlm-base-plus"
    input_path = Path(args.input_jsonl).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    feature_root = Path(args.feature_root).expanduser()
    audio_root = Path(args.audio_root).expanduser() if args.audio_root else None
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists, pass --overwrite: {output_path}")
    if int(args.num_shards) <= 0:
        raise ValueError("--num-shards must be positive")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if int(args.downsample_stride) <= 0:
        raise ValueError("--downsample-stride must be positive")

    processor = model = dtype = None
    if not args.dry_run:
        processor, model, dtype = load_model(args)
        semantic_sample_rate = processor_sample_rate(processor)
    else:
        semantic_sample_rate = 16000

    splits = ("source", "target") if args.source == "both" else (args.source,)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    stats = Counter()
    progress_every = max(1, int(args.progress_every))
    with tmp_path.open("w", encoding="utf-8") as handle:
        for input_idx, row in enumerate(iter_jsonl(input_path)):
            if args.max_rows > 0 and input_idx >= args.max_rows:
                break
            stats["input_rows_seen"] += 1
            if int(args.num_shards) > 1 and (input_idx % int(args.num_shards)) != int(args.shard_index):
                continue
            out = dict(row)
            sample_id = str(out.get("sample_id") or input_idx)
            for split in splits:
                audio_path = resolve_audio(out, split, audio_root)
                if not audio_path:
                    stats[f"missing_{split}_audio"] += 1
                    continue
                path = feature_path(feature_root, split, sample_id, audio_path, args.extractor, int(args.layer))
                should_extract = not args.dry_run and (
                    not path.exists() or (args.overwrite and not args.reuse_existing_features)
                )
                if should_extract:
                    features = extract_features(
                        audio_path=audio_path,
                        processor=processor,
                        model=model,
                        device=args.device,
                        dtype=dtype,
                        layer=int(args.layer),
                        downsample_stride=int(args.downsample_stride),
                    )
                    save_features(
                        path,
                        features=features,
                        audio_path=audio_path,
                        split=split,
                        args=args,
                        sample_rate=semantic_sample_rate,
                    )
                    stats[f"{split}_features_extracted"] += 1
                    stats["feature_dim"] = int(features.shape[-1])
                elif path.exists():
                    stats[f"{split}_features_reused"] += 1
                    if not stats.get("feature_dim"):
                        feature_dim = load_saved_feature_dim(path)
                        if feature_dim:
                            stats["feature_dim"] = int(feature_dim)
                else:
                    stats[f"{split}_features_dry_run"] += 1
                out[f"{split}_semantic_features_path"] = str(path.resolve(strict=False))
                out[f"{split}_{args.extractor}_features_path"] = str(path.resolve(strict=False))
                if args.extractor == "wavlm":
                    out[f"{split}_wavlm_bnf_features_path"] = str(path.resolve(strict=False))
                if stats.get("feature_dim"):
                    out[f"{split}_semantic_feature_dim"] = int(stats["feature_dim"])
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            if stats["rows"] % progress_every == 0:
                print(f"[semantic-features] rows={stats['rows']} stats={dict(stats)}", flush=True)

    tmp_path.replace(output_path)
    summary = {
        "status": "dry_run" if args.dry_run else "complete",
        "rows": int(stats["rows"]),
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "feature_root": str(feature_root),
        "extractor": args.extractor,
        "model_name_or_path": args.model_name_or_path,
        "layer": int(args.layer),
        "source": args.source,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "save_dtype": args.save_dtype,
        "use_safetensors": args.use_safetensors,
        "reuse_existing_features": bool(args.reuse_existing_features),
        "downsample_stride": int(args.downsample_stride),
        "stats": dict(stats),
    }
    write_json_atomic(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    write_json_atomic(output_path.with_name(output_path.name + ".done.json"), summary)
    print(f"[semantic-features] wrote rows={stats['rows']} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
