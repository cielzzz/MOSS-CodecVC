#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


SOURCE_KEYS = (
    "source_asr_bnf_feature_path",
    "source_asr_bnf_features_path",
    "source_bnf_feature_path",
    "source_bnf_features_path",
    "source_wavlm_bnf_feature_path",
    "source_wavlm_bnf_features_path",
    "source_wavlm_feature_path",
    "source_wavlm_features_path",
    "source_semantic_feature_path",
    "source_semantic_features_path",
    "source_hubert_feature_path",
    "source_hubert_features_path",
)
TARGET_KEYS = (
    "teacher_target_semantic_feature_path",
    "teacher_target_semantic_features_path",
    "target_asr_bnf_feature_path",
    "target_asr_bnf_features_path",
    "target_bnf_feature_path",
    "target_bnf_features_path",
    "target_wavlm_bnf_feature_path",
    "target_wavlm_bnf_features_path",
    "target_wavlm_feature_path",
    "target_wavlm_features_path",
    "target_semantic_feature_path",
    "target_semantic_features_path",
    "target_hubert_feature_path",
    "target_hubert_features_path",
)
FEATURE_KEYS = (
    "semantic_features",
    "features",
    "hidden_states",
    "source_semantic_features",
    "target_semantic_features",
    "source_wavlm_bnf_features",
    "target_wavlm_bnf_features",
    "wavlm_bnf_features",
    "hubert_features",
    "wavlm_features",
)


def record_value(record: dict[str, Any], *keys: str) -> Any | None:
    meta = record.get("moss_codecvc_meta")
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
        if isinstance(meta, dict):
            value = meta.get(key)
            if value not in (None, ""):
                return value
    return None


def load_feature(path: str | Path | None) -> torch.Tensor | None:
    if not path:
        return None
    path = Path(path).expanduser()
    if not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        import numpy as np

        value = np.load(path)
    elif path.suffix.lower() == ".npz":
        import numpy as np

        payload = dict(np.load(path))
        value = next((payload[key] for key in FEATURE_KEYS if key in payload), None)
    else:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            value = next((payload.get(key) for key in FEATURE_KEYS if payload.get(key) is not None), None)
        else:
            value = payload
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.dim() != 2 or tensor.numel() == 0:
        return None
    return tensor


def pooled(feature: torch.Tensor) -> torch.Tensor:
    return feature.float().mean(dim=0)


def load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_summary(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    value = tensor.detach().float()
    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=value.device)
        if value.dim() == 3:
            value = value.masked_select(mask.unsqueeze(-1)).reshape(-1, value.shape[-1])
        else:
            value = value.masked_select(mask)
    if value.numel() == 0:
        return {"numel": 0}
    finite = torch.isfinite(value)
    clean = value.masked_select(finite)
    if clean.numel() == 0:
        return {
            "numel": int(value.numel()),
            "finite_ratio": 0.0,
            "nan_count": int(torch.isnan(value).sum().item()),
            "inf_count": int(torch.isinf(value).sum().item()),
        }
    return {
        "numel": int(value.numel()),
        "shape": list(tensor.shape),
        "finite_ratio": float(finite.float().mean().item()),
        "nan_count": int(torch.isnan(value).sum().item()),
        "inf_count": int(torch.isinf(value).sum().item()),
        "mean": float(clean.mean().item()),
        "std": float(clean.std(unbiased=False).item()) if clean.numel() > 1 else 0.0,
        "abs_mean": float(clean.abs().mean().item()),
        "min": float(clean.min().item()),
        "max": float(clean.max().item()),
    }


def audit_train_batch(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.train_jsonl_spec:
        return None
    train_mod = load_module_from_file("moss_codecvc_train_mod", ROOT / "scripts/002002_train_moss_codecvc_lora.py")
    cfg = train_mod.load_config(args.config)
    moss_root = train_mod.deep_get(cfg, "moss.root")
    model_path = args.model_path or train_mod.deep_get(cfg, "moss.model_path")
    codec_path = args.codec_path or train_mod.deep_get(cfg, "moss.codec_path")
    n_vq = int(args.n_vq or train_mod.deep_get(cfg, "training.n_vq", train_mod.deep_get(cfg, "moss.default_n_vq", 32)))
    if moss_root and str(moss_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(moss_root))
    from moss_tts_delay.finetuning.dataset import MossTTSSFTDataset

    class LazyMossTTSSFTDataset(MossTTSSFTDataset):
        def __init__(self, records, processor, n_vq=None) -> None:
            self.records = records
            self.processor = processor
            self.n_vq = n_vq
            self._audio_cache = {}

    train_args = argparse.Namespace(
        train_jsonl=None,
        train_jsonl_spec=args.train_jsonl_spec,
        max_rows=int(args.batch_max_rows),
        jsonl_index_path=None,
        rebuild_jsonl_index=False,
    )
    records = train_mod.load_training_records(train_args)
    processor = train_mod.build_processor(model_path, codec_path, moss_root)
    base = LazyMossTTSSFTDataset(records=records, processor=processor, n_vq=n_vq)
    dataset = train_mod.MossCodecVCTimbreSFTDataset(
        records=records,
        base_dataset=base,
        n_vq=n_vq,
        audio_pad_code=int(processor.model_config.audio_pad_code),
        content_tokenizer=None,
        content_ctc_token_offset=1,
        timbre_side_only=bool(args.timbre_side_only),
    )
    batch_size = max(1, int(args.batch_size))
    max_batches = max(1, int(args.max_batches))
    rows: list[dict[str, Any]] = []
    mode_counts: dict[str, int] = {}
    source_path_available = 0
    for batch_idx in range(max_batches):
        start = batch_idx * batch_size
        if start >= len(dataset):
            break
        indices = list(range(start, min(start + batch_size, len(dataset))))
        raw_records = [records[idx] for idx in indices]
        items = [dataset[idx] for idx in indices]
        batch = dataset.collate_fn(items)
        mode_ids = batch.get("vc_mode_id")
        if torch.is_tensor(mode_ids):
            for mode_id in mode_ids.detach().cpu().tolist():
                key = str(int(mode_id))
                mode_counts[key] = mode_counts.get(key, 0) + 1
        for record in raw_records:
            if record_value(record, *SOURCE_KEYS):
                source_path_available += 1
        source_features = batch.get("source_semantic_features")
        source_mask = batch.get("source_semantic_features_mask")
        target_mask = batch.get("target_assistant_positions")
        role_ids = batch.get("role_ids")
        row: dict[str, Any] = {
            "batch_idx": batch_idx,
            "indices": indices,
            "input_ids_shape": list(batch["input_ids"].shape),
            "labels_shape": list(batch["labels"].shape),
            "target_mask_count": int(target_mask.sum().item()) if torch.is_tensor(target_mask) else None,
            "target_mask_ratio": float(target_mask.float().mean().item()) if torch.is_tensor(target_mask) else None,
        }
        if torch.is_tensor(target_mask) and bool(target_mask.any().item()):
            active = torch.nonzero(target_mask, as_tuple=False)
            row["target_first_index"] = int(active[:, 1].min().item())
            row["target_last_index"] = int(active[:, 1].max().item())
        if torch.is_tensor(role_ids):
            from moss_codecvc.roles import count_roles

            row["role_counts"] = count_roles(role_ids).as_dict()
        if torch.is_tensor(source_features):
            row["source_semantic_features_shape"] = list(source_features.shape)
            if torch.is_tensor(source_mask):
                lengths = source_mask.long().sum(dim=1)
                row["source_semantic_mask_shape"] = list(source_mask.shape)
                row["source_semantic_available_ratio"] = float(lengths.gt(0).float().mean().item())
                row["source_semantic_length_min"] = int(lengths.min().item())
                row["source_semantic_length_max"] = int(lengths.max().item())
                row["source_semantic_length_mean"] = float(lengths.float().mean().item())
                row["source_semantic_tensor"] = tensor_summary(source_features, source_mask)
            else:
                row["source_semantic_tensor"] = tensor_summary(source_features)
        else:
            row["source_semantic_features_shape"] = None
        rows.append(row)
    total_items = sum(len(row["indices"]) for row in rows)
    return {
        "train_jsonl_spec": args.train_jsonl_spec,
        "records_effective": len(records),
        "batch_size": batch_size,
        "batches_scanned": len(rows),
        "items_scanned": total_items,
        "source_feature_path_available_ratio": source_path_available / max(1, total_items),
        "mode_id_counts": mode_counts,
        "batches": rows,
    }


def compare_offline_online(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.compare_audio:
        return None
    if not args.offline_feature_path:
        raise ValueError("--compare-audio requires --offline-feature-path")
    infer_mod = load_module_from_file("moss_codecvc_infer_mod", ROOT / "scripts/003001_infer_moss_codecvc.py")
    offline = load_feature(args.offline_feature_path)
    if offline is None:
        raise ValueError(f"failed to load offline feature: {args.offline_feature_path}")
    model_name_or_path = args.hubert_model or str(getattr(infer_mod, "DEFAULT_HUBERT_MODEL"))
    cache_dir = args.hubert_cache_dir or str(getattr(infer_mod, "DEFAULT_HUBERT_CACHE_DIR"))
    online = infer_mod.extract_source_semantic_features_online(
        audio_path=args.compare_audio,
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
        local_files_only=bool(args.local_files_only),
        layer=int(args.hubert_layer),
        device=args.online_device,
        dtype_name=args.online_dtype,
        downsample_stride=int(args.downsample_stride),
    )
    pooled_cos = float(F.cosine_similarity(pooled(offline), pooled(online), dim=0).item())
    if int(offline.shape[-1]) == int(online.shape[-1]):
        online_interp = F.interpolate(
            online.T.unsqueeze(0),
            size=int(offline.shape[0]),
            mode="linear",
            align_corners=False,
        ).squeeze(0).T
        frame_cos = F.cosine_similarity(offline.float(), online_interp.float(), dim=-1)
        frame_cos_mean = float(frame_cos.mean().item())
    else:
        frame_cos_mean = None
    return {
        "audio": args.compare_audio,
        "offline_feature_path": args.offline_feature_path,
        "online_model": model_name_or_path,
        "online_layer": int(args.hubert_layer),
        "online_downsample_stride": int(args.downsample_stride),
        "offline_shape": list(offline.shape),
        "online_shape": list(online.shape),
        "offline_summary": tensor_summary(offline),
        "online_summary": tensor_summary(online),
        "pooled_cosine": pooled_cos,
        "frame_cosine_after_interp_mean": frame_cos_mean,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Ver2.5 source semantic feature availability and basic separability.")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--train-jsonl-spec", default="")
    ap.add_argument("--config", default=str(ROOT / "configs/remote_full.yaml"))
    ap.add_argument("--model-path", default="")
    ap.add_argument("--codec-path", default="")
    ap.add_argument("--n-vq", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=2)
    ap.add_argument("--batch-max-rows", type=int, default=0)
    ap.add_argument("--timbre-side-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--compare-audio", default="")
    ap.add_argument("--offline-feature-path", default="")
    ap.add_argument("--hubert-model", default="")
    ap.add_argument("--hubert-cache-dir", default="")
    ap.add_argument("--hubert-layer", type=int, default=9)
    ap.add_argument("--downsample-stride", type=int, default=1)
    ap.add_argument("--online-device", default="cpu")
    ap.add_argument("--online-dtype", default="float32")
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/debug_semantic_memory"))
    ap.add_argument("--max-rows", type=int, default=2000)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {}
    if args.manifest:
        manifest = Path(args.manifest).expanduser()
    else:
        manifest = None

    rows = 0
    source_ok = 0
    target_ok = 0
    paired_sims: list[float] = []
    source_lengths: list[int] = []
    target_lengths: list[int] = []
    dims: dict[int, int] = {}
    norm_values: list[float] = []
    bad_examples: list[dict[str, Any]] = []

    if manifest is not None:
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if args.max_rows > 0 and rows >= args.max_rows:
                    break
                if not line.strip():
                    continue
                rows += 1
                record = json.loads(line)
                source_path = record_value(record, *SOURCE_KEYS)
                target_path = record_value(record, *TARGET_KEYS)
                source = load_feature(source_path)
                target = load_feature(target_path)
                if source is None:
                    bad_examples.append({"row": rows, "reason": "missing_source_feature", "sample_id": record.get("sample_id"), "path": source_path})
                    continue
                source_ok += 1
                source_lengths.append(int(source.shape[0]))
                dims[int(source.shape[-1])] = dims.get(int(source.shape[-1]), 0) + 1
                finite = torch.isfinite(source).all()
                norm_values.append(float(source.norm(dim=-1).mean().item()) if bool(finite.item()) else float("nan"))
                if target is not None:
                    target_ok += 1
                    target_lengths.append(int(target.shape[0]))
                    paired_sims.append(float(torch.nn.functional.cosine_similarity(pooled(source), pooled(target), dim=0).item()))

    def mean(values: list[float]) -> float | None:
        clean = [value for value in values if value == value]
        return sum(clean) / len(clean) if clean else None

    if manifest is not None:
        report["manifest_audit"] = {
            "manifest": str(manifest),
            "rows_scanned": rows,
            "source_available": source_ok,
            "target_available": target_ok,
            "source_available_ratio": source_ok / max(1, rows),
            "target_available_ratio": target_ok / max(1, rows),
            "feature_dims": dims,
            "source_length_min": min(source_lengths) if source_lengths else None,
            "source_length_max": max(source_lengths) if source_lengths else None,
            "source_length_mean": mean([float(v) for v in source_lengths]),
            "target_length_mean": mean([float(v) for v in target_lengths]),
            "source_feature_norm_mean": mean(norm_values),
            "source_target_pooled_cos_mean": mean(paired_sims),
            "bad_example_count": len(bad_examples),
        }
    batch_report = audit_train_batch(args)
    if batch_report is not None:
        report["batch_audit"] = batch_report
    compare_report = compare_offline_online(args)
    if compare_report is not None:
        report["offline_online_compare"] = compare_report
    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with (output_dir / "audit_examples.jsonl").open("w", encoding="utf-8") as handle:
        for item in bad_examples[:200]:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
