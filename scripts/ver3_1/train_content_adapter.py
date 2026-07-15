#!/usr/bin/env python
"""Train the ver3.1 WavLM -> 12.5-Hz content adapter.

This is a deliberately small, auditable pre-training loop.  It consumes the
already materialised v1 WavLM layer-9 features when they exist and never loads
the MOSS-TTS decoder.  The default label fallback is the manifest's
``content_token_ids`` (SentencePiece tokens); true MFA/WhisperX phoneme IDs
can be used by adding a ``phoneme_ids``/``phoneme_token_ids`` field to a
manifest.  The summary records which label source was actually used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.models.content_adapter_v3_1 import ContentAdapterV31, count_parameters
from moss_codecvc.models.content_cross_attn import compute_phoneme_classifier_loss


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="SPLIT=JSONL",
        help="One or more manifests, e.g. no_text=/path/no_text.train.jsonl",
    )
    ap.add_argument("--output-root", required=True)
    ap.add_argument(
        "--feature-key",
        default="source_wavlm_bnf_features_path",
        help="Feature sidecar field. Step-3 Path-A defaults to source BNF.",
    )
    ap.add_argument(
        "--mode-filter",
        default="no_text",
        help="Comma-separated moss_codecvc_mode values; use 'all' to include every row. ",
    )
    ap.add_argument(
        "--allow-target-feature-for-text",
        action="store_true",
        help="Explicitly permit target BNF for text rows when source BNF is absent. "
        "This is speaker-leaky and is disabled by default.",
    )
    ap.add_argument("--label-key", default="auto")
    ap.add_argument("--input-dim", default="auto")
    ap.add_argument("--semantic-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--downsample-kernel", type=int, default=4)
    ap.add_argument("--downsample-stride", type=int, default=4)
    ap.add_argument("--classifier-adapter-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=1.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--log-steps", type=int, default=50)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", choices=("float32", "bf16", "fp16"), default="bf16")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def parse_input_specs(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--input must be SPLIT=JSONL, got {value!r}")
        split, path = value.split("=", 1)
        split = split.strip()
        if not split or not path:
            raise ValueError(f"invalid --input {value!r}")
        result.append((split, Path(path).expanduser().resolve()))
    return result


def nested_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def row_mode(row: dict[str, Any]) -> str:
    value = row.get("moss_codecvc_mode")
    if value in (None, ""):
        meta = row.get("moss_codecvc_meta")
        if isinstance(meta, dict):
            value = meta.get("moss_codecvc_mode") or meta.get("mode")
    return str(value or "unknown")


def parse_mode_filter(value: str) -> set[str] | None:
    values = {part.strip() for part in str(value).split(",") if part.strip()}
    if not values or "all" in values:
        return None
    return values


def select_feature_path(
    row: dict[str, Any],
    *,
    feature_key: str,
    allow_target_feature_for_text: bool,
) -> tuple[str | None, str]:
    """Select a speaker-safe WavLM sidecar.

    Text rows in the legacy v1 manifest intentionally have no source BNF
    sidecar: the source audio is a prosody/style carrier and its content does
    not match the target text labels.  We therefore refuse to silently train
    against ``target_wavlm_bnf_features_path``.  A caller may opt in for an
    explicitly exploratory, speaker-leaky run and the provenance records that
    choice.
    """
    source_key = feature_key
    if source_key in ("auto", "source_wavlm_bnf_features_path"):
        source_key = "source_wavlm_bnf_features_path"
    value = nested_value(row, source_key)
    if value not in (None, ""):
        return str(value), "source_bnf_sidecar"
    mode = row_mode(row)
    if mode == "text" and allow_target_feature_for_text:
        target = nested_value(row, "target_wavlm_bnf_features_path")
        if target not in (None, ""):
            return str(target), "target_bnf_leaky_opt_in"
    return None, "missing_source_bnf"


def iter_rows(specs: list[tuple[str, Path]]) -> Iterator[tuple[str, dict[str, Any]]]:
    for split, path in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
                yield split, row


def load_feature(path: str | Path) -> torch.Tensor:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        value = None
        for key in ("wavlm_bnf_features", "wavlm_features", "semantic_features"):
            candidate = payload.get(key)
            if torch.is_tensor(candidate):
                value = candidate
                break
    elif torch.is_tensor(payload):
        value = payload
    else:
        value = None
    if value is None:
        raise ValueError(f"feature file has no tensor payload: {path}")
    value = value.detach().float().cpu()
    if value.dim() != 2:
        raise ValueError(f"feature tensor must be [T,D], got {tuple(value.shape)}: {path}")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"empty feature tensor: {path}")
    return value


def choose_label_key(row: dict[str, Any], requested: str) -> str | None:
    if requested != "auto":
        value = row.get(requested)
        return requested if isinstance(value, list) and value else None
    for key in ("phoneme_ids", "phoneme_token_ids", "content_phoneme_ids", "content_token_ids"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return key
    return None


def row_to_item(
    split: str,
    row: dict[str, Any],
    *,
    feature_key: str,
    label_key: str,
    mode_filter: set[str] | None,
    allow_target_feature_for_text: bool,
) -> dict[str, Any] | None:
    mode = row_mode(row)
    if mode_filter is not None and mode not in mode_filter:
        return None
    feature_path, feature_kind = select_feature_path(
        row,
        feature_key=feature_key,
        allow_target_feature_for_text=allow_target_feature_for_text,
    )
    if feature_path in (None, ""):
        return None
    selected_label_key = choose_label_key(row, label_key)
    if selected_label_key is None:
        return None
    labels = [int(value) for value in row[selected_label_key]]
    if not labels:
        return None
    return {
        "split": split,
        "sample_id": str(row.get("sample_id") or row.get("utt_id") or "row"),
        "feature_path": str(feature_path),
        "labels": labels,
        "label_key": selected_label_key,
        "mode": mode,
        "feature_kind": feature_kind,
    }


def scan_specs(
    specs: list[tuple[str, Path]],
    *,
    feature_key: str,
    label_key: str,
    mode_filter: set[str] | None,
    allow_target_feature_for_text: bool,
    max_probe: int = 256,
) -> dict[str, Any]:
    input_dim: int | None = None
    vocab_size = 0
    label_counts: dict[str, int] = {}
    feature_rows = 0
    skipped = 0
    examples: list[dict[str, Any]] = []
    for split, row in iter_rows(specs):
        item = row_to_item(
            split,
            row,
            feature_key=feature_key,
            label_key=label_key,
            mode_filter=mode_filter,
            allow_target_feature_for_text=allow_target_feature_for_text,
        )
        if item is None:
            skipped += 1
            continue
        feature_rows += 1
        label_counts[item["label_key"]] = label_counts.get(item["label_key"], 0) + 1
        vocab_size = max(vocab_size, max(item["labels"]) + 1)
        metadata_vocab = row.get("content_ctc_vocab_size")
        if metadata_vocab not in (None, ""):
            try:
                vocab_size = max(vocab_size, int(metadata_vocab))
            except (TypeError, ValueError):
                pass
        if len(examples) < max_probe:
            examples.append(item)
            if input_dim is None:
                input_dim = int(load_feature(item["feature_path"]).shape[-1])
        if feature_rows >= max_probe:
            break
    if input_dim is None:
        raise RuntimeError("no rows with both feature and labels were found")
    return {
        "input_dim": input_dim,
        "vocab_size_from_labels": vocab_size,
        "label_counts_probe": label_counts,
        "feature_rows_probe": feature_rows,
        "skipped_rows_probe": skipped,
        "mode_filter": sorted(mode_filter) if mode_filter is not None else ["all"],
        "feature_kind_probe": sorted({item["feature_kind"] for item in examples}),
        "examples": examples[:3],
    }


def collate(items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    features = [load_feature(item["feature_path"]) for item in items]
    max_frames = max(int(value.shape[0]) for value in features)
    feature_dim = int(features[0].shape[1])
    batch_features = torch.zeros((len(items), max_frames, feature_dim), dtype=torch.float32)
    feature_mask = torch.zeros((len(items), max_frames), dtype=torch.bool)
    for index, value in enumerate(features):
        if int(value.shape[1]) != feature_dim:
            raise ValueError(f"mixed feature dimensions in one batch: {value.shape[1]} != {feature_dim}")
        frames = int(value.shape[0])
        batch_features[index, :frames] = value
        feature_mask[index, :frames] = True
    max_labels = max(len(item["labels"]) for item in items)
    labels = torch.full((len(items), max_labels), -1, dtype=torch.long)
    label_mask = torch.zeros((len(items), max_labels), dtype=torch.bool)
    for index, item in enumerate(items):
        values = torch.tensor(item["labels"], dtype=torch.long)
        labels[index, : values.numel()] = values
        label_mask[index, : values.numel()] = True
    return batch_features, feature_mask, labels, label_mask


def stream_items(
    specs: list[tuple[str, Path]],
    *,
    feature_key: str,
    label_key: str,
    mode_filter: set[str] | None,
    allow_target_feature_for_text: bool,
    max_rows: int,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Iterator[dict[str, Any]]:
    yielded = 0
    for global_index, (split, row) in enumerate(iter_rows(specs)):
        if int(global_index) % max(1, int(num_shards)) != int(shard_index):
            continue
        item = row_to_item(
            split,
            row,
            feature_key=feature_key,
            label_key=label_key,
            mode_filter=mode_filter,
            allow_target_feature_for_text=allow_target_feature_for_text,
        )
        if item is None:
            continue
        yield item
        yielded += 1
        if int(max_rows) > 0 and yielded >= int(max_rows):
            return


def next_batch(iterator: Iterator[dict[str, Any]], batch_size: int) -> tuple[list[dict[str, Any]], Iterator[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    while len(batch) < int(batch_size):
        try:
            batch.append(next(iterator))
        except StopIteration:
            return batch, iter(())
    return batch, iterator


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(output_root: Path, step: int, model: ContentAdapterV31, optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> Path:
    ckpt = output_root / f"step-{int(step):06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": int(step), "config": config}, ckpt / "adapter.pt")
    atomic_json(ckpt / "adapter_config.json", config)
    atomic_json(output_root / "LATEST.json", {"step": int(step), "path": str(ckpt.resolve())})
    return ckpt


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "float32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    specs = parse_input_specs(args.input)
    mode_filter = parse_mode_filter(args.mode_filter)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scan = scan_specs(
        specs,
        feature_key=args.feature_key,
        label_key=args.label_key,
        mode_filter=mode_filter,
        allow_target_feature_for_text=bool(args.allow_target_feature_for_text),
    )
    if str(args.input_dim).lower() == "auto":
        input_dim = int(scan["input_dim"])
    else:
        input_dim = int(args.input_dim)
        if input_dim != int(scan["input_dim"]):
            raise ValueError(f"requested input_dim={input_dim}, manifest feature dim={scan['input_dim']}")
    vocab_size = max(2, int(scan["vocab_size_from_labels"]))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed adapter training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    model = ContentAdapterV31(
        input_dim=input_dim,
        semantic_dim=int(args.semantic_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        downsample_kernel_size=int(args.downsample_kernel),
        downsample_stride=int(args.downsample_stride),
        vocab_size=vocab_size,
        classifier_adapter_dim=int(args.classifier_adapter_dim),
    ).to(device)
    model_for_save = model
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    config = {
        "schema": "ver3_1_content_adapter_v1",
        "input_dim": input_dim,
        "semantic_dim": int(args.semantic_dim),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "downsample_kernel": int(args.downsample_kernel),
        "downsample_stride": int(args.downsample_stride),
        "vocab_size": vocab_size,
        "feature_key": args.feature_key,
        "mode_filter": sorted(mode_filter) if mode_filter is not None else ["all"],
        "allow_target_feature_for_text": bool(args.allow_target_feature_for_text),
        "label_key_requested": args.label_key,
        "label_semantics": "sentencepiece_content_token_proxy_positional_ce",
        "label_source_probe": scan["label_counts_probe"],
        "feature_kind_probe": scan["feature_kind_probe"],
        "parameter_count": count_parameters(model),
        "trainable_parameter_count": count_parameters(model, trainable_only=True),
        "input_manifests": {split: str(path) for split, path in specs},
        "seed": int(args.seed),
    }
    if rank == 0:
        atomic_json(output_root / "adapter_config.json", config)
    if distributed:
        dist.barrier()
    if args.dry_run:
        iterator = stream_items(
            specs,
            feature_key=args.feature_key,
            label_key=args.label_key,
            mode_filter=mode_filter,
            allow_target_feature_for_text=bool(args.allow_target_feature_for_text),
            max_rows=args.max_rows,
            shard_index=rank,
            num_shards=world_size,
        )
        batch, _ = next_batch(iterator, max(1, int(args.batch_size)))
        if not batch:
            raise RuntimeError("dry-run found no usable batch")
        features, feature_mask, labels, label_mask = collate(batch)
        with torch.no_grad():
            out = model(features.to(device), feature_mask.to(device))
            loss, stats = compute_phoneme_classifier_loss(out.logits, labels.to(device), label_mask.to(device))
        payload = {
            "status": "dry_run_pass",
            "batch": len(batch),
            "input_shape": list(features.shape),
            "output_shape": list(out.semantic.shape),
            "output_mask_shape": list(out.semantic_mask.shape),
            "loss": float(loss.item()) if loss is not None else None,
            "stats": stats,
            "config": config,
        }
        if rank == 0:
            atomic_json(output_root / "DRY_RUN.json", payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    metrics_path = output_root / "train_metrics.jsonl"
    iterator = stream_items(
        specs,
        feature_key=args.feature_key,
        label_key=args.label_key,
        mode_filter=mode_filter,
        allow_target_feature_for_text=bool(args.allow_target_feature_for_text),
        max_rows=args.max_rows,
        shard_index=rank,
        num_shards=world_size,
    )
    step = 0
    epoch = 0
    running_loss: list[float] = []
    started = time.time()
    metrics_file = metrics_path.open("w", encoding="utf-8") if rank == 0 else None
    try:
        while step < int(args.max_steps):
            batch, iterator = next_batch(iterator, max(1, int(args.batch_size)))
            if len(batch) < max(1, int(args.batch_size)):
                epoch += 1
                iterator = stream_items(
                    specs,
                    feature_key=args.feature_key,
                    label_key=args.label_key,
                    mode_filter=mode_filter,
                    allow_target_feature_for_text=bool(args.allow_target_feature_for_text),
                    max_rows=args.max_rows,
                    shard_index=rank,
                    num_shards=world_size,
                )
                needed = max(1, int(args.batch_size)) - len(batch)
                if needed > 0:
                    remainder, iterator = next_batch(iterator, needed)
                    batch.extend(remainder)
                if not batch:
                    raise RuntimeError("training stream yielded no usable rows")
                if len(batch) < max(1, int(args.batch_size)):
                    raise RuntimeError(
                        "training stream exhausted before a full batch; "
                        f"got {len(batch)}/{max(1, int(args.batch_size))} usable rows"
                    )
            features, feature_mask, labels, label_mask = collate(batch)
            features = features.to(device, non_blocking=True)
            feature_mask = feature_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_mask = label_mask.to(device, non_blocking=True)
            step += 1
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                out = model(features, feature_mask)
                loss, stats = compute_phoneme_classifier_loss(out.logits, labels, label_mask)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(f"invalid adapter loss at step {step}: {loss}")
            loss.backward()
            grad_norm = float(clip_grad_norm_(model.parameters(), float(args.grad_clip)).item())
            if int(args.warmup_steps) > 0:
                scale = min(1.0, step / float(args.warmup_steps))
                for group in optimizer.param_groups:
                    group["lr"] = float(args.learning_rate) * scale
            optimizer.step()
            loss_value = float(loss.detach().float().item())
            running_loss.append(loss_value)
            record = {
                "step": step,
                "epoch": epoch,
                "loss": loss_value,
                "grad_norm": grad_norm,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "batch": len(batch),
                "input_frames": float(feature_mask.sum(dim=1).float().mean().item()),
                "output_frames": float(out.semantic_mask.sum(dim=1).float().mean().item()),
                **{key: float(value) for key, value in stats.items()},
            }
            if step == 1 or step % max(1, int(args.log_steps)) == 0 or step == int(args.max_steps):
                if rank == 0:
                    assert metrics_file is not None
                    metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics_file.flush()
                    print(
                        f"[adapter] step={step}/{args.max_steps} loss={loss_value:.4f} "
                        f"acc={float(stats.get('content_phoneme_classifier_acc', 0.0)):.4f} "
                        f"out_frames={record['output_frames']:.1f}",
                        flush=True,
                    )
            if step % max(1, int(args.save_steps)) == 0 or step == int(args.max_steps):
                if rank == 0:
                    save_checkpoint(output_root, step, model_for_save, optimizer, config)
                if distributed:
                    dist.barrier()
    finally:
        if metrics_file is not None:
            metrics_file.close()
    summary = {
        "schema": "ver3_1_content_adapter_train_summary_v1",
        "status": "completed",
        "steps": int(step),
        "loss_last": running_loss[-1] if running_loss else None,
        "loss_mean_last_100": sum(running_loss[-100:]) / max(1, len(running_loss[-100:])),
        "elapsed_seconds": time.time() - started,
        "config": config,
        "latest": str((output_root / f"step-{step:06d}").resolve()),
    }
    if rank == 0:
        atomic_json(output_root / "COMPLETED.json", summary)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
