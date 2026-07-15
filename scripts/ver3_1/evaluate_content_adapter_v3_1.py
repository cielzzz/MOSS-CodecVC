#!/usr/bin/env python
"""Evaluate a ver3.1 content adapter on manifest rows.

The v1 manifests expose SentencePiece ``content_token_ids`` rather than
MFA/WhisperX phoneme alignments.  The output therefore names the metric
``content_token_proxy`` and never presents it as phoneme accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.models.content_adapter_v3_1 import ContentAdapterV31
from moss_codecvc.models.content_cross_attn import compute_phoneme_classifier_loss

# Reuse the audited manifest/feature selection and batching helpers.
from train_content_adapter import (
    collate,
    next_batch,
    parse_input_specs,
    parse_mode_filter,
    stream_items,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", required=True, metavar="SPLIT=JSONL")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--mode-filter", default="no_text")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", choices=("float32", "bf16", "fp16"), default="bf16")
    return ap.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[ContentAdapterV31, dict[str, Any]]:
    path = path.expanduser().resolve()
    config_path = path / "adapter_config.json" if path.is_dir() else None
    if config_path is not None and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        state_path = path / "adapter.pt"
    else:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        config = dict(payload["config"])
        state_path = path
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
    payload = torch.load(str(state_path), map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, config


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "float32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    model, config = load_checkpoint(Path(args.checkpoint), device)
    mode_filter = parse_mode_filter(args.mode_filter)
    iterator = stream_items(
        parse_input_specs(args.input),
        feature_key=str(config.get("feature_key", "source_wavlm_bnf_features_path")),
        label_key=str(config.get("label_key_requested", "auto")),
        mode_filter=mode_filter,
        allow_target_feature_for_text=False,
        max_rows=int(args.max_rows),
    )
    total_rows = 0
    total_tokens = 0.0
    loss_sum = 0.0
    acc_sum = 0.0
    frames_sum = 0.0
    batches = 0
    while True:
        batch, iterator = next_batch(iterator, max(1, int(args.batch_size)))
        if not batch:
            break
        if len(batch) < max(1, int(args.batch_size)):
            # Evaluation accepts a final short batch.
            pass
        features, feature_mask, labels, label_mask = collate(batch)
        with autocast_context(device, args.precision):
            output = model(features.to(device), feature_mask.to(device))
            loss, stats = compute_phoneme_classifier_loss(
                output.logits,
                labels.to(device),
                label_mask.to(device),
            )
        if loss is None:
            continue
        valid = float(stats.get("content_phoneme_classifier_valid_tokens", 0.0))
        total_rows += len(batch)
        total_tokens += valid
        loss_sum += float(loss.float().item()) * valid
        acc_sum += float(stats.get("content_phoneme_classifier_acc", 0.0)) * valid
        frames_sum += float(output.semantic_mask.sum().item())
        batches += 1
        if int(args.max_rows) > 0 and total_rows >= int(args.max_rows):
            break
    if total_tokens <= 0:
        raise RuntimeError("no valid content-token labels were evaluated")
    result = {
        "schema": "ver3_1_content_adapter_eval_v1",
        "status": "completed",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "input": [f"{split}={path}" for split, path in parse_input_specs(args.input)],
        "mode_filter": sorted(mode_filter) if mode_filter is not None else ["all"],
        "label_semantics": "sentencepiece_content_token_proxy_positional_ce",
        "rows": total_rows,
        "batches": batches,
        "valid_tokens": int(total_tokens),
        "mean_loss": loss_sum / total_tokens,
        "content_token_proxy_accuracy": acc_sum / total_tokens,
        "mean_output_frames": frames_sum / max(1, total_rows),
        "adapter_config": config,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
