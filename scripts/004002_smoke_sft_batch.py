#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config


def make_codes(length: int, n_vq: int, audio_vocab_size: int, offset: int) -> list[list[int]]:
    values = torch.arange(length * n_vq, dtype=torch.long).reshape(length, n_vq)
    return ((values + offset) % audio_vocab_size).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Smoke-test MOSS-CodecVC SFT packing without loading model weights."
    )
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--n-vq", type=int, default=None)
    ap.add_argument("--length", type=int, default=6)
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    moss_root = deep_get(cfg, "moss.root")
    model_path = args.model_path or deep_get(cfg, "moss.model_path")
    n_vq = args.n_vq or int(deep_get(cfg, "training.n_vq", deep_get(cfg, "moss.default_n_vq", 32)))

    if moss_root and str(moss_root) not in sys.path:
        sys.path.insert(0, str(moss_root))

    from moss_tts_delay.finetuning.dataset import MossTTSSFTDataset
    from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor

    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if int(getattr(model_config, "n_vq", 0) or 0) != n_vq:
        raise ValueError(f"n_vq mismatch: requested {n_vq}, model config has {model_config.n_vq}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    processor = MossTTSDelayProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=None,
        model_config=model_config,
    )

    record = {
        "sample_id": "smoke_sft_batch",
        "text": "Smoke test for MOSS CodecVC.",
        "instruction": (
            "Voice conversion task. [S1] carries content and prosody. "
            "[S2] carries target timbre."
        ),
        "language": "en",
        "quality": "high",
        "tokens": args.length,
        "reference_audio_codes": [
            make_codes(args.length, n_vq, model_config.audio_vocab_size, 0),
            make_codes(max(1, args.length - 1), n_vq, model_config.audio_vocab_size, 17),
        ],
        "audio_codes": make_codes(args.length, n_vq, model_config.audio_vocab_size, 33),
    }

    dataset = MossTTSSFTDataset([record], processor=processor, n_vq=n_vq)
    item = dataset[0]
    batch = dataset.collate_fn([item])

    valid_labels = int((batch["labels"] != -100).sum().item())
    print(f"model_type={model_config.model_type} n_vq={model_config.n_vq}")
    print(f"item.input_ids={tuple(item['input_ids'].shape)}")
    print(f"item.loss_mask={tuple(item['loss_mask'].shape)} valid={int(item['loss_mask'].sum().item())}")
    print(f"batch.input_ids={tuple(batch['input_ids'].shape)}")
    print(f"batch.attention_mask={tuple(batch['attention_mask'].shape)} dtype={batch['attention_mask'].dtype}")
    print(f"batch.labels={tuple(batch['labels'].shape)} valid={valid_labels}")

    if batch["input_ids"].shape[-1] != n_vq + 1:
        raise AssertionError("input_ids channel dimension does not equal n_vq + 1")
    if batch["labels"].shape[-1] != n_vq + 1:
        raise AssertionError("labels channel dimension does not equal n_vq + 1")
    if valid_labels <= 0:
        raise AssertionError("labels contain no supervised positions")

    print("SFT batch smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
