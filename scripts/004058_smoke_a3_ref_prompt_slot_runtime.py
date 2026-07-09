#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.models import MossCodecVCTimbreMemoryWrapper, TimbreMemoryConfig


def load_fake_model_class():
    path = ROOT / "scripts/004003_smoke_timbre_memory.py"
    spec = importlib.util.spec_from_file_location("moss_codecvc_smoke_timbre_memory", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load fake model from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FakeMossDelayModel


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke A3 T_ref-token prompt-slot injection with K=16.")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    torch.manual_seed(13)
    token_count = int(args.tokens)
    batch_size = 1
    seq_len = 40
    n_vq = 4
    hidden_size = 16
    audio_vocab_size = 20
    fake_model_cls = load_fake_model_class()
    model = fake_model_cls(n_vq=n_vq, hidden_size=hidden_size, audio_vocab_size=audio_vocab_size)
    for param in model.parameters():
        param.requires_grad = False

    wrapper = MossCodecVCTimbreMemoryWrapper(
        model,
        TimbreMemoryConfig(
            enabled=True,
            timbre_side_only=True,
            num_memory_tokens=token_count,
            adapter_layers="last_2",
            num_heads=4,
            adapter_dim=32,
            dropout=0.0,
            speaker_embedding_dim=0,
            ref_speaker_prompt_tokens=token_count,
            ref_speaker_prompt_mode="slot",
            ref_speaker_prompt_token_source="timbre_memory",
            ref_speaker_prompt_slot=True,
            ref_speaker_prompt_slot_code=0,
            ref_speaker_prompt_slot_pack_mode="audio_like",
            ref_speaker_prompt_output_norm=True,
            ref_speaker_prompt_output_scale=0.02,
            target_front_ce_weight=4.0,
            target_front_ce_seconds=0.75,
            use_role_routing=True,
            route_loss_weight=0.01,
            prosody_memory_tokens=3,
            target_head_routing=True,
        ),
    )
    if wrapper.ref_speaker_prompt is not None:
        raise AssertionError("A3 should not create speaker_mlp ref_speaker_prompt")

    input_ids = torch.zeros(batch_size, seq_len, n_vq + 1, dtype=torch.long)
    input_ids[..., 0] = torch.randint(0, model.config.language_config.vocab_size, (batch_size, seq_len))
    input_ids[..., 1:] = torch.randint(0, audio_vocab_size, (batch_size, seq_len, n_vq))
    labels = torch.full_like(input_ids, -100)
    labels[:, -5:, 1:] = torch.randint(0, audio_vocab_size, (batch_size, 5, n_vq))
    source_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    source_positions[:, :5] = True
    timbre_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    timbre_positions[:, 5 : 5 + token_count] = True
    slot_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    slot_positions[:, 5 : 5 + token_count] = True
    target_positions = (labels != -100).any(dim=-1)
    timbre_ref_codes = torch.randint(0, audio_vocab_size, (batch_size, 30, n_vq))
    timbre_ref_mask = torch.ones(batch_size, 30, dtype=torch.bool)

    outputs = wrapper(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
        labels=labels,
        timbre_ref_codes=timbre_ref_codes,
        timbre_ref_mask=timbre_ref_mask,
        target_position_mask=target_positions,
        source_prompt_positions=source_positions,
        timbre_ref_prompt_positions=timbre_positions,
        ref_speaker_prompt_slot_positions=slot_positions,
    )
    if outputs.loss is None or not torch.isfinite(outputs.loss):
        raise AssertionError("A3 smoke did not produce a finite loss")
    stats = dict(wrapper.last_ref_speaker_prompt_slot_stats or {})
    wrote = int(round(float(stats.get("ref_speaker_prompt_slot_wrote", -1))))
    span = int(slot_positions.sum().item())
    token_source = float(stats.get("ref_speaker_prompt_slot_token_source", 0.0))
    if span != token_count:
        raise AssertionError(f"slot span length {span} != expected {token_count}")
    if wrote != token_count:
        raise AssertionError(f"ref_speaker_prompt_slot_wrote {wrote} != expected {token_count}")
    if token_source != 1.0:
        raise AssertionError(f"expected timbre_memory token source stat 1.0, got {token_source}")
    payload = {
        "status": "ok",
        "ref_speaker_prompt_tokens": token_count,
        "slot_span_length": span,
        "ref_speaker_prompt_slot_wrote": wrote,
        "ref_speaker_prompt_slot_token_source": token_source,
        "ref_speaker_prompt_is_none": wrapper.ref_speaker_prompt is None,
        "loss": float(outputs.loss.detach().item()),
        "slot_stats": stats,
        "forward_debug": wrapper.last_forward_debug,
    }
    if args.output_json:
        out = Path(args.output_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
