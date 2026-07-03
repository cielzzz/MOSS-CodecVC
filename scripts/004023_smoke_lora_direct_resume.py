#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "002002_train_moss_codecvc_lora.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("moss_codecvc_train_lora", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummySelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4096, 4096, bias=False, device="meta")
        self.k_proj = nn.Linear(4096, 1024, bias=False, device="meta")
        self.v_proj = nn.Linear(4096, 1024, bias=False, device="meta")
        self.o_proj = nn.Linear(4096, 4096, bias=False, device="meta")


class DummyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4096, 12288, bias=False, device="meta")
        self.up_proj = nn.Linear(4096, 12288, bias=False, device="meta")
        self.down_proj = nn.Linear(12288, 4096, bias=False, device="meta")


class DummyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = DummySelfAttention()
        self.mlp = DummyMLP()


class DummyConfig:
    n_vq = 32
    model_type = "moss_tts_delay_dummy"


class DummyMossDelayModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = DummyConfig()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([DummyLayer() for _ in range(36)])

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs


def build_dummy_peft_model(train_module):
    model = DummyMossDelayModel()
    return get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=list(train_module.LORA_TARGET_MODULES),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test direct LoRA resume key mapping without loading full MOSS weights.")
    parser.add_argument("adapter_dirs", nargs="+", help="Checkpoint directories containing adapter_model.safetensors.")
    args = parser.parse_args()

    train_module = load_train_module()
    for adapter_dir in args.adapter_dirs:
        model = build_dummy_peft_model(train_module)
        train_module.load_lora_adapter_direct(model, adapter_dir)
        print(f"[smoke-ok] {adapter_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
