#!/usr/bin/env python
"""Train the ver3.1 dequantized-latent CFM probe.

This runner deliberately keeps the data contract small and explicit:

* zq targets are the verified ``[768,T]`` float32 arrays from Step 1;
* no_text semantic memory is the verified adapter ``[T,512]`` array;
* text semantic memory is produced online from ``content_token_ids`` by the
  existing ``SourceTokenMemoryEncoder``; no target/source BNF is loaded;
* the reference speaker sidecar is the existing 192-D ECAPA embedding.

It supports a single process for local smoke tests and torchrun/DDP for the
8-GPU QZ probe.  Evaluation/inference is intentionally separate so a failed
training process cannot leave partially published WAVs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]

from moss_codecvc.models.ddlfm_decoder import DDLFMDecoder
from moss_codecvc.models.source_semantic_memory import SourceTokenMemoryEncoder
from moss_codecvc.audio.zq_normalization import load_zq_channel_stats, sha256_file


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(ROOT / "prepared/ddlfm_v1_index.jsonl"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--warmup-steps", type=int, default=3000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--precision", choices=("float32", "bf16", "fp16"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--latent-dim", type=int, default=768)
    ap.add_argument("--semantic-dim", type=int, default=512)
    ap.add_argument("--speaker-dim", type=int, default=192)
    ap.add_argument("--hidden-size", type=int, default=768)
    ap.add_argument("--num-layers", type=int, default=12)
    ap.add_argument("--num-heads", type=int, default=12)
    ap.add_argument("--ffn-size", type=int, default=3072)
    ap.add_argument("--num-speaker-prompt-tokens", type=int, default=4)
    ap.add_argument("--speaker-condition-scale", type=float, default=4.0)
    ap.add_argument("--speaker-input-scale", type=float, default=1.0)
    ap.add_argument("--text-vocab-size", type=int, default=8001)
    ap.add_argument("--text-padding-id", type=int, default=0)
    ap.add_argument(
        "--mode",
        choices=("no_text", "all"),
        default="no_text",
        help="Batch-46 M1 is deliberately no_text-only; 'all' is retained for diagnostics.",
    )
    ap.add_argument(
        "--t-sampling",
        choices=("shift_low", "mode_shift_low", "logit_normal", "cosine", "uniform"),
        default="shift_low",
    )
    ap.add_argument("--t-logit-mu", type=float, default=0.0)
    ap.add_argument("--t-logit-sigma", type=float, default=1.0)
    ap.add_argument(
        "--t-shift-power",
        type=float,
        default=4.0,
        help="Power for shift_low: t=u**power. 4.0 meets the t<0.3 >70%% contract.",
    )
    ap.add_argument(
        "--t-mode-shift-m",
        type=float,
        default=3.0,
        help="Low-direction mode shift m: t=u/(m-(m-1)u).",
    )
    ap.add_argument(
        "--loss-weighting",
        choices=("low_t", "high_t", "none"),
        default="low_t",
        help=(
            "low_t uses 1/(t+eps) to emphasize the noise endpoint.  high_t keeps the "
            "literal 1/(1-t+eps) alternative for controlled comparison."
        ),
    )
    ap.add_argument("--loss-weight-eps", type=float, default=0.02)
    ap.add_argument("--loss-weight-cap", type=float, default=25.0)
    ap.add_argument("--speaker-dropout", type=float, default=0.25)
    ap.add_argument("--semantic-dropout", type=float, default=0.15)
    ap.add_argument("--speaker-cfg-scale", type=float, default=2.5)
    ap.add_argument("--semantic-cfg-scale", type=float, default=2.0)
    ap.add_argument("--aux-loss-weight", type=float, default=1.0)
    ap.add_argument("--aux-warmup-steps", type=int, default=500)
    ap.add_argument("--cross-gate-init", type=float, default=0.05)
    ap.add_argument("--gate-warmup-steps", type=int, default=500)
    ap.add_argument("--gate-warmup-start", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument(
        "--ema-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ramp the effective EMA decay toward --ema-decay using the standard "
            "min(decay, (1+n)/(10+n)) schedule. This prevents a 3k probe from "
            "evaluating an EMA that is still dominated by initialization."
        ),
    )
    ap.add_argument(
        "--zq-channel-stats",
        default=str(ROOT / "prepared/zq_targets_v1/channel_stats.pt"),
    )
    ap.add_argument("--max-rows", type=int, default=0, help="Smoke-test cap; 0 means all rows")
    ap.add_argument("--smoke-small-model", action="store_true")
    return ap.parse_args()


def init_distributed() -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_dist = world_size > 1
    if use_dist:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA in the ver3.1 runner")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        if str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip() and torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif str(os.environ.get("DEVICE", "")).startswith("cuda") and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    return use_dist, rank, world_size, device


def seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def load_embedding(path: str, expected_dim: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value: Any = payload
    if isinstance(payload, dict):
        for key in ("embedding", "speaker_embedding", "emb", "vector"):
            if torch.is_tensor(payload.get(key)):
                value = payload[key]
                break
    value = torch.as_tensor(value, dtype=torch.float32).flatten()
    if value.numel() != int(expected_dim):
        raise ValueError(f"speaker embedding {path} has {value.numel()} dims, expected {expected_dim}")
    return value


@dataclass
class CFMItem:
    mode: int
    zq_path: str
    semantic_path: str | None
    token_ids: list[int] | None
    speaker_path: str


class DDLFMDataset(Dataset[CFMItem]):
    def __init__(
        self,
        index_path: str | Path,
        *,
        max_rows: int = 0,
        max_frames: int = 0,
        mode: str = "no_text",
    ) -> None:
        self.max_frames = max(0, int(max_frames))
        self.mode = str(mode)
        if self.mode not in {"no_text", "all"}:
            raise ValueError(f"unsupported DDLFM dataset mode: {self.mode}")
        self.items: list[CFMItem] = []
        with Path(index_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                mode_name = str(row.get("moss_codecvc_mode") or "")
                if mode_name not in {"no_text", "text"}:
                    continue
                if self.mode == "no_text" and mode_name != "no_text":
                    continue
                self.items.append(
                    CFMItem(
                        mode=0 if mode_name == "no_text" else 1,
                        zq_path=str(row["zq_path"]),
                        semantic_path=str(row["semantic_path"]) if mode_name == "no_text" else None,
                        token_ids=list(row["content_token_ids"]) if mode_name == "text" else None,
                        speaker_path=str(row["speaker_embedding_path"]),
                    )
                )
                if max_rows and len(self.items) >= int(max_rows):
                    break
        if not self.items:
            raise ValueError(f"empty DDLFM index: {index_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        zq = torch.from_numpy(np.load(item.zq_path).astype("float32", copy=False)).transpose(0, 1).contiguous()
        if zq.ndim != 2 or zq.shape[1] != 768:
            raise ValueError(f"zq must be [T,768], got {tuple(zq.shape)}: {item.zq_path}")
        speaker = load_embedding(item.speaker_path, 192)
        if item.mode == 0:
            semantic = torch.from_numpy(np.load(item.semantic_path).astype("float32", copy=False))
            if semantic.ndim != 2 or semantic.shape[1] != 512:
                raise ValueError(f"semantic must be [T,512], got {tuple(semantic.shape)}: {item.semantic_path}")
            if semantic.shape[0] != zq.shape[0]:
                length = min(int(semantic.shape[0]), int(zq.shape[0]))
                semantic, zq = semantic[:length], zq[:length]
            if self.max_frames and zq.shape[0] > self.max_frames:
                start = random.randint(0, int(zq.shape[0]) - self.max_frames)
                zq = zq[start : start + self.max_frames]
                semantic = semantic[start : start + self.max_frames]
            return {"mode": item.mode, "zq": zq, "semantic": semantic, "token_ids": None, "speaker": speaker}
        if self.max_frames and zq.shape[0] > self.max_frames:
            start = random.randint(0, int(zq.shape[0]) - self.max_frames)
            zq = zq[start : start + self.max_frames]
        tokens = torch.as_tensor(item.token_ids, dtype=torch.long)
        if tokens.ndim != 1 or tokens.numel() <= 0:
            raise ValueError(f"text row has invalid token ids: {item.zq_path}")
        return {"mode": item.mode, "zq": zq, "semantic": None, "token_ids": tokens, "speaker": speaker}


def collate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty batch")
    max_t = max(int(row["zq"].shape[0]) for row in rows)
    latent_dim = int(rows[0]["zq"].shape[1])
    zq = torch.zeros(len(rows), max_t, latent_dim, dtype=torch.float32)
    target_mask = torch.zeros(len(rows), max_t, dtype=torch.bool)
    speakers = torch.stack([row["speaker"] for row in rows])
    modes = torch.tensor([int(row["mode"]) for row in rows], dtype=torch.long)
    semantics: list[torch.Tensor] = []
    token_rows: list[torch.Tensor] = []
    for idx, row in enumerate(rows):
        cur = row["zq"]
        zq[idx, : cur.shape[0]] = cur
        target_mask[idx, : cur.shape[0]] = True
        if row["mode"] == 0:
            semantics.append(row["semantic"])
            token_rows.append(torch.empty(0, dtype=torch.long))
        else:
            semantics.append(torch.empty(0, 512))
            token_rows.append(row["token_ids"])
    max_s = max(int(x.shape[0]) for x in semantics)
    max_l = max(int(x.shape[0]) for x in token_rows)
    semantic = torch.zeros(len(rows), max_s, 512, dtype=torch.float32)
    semantic_mask = torch.zeros(len(rows), max_s, dtype=torch.bool)
    token_ids = torch.zeros(len(rows), max_l if max_l > 0 else 1, dtype=torch.long)
    token_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for idx, (sem, tok) in enumerate(zip(semantics, token_rows)):
        if sem.numel():
            semantic[idx, : sem.shape[0]] = sem
            semantic_mask[idx, : sem.shape[0]] = True
        if tok.numel():
            token_ids[idx, : tok.shape[0]] = tok
            token_mask[idx, : tok.shape[0]] = True
    return {
        "zq": zq,
        "target_mask": target_mask,
        "speaker": speakers,
        "mode": modes,
        "semantic": semantic,
        "semantic_mask": semantic_mask,
        "token_ids": token_ids,
        "token_mask": token_mask,
    }


class DDLFMTrainModule(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        hidden = int(args.hidden_size)
        layers = int(args.num_layers)
        heads = int(args.num_heads)
        ffn = int(args.ffn_size)
        if args.smoke_small_model:
            hidden, layers, heads, ffn = 64, 2, 4, 256
        self.decoder = DDLFMDecoder(
            latent_dim=int(args.latent_dim),
            semantic_dim=int(args.semantic_dim),
            speaker_dim=int(args.speaker_dim),
            hidden_size=hidden,
            num_layers=layers,
            num_heads=heads,
            ffn_size=ffn,
            cross_gate_init=float(getattr(args, "cross_gate_init", 0.0)),
            num_speaker_prompt_tokens=int(getattr(args, "num_speaker_prompt_tokens", 4)),
            speaker_condition_scale=float(getattr(args, "speaker_condition_scale", 4.0)),
            speaker_input_scale=float(getattr(args, "speaker_input_scale", 1.0)),
        )
        self.text_encoder = SourceTokenMemoryEncoder(
            vocab_size=int(args.text_vocab_size),
            hidden_size=int(args.semantic_dim),
            padding_id=int(args.text_padding_id),
            dropout=0.1,
            position_scale=0.0,
            dedup_units=False,
        )

    def forward(self, *args: Any, **kwargs: Any):
        return self.decoder(*args, **kwargs)

    def build_semantic(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        # The batch collator pads the two modalities separately.  Build one
        # common memory tensor so the DiT can run a mixed no_text/text batch.
        bsz = int(batch["zq"].shape[0])
        memories: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for idx in range(bsz):
            if int(batch["mode"][idx].item()) == 0:
                length = int(batch["semantic_mask"][idx].sum().item())
                memories.append(batch["semantic"][idx, :length])
                masks.append(torch.ones(length, dtype=torch.bool, device=batch["zq"].device))
            else:
                length = int(batch["token_mask"][idx].sum().item())
                ids = batch["token_ids"][idx : idx + 1, :length]
                token_mask = batch["token_mask"][idx : idx + 1, :length]
                state = self.text_encoder(ids, token_mask)
                memories.append(state.memory[0])
                masks.append(state.mask[0])
        max_len = max(int(x.shape[0]) for x in memories)
        semantic = batch["zq"].new_zeros((bsz, max_len, int(self.decoder.semantic_dim)))
        mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=batch["zq"].device)
        for idx, (memory, valid) in enumerate(zip(memories, masks)):
            semantic[idx, : memory.shape[0]] = memory.to(device=semantic.device, dtype=semantic.dtype)
            mask[idx, : memory.shape[0]] = valid.to(device=mask.device)
        return semantic, mask


def sample_cfm_time(
    batch_size: int,
    *,
    device: torch.device,
    schedule: str,
    logit_mu: float = 0.0,
    logit_sigma: float = 1.0,
    shift_power: float = 4.0,
    mode_shift_m: float = 3.0,
) -> torch.Tensor:
    """Sample CFM interpolation time with an auditable schedule."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    schedule = str(schedule)
    if schedule == "uniform":
        return torch.rand(int(batch_size), device=device)
    if schedule == "shift_low":
        if float(shift_power) <= 0.0 or not math.isfinite(float(shift_power)):
            raise ValueError("shift_power must be finite and positive")
        # Power-law low shift.  With power=4, P(t<0.3)=0.740; power=3 is
        # retained as an explicit ablation but only gives about 0.669.
        return torch.rand(int(batch_size), device=device).pow(float(shift_power))
    if schedule == "mode_shift_low":
        if float(mode_shift_m) <= 1.0 or not math.isfinite(float(mode_shift_m)):
            raise ValueError("mode_shift_m must be finite and greater than 1")
        u = torch.rand(int(batch_size), device=device)
        # Low-direction inverse of the commonly quoted high-shift formula;
        # median(u)=0.5 maps to 0.5/(m-(m-1)*0.5)=1/(m+1).
        return u / (float(mode_shift_m) - (float(mode_shift_m) - 1.0) * u)
    if schedule == "cosine":
        u = torch.rand(int(batch_size), device=device)
        return 1.0 - torch.cos(0.5 * math.pi * u)
    if schedule == "logit_normal":
        if float(logit_sigma) <= 0.0:
            raise ValueError("logit_sigma must be positive")
        normal = torch.randn(int(batch_size), device=device)
        return torch.sigmoid(float(logit_mu) + float(logit_sigma) * normal)
    raise ValueError(f"unsupported t schedule: {schedule}")


def cfm_loss_weights(
    t: torch.Tensor,
    *,
    mode: str,
    eps: float,
    cap: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Return bounded per-example velocity-loss weights.

    With ``normalize=True`` the result is mean-one within the supplied batch;
    Batch-47 training uses ``normalize=False`` plus a fixed schedule-wide
    reference so a batch of one cannot accidentally cancel the weighting.
    The Batch-46 intent is to strengthen the real inference endpoint ``t≈0``.
    That corresponds to ``1/(t+eps)``.  The literal ``1/(1-t+eps)`` proposal
    instead emphasizes the target endpoint, so it remains available only as
    the explicitly named ``high_t`` control.
    """

    if t.ndim != 1:
        raise ValueError(f"t must be [B], got {tuple(t.shape)}")
    if float(eps) <= 0.0:
        raise ValueError("loss-weight eps must be positive")
    if float(cap) < 1.0:
        raise ValueError("loss-weight cap must be >= 1")
    mode = str(mode)
    if mode == "none":
        raw = torch.ones_like(t)
    elif mode == "low_t":
        raw = 1.0 / (t + float(eps))
    elif mode == "high_t":
        raw = 1.0 / (1.0 - t + float(eps))
    else:
        raise ValueError(f"unsupported loss weighting: {mode}")
    raw = raw.clamp(max=float(cap))
    if not normalize:
        return raw
    return raw / raw.detach().mean().clamp_min(1.0e-8)


def estimate_cfm_weight_reference(
    *,
    schedule: str,
    eps: float,
    cap: float,
    shift_power: float = 4.0,
    mode_shift_m: float = 3.0,
    samples: int = 200_000,
) -> float:
    """Estimate a fixed schedule-wide weight mean for DDP-stable scaling."""

    if str(schedule) == "none" or str(schedule) == "uniform":
        return 1.0
    generator = torch.Generator(device="cpu").manual_seed(20260716)
    if str(schedule) == "shift_low":
        t = torch.rand(int(samples), generator=generator).pow(float(shift_power))
    elif str(schedule) == "mode_shift_low":
        u = torch.rand(int(samples), generator=generator)
        t = u / (float(mode_shift_m) - (float(mode_shift_m) - 1.0) * u)
    elif str(schedule) == "logit_normal":
        t = torch.sigmoid(float(torch.randn(int(samples), generator=generator).mul(1.0)))
    elif str(schedule) == "cosine":
        u = torch.rand(int(samples), generator=generator)
        t = 1.0 - torch.cos(0.5 * math.pi * u)
    else:
        raise ValueError(f"unsupported schedule for weight reference: {schedule}")
    return float(cfm_loss_weights(t, mode="low_t", eps=eps, cap=cap, normalize=False).mean().item())


def apply_speaker_dropout(
    speaker: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero whole speaker embeddings for classifier-free guidance training."""

    if speaker.ndim != 2:
        raise ValueError(f"speaker must be [B,D], got {tuple(speaker.shape)}")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("speaker dropout probability must be in [0, 1]")
    mask = torch.rand(int(speaker.shape[0]), device=speaker.device) < float(probability)
    if not bool(mask.any().item()):
        return speaker, mask
    dropped = speaker.clone()
    dropped[mask] = 0.0
    return dropped, mask


def apply_semantic_dropout(
    semantic: torch.Tensor,
    semantic_mask: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop complete semantic memories for semantic classifier-free guidance."""

    if semantic.ndim != 3:
        raise ValueError(f"semantic must be [B,S,D], got {tuple(semantic.shape)}")
    if semantic_mask.ndim != 2 or tuple(semantic_mask.shape[:2]) != tuple(semantic.shape[:2]):
        raise ValueError("semantic_mask must be [B,S] matching semantic")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("semantic dropout probability must be in [0, 1]")
    mask = torch.rand(int(semantic.shape[0]), device=semantic.device) < float(probability)
    dropped = semantic.clone()
    dropped_mask = semantic_mask.clone()
    if bool(mask.any().item()):
        dropped[mask] = 0.0
        dropped_mask[mask] = False
    return dropped, dropped_mask, mask


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    normalize_sample_weight: bool = True,
) -> torch.Tensor:
    valid = mask.to(dtype=prediction.dtype)
    squared = (prediction - target).square().mean(dim=-1)
    per_example_denom = valid.sum(dim=-1).clamp_min(1.0)
    per_example = (squared * valid).sum(dim=-1) / per_example_denom
    if sample_weight is None:
        return per_example.mean()
    if sample_weight.ndim != 1 or int(sample_weight.shape[0]) != int(prediction.shape[0]):
        raise ValueError("sample_weight must be [B]")
    weights = sample_weight.to(device=prediction.device, dtype=prediction.dtype)
    if normalize_sample_weight:
        weights = weights / weights.detach().mean().clamp_min(1.0e-8)
    return (per_example * weights).mean()


class ExponentialMovingAverage:
    """Device-local EMA of a module state, serialized with each checkpoint."""

    def __init__(self, module: nn.Module, decay: float, *, warmup: bool = True) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.last_decay = 0.0
        self.shadow = {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
        }

    def effective_decay(self, num_updates: int | None = None) -> float:
        updates = self.num_updates if num_updates is None else int(num_updates)
        if updates <= 0:
            return 0.0
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + float(updates)) / (10.0 + float(updates)))

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        state = module.state_dict()
        if state.keys() != self.shadow.keys():
            raise RuntimeError("EMA/module state keys changed during training")
        self.num_updates += 1
        decay = self.effective_decay()
        self.last_decay = float(decay)
        for name, value in state.items():
            target = self.shadow[name]
            source = value.detach().to(device=target.device)
            if torch.is_floating_point(target):
                target.mul_(decay).add_(source.to(dtype=target.dtype), alpha=1.0 - decay)
            else:
                target.copy_(source)

    def state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.shadow.items()}


def _gradient_l2(parameters: list[torch.Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total += float(grad.square().sum().item())
    return math.sqrt(total)


def speaker_gradient_diagnostics(module: DDLFMTrainModule) -> dict[str, float]:
    """Return rank-local, pre-clip speaker/semantic gradient norms."""

    speaker_proj_parameters = [
        parameter
        for submodule in module.decoder.speaker_proj
        for parameter in submodule.parameters(recurse=False)
    ]
    return {
        "speaker_prompt_grad_l2_rank_local_preclip": _gradient_l2(
            [module.decoder.speaker_prompt]
        ),
        "speaker_proj_grad_l2_rank_local_preclip": _gradient_l2(
            speaker_proj_parameters
        ),
        "semantic_proj_grad_l2_rank_local_preclip": _gradient_l2(
            list(module.decoder.semantic_proj.parameters())
        ),
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Publish a checkpoint only after ``torch.save`` fully succeeds."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_link(source: Path, destination: Path) -> None:
    """Atomically repoint ``destination`` at a completed same-filesystem file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            # Some network/object-backed filesystems disable hard links.  A
            # same-directory temporary copy preserves atomic publication.
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(
    module: DDLFMTrainModule,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    step: int,
    args: argparse.Namespace,
    output_dir: Path,
    rank: int,
) -> None:
    if rank != 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    state = module.state_dict()
    path = output_dir / f"step-{int(step):06d}.pt"
    ema_state = ema.state_dict_cpu()
    ema_metadata = {
        "target_decay": ema.decay,
        "warmup": ema.warmup,
        "num_updates": ema.num_updates,
        "effective_decay": ema.last_decay,
    }
    atomic_torch_save(
        {
            "step": int(step),
            "model": state,
            "ema_model": ema_state,
            "ema": ema_metadata,
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
        },
        path,
    )
    infer_path = output_dir / f"step-{int(step):06d}.infer.pt"
    atomic_torch_save(
        {
            "step": int(step),
            "model": state,
            "ema_model": ema_state,
            "ema": ema_metadata,
            "config": vars(args),
        },
        infer_path,
    )
    last_path = output_dir / "last.pt"
    atomic_link(infer_path, last_path)
    atomic_json_write(
        {
            "schema": "ver3_1_ddlfm_checkpoint_ready_v1",
            "status": "ready",
            "step": int(step),
            "checkpoint": str(path),
            "checkpoint_size_bytes": int(path.stat().st_size),
            "inference_checkpoint": str(infer_path),
            "inference_checkpoint_size_bytes": int(infer_path.stat().st_size),
            "last_checkpoint": str(last_path),
            "last_checkpoint_size_bytes": int(last_path.stat().st_size),
            "ema": ema_metadata,
            "created_at_unix": time.time(),
        },
        output_dir / f"step-{int(step):06d}.ready.json",
    )


def infinite_loader(loader: DataLoader):
    """Yield batches forever without ``itertools.cycle``'s unbounded cache.

    ``itertools.cycle(loader)`` retains every batch from the first pass so it
    can replay them.  With the full v1 index that silently grows host memory
    for the duration of the probe.  Re-entering the DataLoader iterator at
    epoch boundaries preserves the intended shuffle behavior without retaining
    any batch tensors.
    """
    while True:
        for batch in loader:
            yield batch


def main() -> int:
    args = parse_args()
    if not 0.0 <= float(args.speaker_dropout) < 1.0:
        raise ValueError("speaker-dropout must be in [0, 1)")
    if not 0.0 <= float(args.semantic_dropout) < 1.0:
        raise ValueError("semantic-dropout must be in [0, 1)")
    if float(args.aux_loss_weight) < 0.0:
        raise ValueError("aux-loss-weight must be non-negative")
    if int(args.aux_warmup_steps) <= 0:
        raise ValueError("aux-warmup-steps must be positive")
    if int(args.num_speaker_prompt_tokens) <= 0:
        raise ValueError("num-speaker-prompt-tokens must be positive")
    if not math.isfinite(float(args.speaker_condition_scale)) or float(args.speaker_condition_scale) < 0.0:
        raise ValueError("speaker-condition-scale must be finite and non-negative")
    if not math.isfinite(float(args.speaker_input_scale)) or float(args.speaker_input_scale) < 0.0:
        raise ValueError("speaker-input-scale must be finite and non-negative")
    if not math.isfinite(float(args.speaker_cfg_scale)) or float(args.speaker_cfg_scale) < 0.0:
        raise ValueError("speaker-cfg-scale must be finite and non-negative")
    if not math.isfinite(float(args.semantic_cfg_scale)) or float(args.semantic_cfg_scale) < 0.0:
        raise ValueError("semantic-cfg-scale must be finite and non-negative")
    loss_weight_reference = estimate_cfm_weight_reference(
        schedule=str(args.t_sampling),
        eps=float(args.loss_weight_eps),
        cap=float(args.loss_weight_cap),
        shift_power=float(args.t_shift_power),
        mode_shift_m=float(args.t_mode_shift_m),
    )
    args.loss_weight_reference = float(loss_weight_reference)
    if not 0.0 <= float(args.gate_warmup_start) <= 1.0:
        raise ValueError("gate-warmup-start must be in [0, 1]")
    use_dist, rank, world_size, device = init_distributed()
    seed_everything(int(args.seed), rank)
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset = DDLFMDataset(
        args.index,
        max_rows=int(args.max_rows),
        max_frames=int(args.max_frames),
        mode=str(args.mode),
    )
    sampler = DistributedSampler(dataset, shuffle=True, seed=int(args.seed)) if use_dist else None
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_rows,
        drop_last=True,
    )
    module = DDLFMTrainModule(args).to(device)
    if use_dist:
        # A local rank can receive a homogeneous no_text or text batch even
        # though the global sampler mixes both modes.  The text-only memory
        # encoder is therefore legitimately unused on some ranks/steps.
        module = DistributedDataParallel(
            module,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    raw_module = module.module if isinstance(module, DistributedDataParallel) else module
    stats_path = Path(args.zq_channel_stats).expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Batch-46 requires complete zq channel stats before training: {stats_path}"
        )
    zq_stats = load_zq_channel_stats(stats_path)
    if str(zq_stats.get("status")) != "completed" or bool(zq_stats.get("partial", False)):
        raise ValueError(f"refusing incomplete zq channel stats: {stats_path}")
    if int(zq_stats["latent_dim"]) != int(args.latent_dim):
        raise ValueError("zq channel stats latent_dim does not match the model")
    args.zq_channel_stats = str(stats_path)
    args.zq_channel_stats_sha256 = sha256_file(stats_path)
    args.zq_normalization_enabled = True
    zq_mean = zq_stats["mean"].to(device=device, dtype=torch.float32).view(1, 1, -1)
    zq_std = zq_stats["std"].to(device=device, dtype=torch.float32).view(1, 1, -1)
    optimizer = torch.optim.AdamW(module.parameters(), lr=float(args.lr), betas=(0.9, 0.95), weight_decay=0.01)
    ema = ExponentialMovingAverage(
        raw_module,
        float(args.ema_decay),
        warmup=bool(args.ema_warmup),
    )
    use_amp = device.type == "cuda" and args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.precision == "fp16")
    log_path = output_dir / "train_log.jsonl"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    iterator = infinite_loader(loader)
    started = time.time()
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    for step in range(1, int(args.steps) + 1):
        if sampler is not None and (step - 1) % max(1, len(loader)) == 0:
            sampler.set_epoch((step - 1) // max(1, len(loader)))
        warmup_scale = min(1.0, float(step) / max(1, int(args.warmup_steps)))
        gate_progress = min(1.0, float(step) / max(1, int(args.gate_warmup_steps)))
        gate_scale = float(args.gate_warmup_start) + (1.0 - float(args.gate_warmup_start)) * gate_progress
        current_lr = float(args.lr) * warmup_scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        cfm_loss_total = 0.0
        aux_loss_total = 0.0
        speaker_drop_total = 0
        semantic_drop_total = 0
        t_sum = 0.0
        t_low_total = 0
        t_shift_low_total = 0
        loss_weight_min = math.inf
        loss_weight_max = 0.0
        last_t: torch.Tensor | None = None
        last_speaker_input: torch.Tensor | None = None
        aux_progress = min(1.0, float(step) / max(1, int(args.aux_warmup_steps)))
        aux_weight = float(args.aux_loss_weight) * aux_progress
        for _micro_step in range(grad_accum_steps):
            batch = next(iterator)
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            target = (batch["zq"] - zq_mean) / zq_std
            t = sample_cfm_time(
                int(target.shape[0]),
                device=device,
                schedule=str(args.t_sampling),
                logit_mu=float(args.t_logit_mu),
                logit_sigma=float(args.t_logit_sigma),
                shift_power=float(args.t_shift_power),
                mode_shift_m=float(args.t_mode_shift_m),
            )
            noise = torch.randn_like(target)
            x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target
            v_target = target - noise
            semantic, semantic_mask = raw_module.build_semantic(batch)
            full_semantic = semantic
            full_semantic_mask = semantic_mask
            full_speaker = batch["speaker"]
            semantic_input, semantic_mask_input, semantic_drop_mask = apply_semantic_dropout(
                semantic,
                semantic_mask,
                float(args.semantic_dropout),
            )
            speaker_input, speaker_drop_mask = apply_speaker_dropout(
                batch["speaker"],
                float(args.speaker_dropout),
            )
            sample_weight = cfm_loss_weights(
                t,
                mode=str(args.loss_weighting),
                eps=float(args.loss_weight_eps),
                cap=float(args.loss_weight_cap),
                normalize=False,
            )
            sample_weight = sample_weight / float(loss_weight_reference)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                prediction = module(
                    x_t,
                    t,
                    semantic_input,
                    speaker_input,
                    target_mask=batch["target_mask"],
                    semantic_mask=semantic_mask_input,
                    semantic_modality=batch["mode"],
                    condition_gate_scale=gate_scale,
                ).velocity
                cfm_loss = masked_mse(
                    prediction,
                    v_target,
                    batch["target_mask"],
                    sample_weight=sample_weight,
                )
                # Endpoint auxiliary objective: the model must predict the
                # velocity from pure noise rather than relying on target
                # information already present in x_t at larger t.
                if aux_weight > 0.0:
                    aux_prediction = module(
                        noise,
                        torch.zeros_like(t),
                        full_semantic,
                        full_speaker,
                        target_mask=batch["target_mask"],
                        semantic_mask=full_semantic_mask,
                        semantic_modality=batch["mode"],
                        condition_gate_scale=gate_scale,
                    ).velocity
                    aux_loss = masked_mse(
                        aux_prediction,
                        v_target,
                        batch["target_mask"],
                    )
                else:
                    aux_loss = cfm_loss.new_zeros(())
                loss = cfm_loss + float(aux_weight) * aux_loss
                scaled_loss = loss / float(grad_accum_steps)
            if not torch.isfinite(loss) or not torch.isfinite(cfm_loss) or not torch.isfinite(aux_loss):
                raise FloatingPointError(f"non-finite CFM loss at step {step}: {loss}")
            loss_total += float(loss.detach().cpu().item())
            cfm_loss_total += float(cfm_loss.detach().cpu().item())
            aux_loss_total += float(aux_loss.detach().cpu().item())
            speaker_drop_total += int(speaker_drop_mask.sum().item())
            semantic_drop_total += int(semantic_drop_mask.sum().item())
            t_sum += float(t.detach().sum().item())
            t_low_total += int((t < 0.2).sum().item())
            t_shift_low_total += int((t < 0.3).sum().item())
            loss_weight_min = min(loss_weight_min, float(sample_weight.detach().min().item()))
            loss_weight_max = max(loss_weight_max, float(sample_weight.detach().max().item()))
            last_t = t.detach()
            last_speaker_input = speaker_input.detach()
            scaler.scale(scaled_loss).backward()
        scaler.unscale_(optimizer)
        should_log = step == 1 or step % max(1, int(args.log_every)) == 0
        should_diagnose = step == 1 or step % 500 == 0
        grad_diagnostics = (
            speaker_gradient_diagnostics(raw_module)
            if rank == 0 and (should_log or should_diagnose)
            else {}
        )
        total_grad_norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()
        ema.update(raw_module)
        local_examples = int(args.batch_size) * grad_accum_steps
        if should_log:
            reduced = torch.tensor(
                [
                    loss_total,
                    cfm_loss_total,
                    aux_loss_total,
                    t_sum,
                    float(t_low_total),
                    float(t_shift_low_total),
                    float(speaker_drop_total),
                    float(semantic_drop_total),
                    float(local_examples),
                ],
                dtype=torch.float64,
                device=device,
            )
            reduced_min = torch.tensor(loss_weight_min, dtype=torch.float64, device=device)
            reduced_max = torch.tensor(loss_weight_max, dtype=torch.float64, device=device)
            if use_dist:
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                dist.all_reduce(reduced_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(reduced_max, op=dist.ReduceOp.MAX)
            global_examples = max(1.0, float(reduced[8].item()))
            global_loss = float(reduced[0].item()) / float(max(1, world_size) * grad_accum_steps)
        if rank == 0 and should_log:
            payload = {
                "step": step,
                "loss": global_loss,
                "cfm_loss": float(reduced[1].item()) / float(max(1, world_size) * grad_accum_steps),
                "aux_loss": float(reduced[2].item()) / float(max(1, world_size) * grad_accum_steps),
                "aux_loss_weight": aux_weight,
                "loss_reduction": "mean_over_ddp_ranks_and_microbatches",
                "lr": current_lr,
                "gate_scale": gate_scale,
                "t_mean": float(reduced[3].item()) / global_examples,
                "t_lt_0_2_ratio": float(reduced[4].item()) / global_examples,
                "t_lt_0_3_ratio": float(reduced[5].item()) / global_examples,
                "loss_weight_min": float(reduced_min.item()),
                "loss_weight_max": float(reduced_max.item()),
                "loss_weight_reference": float(loss_weight_reference),
                "speaker_dropout_ratio": float(reduced[6].item()) / global_examples,
                "semantic_dropout_ratio": float(reduced[7].item()) / global_examples,
                "ema_target_decay": float(ema.decay),
                "ema_effective_decay": float(ema.last_decay),
                "ema_num_updates": int(ema.num_updates),
                "total_grad_norm_rank0_preclip": total_grad_norm,
                "elapsed_sec": time.time() - started,
                "world_size": world_size,
                "rows": len(dataset),
                "mode": str(args.mode),
            }
            payload.update(grad_diagnostics)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            print(json.dumps(payload), flush=True)
        if rank == 0 and should_diagnose:
            if last_t is None or last_speaker_input is None:
                raise RuntimeError("conditioning diagnostics have no sampled batch")
            diagnostics = raw_module.decoder.conditioning_diagnostics(
                last_t,
                last_speaker_input,
                gate_scale=gate_scale,
            )
            diagnostics.update(
                {
                    "step": int(step),
                    "gate_warmup_steps": int(args.gate_warmup_steps),
                    "cross_gate_init": float(args.cross_gate_init),
                    "gradient_norms": grad_diagnostics,
                    "total_grad_norm_rank0_preclip": total_grad_norm,
                }
            )
            diagnostic_path = output_dir / f"adaln_diagnostics_step-{int(step):06d}.json"
            diagnostic_path.write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if step % max(1, int(args.save_every)) == 0 or step == int(args.steps):
            save_checkpoint(raw_module, optimizer, ema, step, args, output_dir, rank)
    if use_dist:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
