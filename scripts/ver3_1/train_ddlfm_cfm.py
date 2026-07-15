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
import itertools
import json
import math
import os
import random
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(ROOT / "prepared/ddlfm_v1_index.jsonl"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
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
    ap.add_argument("--text-vocab-size", type=int, default=8001)
    ap.add_argument("--text-padding-id", type=int, default=0)
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
    def __init__(self, index_path: str | Path, *, max_rows: int = 0, max_frames: int = 0) -> None:
        self.max_frames = max(0, int(max_frames))
        self.items: list[CFMItem] = []
        with Path(index_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                mode_name = str(row.get("moss_codecvc_mode") or "")
                if mode_name not in {"no_text", "text"}:
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


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(dtype=prediction.dtype)
    denom = weight.sum().clamp_min(1.0) * prediction.shape[-1]
    return ((prediction - target).square() * weight).sum() / denom


def save_checkpoint(
    module: DDLFMTrainModule,
    optimizer: torch.optim.Optimizer,
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
    torch.save(
        {
            "step": int(step),
            "model": state,
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
        },
        path,
    )
    torch.save({"step": int(step), "model": state, "config": vars(args)}, output_dir / "last.pt")


def main() -> int:
    args = parse_args()
    use_dist, rank, world_size, device = init_distributed()
    seed_everything(int(args.seed), rank)
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset = DDLFMDataset(args.index, max_rows=int(args.max_rows), max_frames=int(args.max_frames))
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
        module = DistributedDataParallel(module, device_ids=[device.index], broadcast_buffers=False)
    raw_module = module.module if isinstance(module, DistributedDataParallel) else module
    optimizer = torch.optim.AdamW(module.parameters(), lr=float(args.lr), betas=(0.9, 0.95), weight_decay=0.01)
    use_amp = device.type == "cuda" and args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.precision == "fp16")
    log_path = output_dir / "train_log.jsonl"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    iterator = itertools.cycle(loader)
    started = time.time()
    for step in range(1, int(args.steps) + 1):
        if sampler is not None and (step - 1) % max(1, len(loader)) == 0:
            sampler.set_epoch((step - 1) // max(1, len(loader)))
        batch = next(iterator)
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        target = batch["zq"]
        t = torch.rand(target.shape[0], device=device)
        noise = torch.randn_like(target)
        x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target
        v_target = target - noise
        semantic, semantic_mask = raw_module.build_semantic(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            prediction = module(
                x_t,
                t,
                semantic,
                batch["speaker"],
                target_mask=batch["target_mask"],
                semantic_mask=semantic_mask,
                semantic_modality=batch["mode"],
            ).velocity
            loss = masked_mse(prediction, v_target, batch["target_mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CFM loss at step {step}: {loss}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        warmup_scale = min(1.0, float(step) / max(1, int(args.warmup_steps)))
        for group in optimizer.param_groups:
            group["lr"] = float(args.lr) * warmup_scale
        if rank == 0 and (step == 1 or step % max(1, int(args.log_every)) == 0):
            payload = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_sec": time.time() - started,
                "world_size": world_size,
                "rows": len(dataset),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            print(json.dumps(payload), flush=True)
        if step % max(1, int(args.save_every)) == 0 or step == int(args.steps):
            save_checkpoint(raw_module, optimizer, step, args, output_dir, rank)
    if use_dist:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
