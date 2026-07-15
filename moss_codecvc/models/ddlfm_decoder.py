"""Dequantized decoder-latent flow-matching decoder for ver3.1.

The decoder predicts the frozen MOSS audio tokenizer's *decoder-domain*
latent ``zq`` directly.  Inputs and outputs use ``[B, T, D]`` in this module;
the frozen tokenizer wrapper converts the final prediction to native
``[B, D, T]`` before calling :func:`moss_codecvc.audio.decode_latents`.

Semantic memory is intentionally modality-agnostic:

* no_text rows provide the pre-extracted WavLM adapter sequence ``[B,T,512]``;
* text rows provide ``SourceTokenMemoryEncoder`` output ``[B,L,512]``.

The cross-attention key length therefore need not equal the target latent
length.  A learnable modality embedding keeps the two contracts distinct.
The speaker side uses the existing 192-D reference sidecar (currently ECAPA
in v1; the caller records that provenance rather than calling it CAM++).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim != 1:
        raise ValueError(f"t must be [B], got {tuple(t.shape)}")
    half = max(1, int(dim) // 2)
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    phase = t.float().unsqueeze(1) * freq.unsqueeze(0)
    out = torch.cat([phase.sin(), phase.cos()], dim=-1)
    if out.shape[-1] < int(dim):
        out = torch.nn.functional.pad(out, (0, int(dim) - out.shape[-1]))
    return out[:, : int(dim)]


def _modulate(normed: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DDLFMAdaLNBlock(nn.Module):
    """Self-attention + semantic cross-attention + FFN with AdaLN gates."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        condition_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        num_heads = int(num_heads)
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        self.hidden_size = hidden_size
        self.norm_self = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=float(dropout), batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=float(dropout), batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, int(ffn_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ffn_size), hidden_size),
        )
        # shift/scale/gate for self attention, cross attention and FFN.
        self.condition = nn.Sequential(
            nn.LayerNorm(int(condition_size)),
            nn.Linear(int(condition_size), hidden_size * 9),
        )
        # AdaLN-Zero-style start: the residual branches begin as identity and
        # learn their conditioning gates from a stable zero-velocity head.
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        *,
        target_mask: torch.Tensor,
        semantic: torch.Tensor,
        semantic_mask: torch.Tensor,
    ) -> torch.Tensor:
        params = self.condition(condition).chunk(9, dim=-1)
        shift_self, scale_self, gate_self = params[0:3]
        shift_cross, scale_cross, gate_cross = params[3:6]
        shift_ffn, scale_ffn, gate_ffn = params[6:9]

        h = _modulate(self.norm_self(x), shift_self, scale_self)
        h = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=~target_mask,
            need_weights=False,
        )[0]
        x = x + gate_self.unsqueeze(1) * h

        h = _modulate(self.norm_cross(x), shift_cross, scale_cross)
        h = self.cross_attn(
            h,
            semantic,
            semantic,
            key_padding_mask=~semantic_mask,
            need_weights=False,
        )[0]
        x = x + gate_cross.unsqueeze(1) * h

        h = _modulate(self.norm_ffn(x), shift_ffn, scale_ffn)
        x = x + gate_ffn.unsqueeze(1) * self.ffn(h)
        return x.masked_fill(~target_mask.unsqueeze(-1), 0.0)


@dataclass
class DDLFMOutput:
    velocity: torch.Tensor


class DDLFMDecoder(nn.Module):
    """DiT-like conditional flow-matching velocity predictor."""

    def __init__(
        self,
        *,
        latent_dim: int = 768,
        semantic_dim: int = 512,
        speaker_dim: int = 192,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_size: int = 3072,
        dropout: float = 0.0,
        num_modalities: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.semantic_dim = int(semantic_dim)
        self.speaker_dim = int(speaker_dim)
        self.hidden_size = int(hidden_size)
        self.num_modalities = int(num_modalities)
        if self.num_modalities < 1:
            raise ValueError("num_modalities must be positive")

        self.input_proj = nn.Linear(self.latent_dim, self.hidden_size)
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.latent_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.semantic_proj = nn.Linear(self.semantic_dim, self.hidden_size)
        self.speaker_proj = nn.Sequential(
            nn.LayerNorm(self.speaker_dim),
            nn.Linear(self.speaker_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.modality_embedding = nn.Embedding(self.num_modalities, self.semantic_dim)
        self.speaker_prompt = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.layers = nn.ModuleList(
            [
                DDLFMAdaLNBlock(
                    self.hidden_size,
                    int(num_heads),
                    int(ffn_size),
                    self.hidden_size,
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            int(p.numel())
            for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        semantic: torch.Tensor,
        speaker: torch.Tensor,
        *,
        target_mask: torch.Tensor | None = None,
        semantic_mask: torch.Tensor | None = None,
        semantic_modality: torch.Tensor | None = None,
    ) -> DDLFMOutput:
        if x_t.ndim != 3 or int(x_t.shape[-1]) != self.latent_dim:
            raise ValueError(f"x_t must be [B,T,{self.latent_dim}], got {tuple(x_t.shape)}")
        batch, target_len, _ = x_t.shape
        if semantic.ndim != 3 or int(semantic.shape[0]) != batch or int(semantic.shape[-1]) != self.semantic_dim:
            raise ValueError(
                f"semantic must be [B,S,{self.semantic_dim}], got {tuple(semantic.shape)}"
            )
        if speaker.ndim != 2 or tuple(speaker.shape) != (batch, self.speaker_dim):
            raise ValueError(f"speaker must be [B,{self.speaker_dim}], got {tuple(speaker.shape)}")
        if t.ndim != 1 or int(t.shape[0]) != batch:
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        if target_mask is None:
            target_mask = torch.ones(batch, target_len, dtype=torch.bool, device=x_t.device)
        else:
            target_mask = target_mask.to(device=x_t.device).bool()
            if tuple(target_mask.shape) != (batch, target_len):
                raise ValueError(f"target_mask shape {tuple(target_mask.shape)} does not match x_t")
        semantic_len = int(semantic.shape[1])
        if semantic_mask is None:
            semantic_mask = torch.ones(batch, semantic_len, dtype=torch.bool, device=x_t.device)
        else:
            semantic_mask = semantic_mask.to(device=x_t.device).bool()
            if tuple(semantic_mask.shape) != (batch, semantic_len):
                raise ValueError(f"semantic_mask shape {tuple(semantic_mask.shape)} does not match semantic")
        if semantic_modality is None:
            semantic_modality = torch.zeros(batch, dtype=torch.long, device=x_t.device)
        else:
            semantic_modality = semantic_modality.to(device=x_t.device).long()
            if tuple(semantic_modality.shape) != (batch,):
                raise ValueError("semantic_modality must be [B]")
            if bool(torch.any(semantic_modality < 0)) or bool(torch.any(semantic_modality >= self.num_modalities)):
                raise ValueError("semantic_modality contains an invalid modality id")

        pos = _sinusoidal_time_embedding(t, self.hidden_size).to(dtype=x_t.dtype)
        condition = self.time_proj(pos) + self.speaker_proj(speaker.to(dtype=x_t.dtype))
        x = self.input_proj(x_t)
        # Dynamic sinusoidal frame positions avoid a fixed maximum sequence length.
        frame_pos = _sinusoidal_time_embedding(
            torch.arange(target_len, device=x.device, dtype=torch.float32),
            self.hidden_size,
        ).to(dtype=x.dtype)
        x = (x + frame_pos.unsqueeze(0)).masked_fill(~target_mask.unsqueeze(-1), 0.0)

        semantic = semantic.to(dtype=x.dtype)
        semantic = semantic + self.modality_embedding(semantic_modality).unsqueeze(1).to(dtype=x.dtype)
        semantic = self.semantic_proj(semantic).masked_fill(~semantic_mask.unsqueeze(-1), 0.0)
        speaker_token = self.speaker_proj(speaker.to(dtype=x.dtype)).unsqueeze(1) + self.speaker_prompt.to(dtype=x.dtype)
        semantic = torch.cat([semantic, speaker_token], dim=1)
        semantic_mask = torch.cat(
            [semantic_mask, torch.ones(batch, 1, dtype=torch.bool, device=x.device)], dim=1
        )
        for layer in self.layers:
            x = layer(
                x,
                condition,
                target_mask=target_mask,
                semantic=semantic,
                semantic_mask=semantic_mask,
            )
        velocity = self.output_proj(self.output_norm(x)).masked_fill(~target_mask.unsqueeze(-1), 0.0)
        return DDLFMOutput(velocity=velocity)


__all__ = ["DDLFMDecoder", "DDLFMAdaLNBlock", "DDLFMOutput"]
