from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SpeakerCrossAttentionOutput:
    delta: torch.Tensor
    stats: dict[str, float]


def _valid_num_heads(adapter_dim: int, num_heads: int) -> int:
    num_heads = max(1, int(num_heads))
    if int(adapter_dim) % num_heads != 0:
        return 1
    return num_heads


class SpeakerTokenProjector(nn.Module):
    """Project one speaker vector into K hidden-size pseudo-tokens."""

    def __init__(
        self,
        speaker_embedding_dim: int,
        hidden_size: int,
        *,
        num_tokens: int = 8,
        adapter_dim: int = 256,
        dropout: float = 0.0,
        output_init_std: float | None = None,
    ) -> None:
        super().__init__()
        if int(speaker_embedding_dim) <= 0:
            raise ValueError("speaker_embedding_dim must be positive")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if int(num_tokens) <= 0:
            raise ValueError("num_tokens must be positive")
        self.speaker_embedding_dim = int(speaker_embedding_dim)
        self.hidden_size = int(hidden_size)
        self.num_tokens = int(num_tokens)
        self.output_init_std = float(output_init_std) if output_init_std is not None else None
        adapter_dim = max(1, int(adapter_dim))
        output = nn.Linear(adapter_dim, self.num_tokens * self.hidden_size)
        if self.output_init_std is not None:
            nn.init.normal_(output.weight, mean=0.0, std=self.output_init_std)
            nn.init.zeros_(output.bias)
        self.net = nn.Sequential(
            nn.LayerNorm(self.speaker_embedding_dim),
            nn.Linear(self.speaker_embedding_dim, adapter_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            output,
        )
        self.token_norm = nn.LayerNorm(self.hidden_size) if self.output_init_std is None else nn.Identity()

    def forward(self, speaker_embedding: torch.Tensor) -> torch.Tensor:
        if speaker_embedding.dim() != 2:
            raise ValueError(f"speaker_embedding must be [B, D], got {tuple(speaker_embedding.shape)}")
        if int(speaker_embedding.shape[-1]) != self.speaker_embedding_dim:
            raise ValueError(
                f"speaker_embedding dim={speaker_embedding.shape[-1]} "
                f"does not match {self.speaker_embedding_dim}"
            )
        module_dtype = self.net[0].weight.dtype
        tokens = self.net(speaker_embedding.to(dtype=module_dtype))
        tokens = tokens.view(int(speaker_embedding.shape[0]), self.num_tokens, self.hidden_size)
        return self.token_norm(tokens)


class SpeakerSequenceProjector(nn.Module):
    """Project a reference-speaker feature sequence into hidden-size tokens."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_size),
            nn.Dropout(float(dropout)),
        )
        self.token_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, speaker_sequence: torch.Tensor) -> torch.Tensor:
        if speaker_sequence.dim() != 3:
            raise ValueError(f"speaker_sequence must be [B, S, D], got {tuple(speaker_sequence.shape)}")
        if int(speaker_sequence.shape[-1]) != self.input_dim:
            raise ValueError(
                f"speaker_sequence dim={speaker_sequence.shape[-1]} "
                f"does not match {self.input_dim}"
            )
        module_dtype = self.net[0].weight.dtype
        tokens = self.net(speaker_sequence.to(dtype=module_dtype))
        return self.token_norm(tokens)


class SpeakerCrossAttentionLayer(nn.Module):
    """Target-position residual cross-attention over speaker pseudo-tokens."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 8,
        adapter_dim: int = 256,
        dropout: float = 0.0,
        gate_init: float = 0.0,
        output_scale: float = 1.0,
        normalize_tokens: bool = True,
    ) -> None:
        super().__init__()
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        num_heads = _valid_num_heads(adapter_dim, int(num_heads))
        self.hidden_size = int(hidden_size)
        self.adapter_dim = int(adapter_dim)
        self.num_heads = int(num_heads)
        self.output_scale = float(output_scale)
        self.runtime_scale_multiplier = 1.0
        self.normalize_tokens = bool(normalize_tokens)
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.token_norm = nn.LayerNorm(self.hidden_size) if self.normalize_tokens else nn.Identity()
        self.query_down = nn.Linear(self.hidden_size, self.adapter_dim)
        self.token_down = nn.Linear(self.hidden_size, self.adapter_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.adapter_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.adapter_dim),
            nn.Linear(self.adapter_dim, self.hidden_size),
            nn.Dropout(float(dropout)),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def set_runtime_scale_multiplier(self, multiplier: float) -> None:
        self.runtime_scale_multiplier = float(multiplier)

    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_tokens: torch.Tensor,
        target_mask: torch.Tensor,
        speaker_mask: torch.Tensor | None = None,
    ) -> SpeakerCrossAttentionOutput:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, T, H], got {tuple(hidden_states.shape)}")
        if speaker_tokens.dim() != 3:
            raise ValueError(f"speaker_tokens must be [B, K, H], got {tuple(speaker_tokens.shape)}")
        batch_size, _target_len, hidden_size = hidden_states.shape
        if int(hidden_size) != self.hidden_size:
            raise ValueError(f"hidden size mismatch: got {hidden_size}, expected {self.hidden_size}")
        if speaker_tokens.shape[0] != batch_size or int(speaker_tokens.shape[-1]) != self.hidden_size:
            raise ValueError("speaker_tokens shape does not match hidden_states")
        target_mask = target_mask.to(device=hidden_states.device).bool()
        if target_mask.shape != hidden_states.shape[:2]:
            raise ValueError(f"target_mask shape {tuple(target_mask.shape)} does not match {tuple(hidden_states.shape[:2])}")
        speaker_token_count = int(speaker_tokens.shape[1])
        if speaker_mask is None:
            speaker_token_mask = torch.ones(
                batch_size,
                speaker_token_count,
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            speaker_mask = speaker_mask.to(device=hidden_states.device).bool()
            if speaker_mask.dim() == 1:
                speaker_mask = speaker_mask.view(-1)
                if speaker_mask.numel() == 1 and batch_size > 1:
                    speaker_mask = speaker_mask.expand(batch_size)
                if speaker_mask.numel() != batch_size:
                    raise ValueError(f"speaker_mask batch mismatch: got {speaker_mask.numel()}, expected {batch_size}")
                speaker_token_mask = speaker_mask.view(batch_size, 1).expand(batch_size, speaker_token_count)
            elif speaker_mask.dim() == 2:
                if speaker_mask.shape[0] != batch_size or speaker_mask.shape[1] != speaker_token_count:
                    raise ValueError(
                        f"speaker_mask shape {tuple(speaker_mask.shape)} does not match "
                        f"{(batch_size, speaker_token_count)}"
                    )
                speaker_token_mask = speaker_mask
            else:
                raise ValueError(f"speaker_mask must be [B] or [B, K], got {tuple(speaker_mask.shape)}")
        speaker_row_mask = speaker_token_mask.any(dim=1)
        residual_dtype = hidden_states.dtype
        adapter_dtype = self.query_norm.weight.dtype
        if not bool(target_mask.any().item()) or not bool(speaker_row_mask.any().item()):
            empty_delta = torch.zeros_like(hidden_states)
            return SpeakerCrossAttentionOutput(
                delta=empty_delta,
                stats={
                    "speaker_cross_attn_gate_mean": float(torch.sigmoid(self.gate_logit.detach().float()).item()),
                    "speaker_cross_attn_delta_norm": 0.0,
                    "speaker_cross_attn_raw_delta_norm": 0.0,
                    "speaker_cross_attn_hidden_norm": 0.0,
                    "speaker_cross_attn_delta_ratio": 0.0,
                    "speaker_cross_attn_valid_tokens": 0.0,
                },
            )

        query = self.query_down(self.query_norm(hidden_states.to(dtype=adapter_dtype)))
        tokens = self.token_down(
            self.token_norm(speaker_tokens.to(device=hidden_states.device, dtype=adapter_dtype))
        )
        safe_speaker_token_mask = speaker_token_mask.clone()
        empty_rows = ~safe_speaker_token_mask.any(dim=1)
        if bool(empty_rows.any().item()) and speaker_token_count > 0:
            safe_speaker_token_mask[empty_rows, 0] = True
        attn_out, _ = self.cross_attn(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=~safe_speaker_token_mask,
            need_weights=False,
        )
        raw_delta = self.out(attn_out).to(dtype=residual_dtype)
        gate = torch.sigmoid(self.gate_logit).to(device=hidden_states.device, dtype=residual_dtype)
        residual_gate = (
            target_mask.unsqueeze(-1).to(dtype=residual_dtype)
            * speaker_row_mask.to(device=hidden_states.device, dtype=residual_dtype).view(batch_size, 1, 1)
            * gate
        )
        effective_output_scale = float(self.output_scale) * float(self.runtime_scale_multiplier)
        output_scale = torch.as_tensor(effective_output_scale, device=hidden_states.device, dtype=residual_dtype)
        delta = output_scale * residual_gate * raw_delta

        stats: dict[str, float] = {
            "speaker_cross_attn_gate_mean": float(gate.detach().float().item()),
            "speaker_cross_attn_output_scale": float(effective_output_scale),
            "speaker_cross_attn_runtime_scale_multiplier": float(self.runtime_scale_multiplier),
        }
        with torch.no_grad():
            token_mask = target_mask.to(device=hidden_states.device).bool()
            hidden_norm = hidden_states.detach().float().norm(dim=-1)
            raw_delta_norm = raw_delta.detach().float().norm(dim=-1)
            effective_delta_norm = delta.detach().float().norm(dim=-1)
            mean_hidden_norm = hidden_norm.masked_select(token_mask).mean()
            mean_raw_delta_norm = raw_delta_norm.masked_select(token_mask).mean()
            mean_effective_delta_norm = effective_delta_norm.masked_select(token_mask).mean()
            stats.update(
                {
                    "speaker_cross_attn_hidden_norm": float(mean_hidden_norm.item()),
                    "speaker_cross_attn_raw_delta_norm": float(mean_raw_delta_norm.item()),
                    "speaker_cross_attn_delta_norm": float(mean_effective_delta_norm.item()),
                    "speaker_cross_attn_delta_ratio": float(
                        (mean_effective_delta_norm / mean_hidden_norm.clamp_min(1.0e-6)).item()
                    ),
                    "speaker_cross_attn_valid_tokens": float(
                        speaker_token_mask.detach().float().sum(dim=1).mean().item()
                    ),
                }
            )
        return SpeakerCrossAttentionOutput(delta=delta, stats=stats)
