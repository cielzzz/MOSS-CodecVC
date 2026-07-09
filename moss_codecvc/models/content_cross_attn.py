from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ContentMemoryState:
    memory: torch.Tensor
    mask: torch.Tensor | None
    stats: dict[str, float]


@dataclass
class ContentCrossAttentionOutput:
    delta: torch.Tensor
    attention_weights: torch.Tensor | None
    stats: dict[str, float]


def _valid_num_heads(adapter_dim: int, num_heads: int) -> int:
    num_heads = max(1, int(num_heads))
    if int(adapter_dim) % num_heads != 0:
        return 1
    return num_heads


class ContentConformerBlock(nn.Module):
    """Small Conformer-style encoder block for frozen BNF source features."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 8,
        dropout: float = 0.0,
        conv_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        num_heads = _valid_num_heads(hidden_size, int(num_heads))
        conv_kernel_size = max(1, int(conv_kernel_size))
        if conv_kernel_size % 2 == 0:
            conv_kernel_size += 1
        ff_dim = hidden_size * 4
        self.ff1_norm = nn.LayerNorm(hidden_size)
        self.ff1 = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(float(dropout)),
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(float(dropout))
        self.conv_norm = nn.LayerNorm(hidden_size)
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=hidden_size,
        )
        self.pointwise = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv_dropout = nn.Dropout(float(dropout))
        self.ff2_norm = nn.LayerNorm(hidden_size)
        self.ff2 = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(float(dropout)),
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = hidden_states
        x = x + 0.5 * self.ff1(self.ff1_norm(x))
        key_padding_mask = None if mask is None else ~mask.to(device=x.device).bool()
        attn_in = self.attn_norm(x)
        attn_out, _ = self.self_attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)
        conv_in = self.conv_norm(x).transpose(1, 2)
        conv_out = self.pointwise(F.silu(self.depthwise(conv_in))).transpose(1, 2)
        x = x + self.conv_dropout(conv_out)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))
        x = self.final_norm(x)
        if mask is not None:
            x = x.masked_fill(~mask.to(device=x.device).bool().unsqueeze(-1), 0.0)
        return x


class ContentConformerEncoder(nn.Module):
    """Project source WavLM/BNF sequences into hidden-size content memory tokens."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        *,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        conv_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.input = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_size),
            nn.Dropout(float(dropout)),
        )
        self.layers = nn.ModuleList(
            [
                ContentConformerBlock(
                    self.hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(max(0, int(num_layers)))
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> ContentMemoryState:
        if features.dim() != 3:
            raise ValueError(f"content features must be [B, S, D], got {tuple(features.shape)}")
        if int(features.shape[-1]) != self.input_dim:
            raise ValueError(f"content feature dim={features.shape[-1]} does not match input_dim={self.input_dim}")
        module_dtype = self.input[0].weight.dtype
        memory = self.input(features.to(dtype=module_dtype))
        if mask is None:
            content_mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        else:
            content_mask = mask.to(device=memory.device).bool()
            if content_mask.shape != memory.shape[:2]:
                raise ValueError(
                    f"content mask shape {tuple(content_mask.shape)} does not match {tuple(memory.shape[:2])}"
                )
        for layer in self.layers:
            memory = layer(memory, content_mask)
        memory = self.output_norm(memory).masked_fill(~content_mask.unsqueeze(-1), 0.0)
        valid_norm = memory.detach().float().norm(dim=-1).masked_select(content_mask)
        stats = {
            "content_cross_attn_memory_tokens": float(memory.shape[1]),
            "content_cross_attn_memory_valid_tokens": float(content_mask.detach().float().sum(dim=1).mean().item()),
            "content_cross_attn_memory_norm": float(valid_norm.mean().item()) if valid_norm.numel() > 0 else 0.0,
        }
        return ContentMemoryState(memory=memory, mask=content_mask, stats=stats)


class ContentCrossAttentionLayer(nn.Module):
    """Target-position residual cross-attention over source content memory."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 8,
        adapter_dim: int = 256,
        dropout: float = 0.0,
        gate_init: float = -0.5,
        output_scale: float = 0.3,
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
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.memory_norm = nn.LayerNorm(self.hidden_size)
        self.query_down = nn.Linear(self.hidden_size, self.adapter_dim)
        self.memory_down = nn.Linear(self.hidden_size, self.adapter_dim)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        content_memory: torch.Tensor,
        target_mask: torch.Tensor,
        content_mask: torch.Tensor | None = None,
    ) -> ContentCrossAttentionOutput:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, T, H], got {tuple(hidden_states.shape)}")
        if content_memory.dim() != 3:
            raise ValueError(f"content_memory must be [B, S, H], got {tuple(content_memory.shape)}")
        batch_size, _target_len, hidden_size = hidden_states.shape
        if int(hidden_size) != self.hidden_size:
            raise ValueError(f"hidden size mismatch: got {hidden_size}, expected {self.hidden_size}")
        if content_memory.shape[0] != batch_size or int(content_memory.shape[-1]) != self.hidden_size:
            raise ValueError("content_memory shape does not match hidden_states")
        target_mask = target_mask.to(device=hidden_states.device).bool()
        if target_mask.shape != hidden_states.shape[:2]:
            raise ValueError(f"target_mask shape {tuple(target_mask.shape)} does not match {tuple(hidden_states.shape[:2])}")
        if content_mask is None:
            content_token_mask = torch.ones(content_memory.shape[:2], device=hidden_states.device, dtype=torch.bool)
        else:
            content_token_mask = content_mask.to(device=hidden_states.device).bool()
            if content_token_mask.shape != content_memory.shape[:2]:
                raise ValueError(
                    f"content_mask shape {tuple(content_token_mask.shape)} "
                    f"does not match {tuple(content_memory.shape[:2])}"
                )
        content_row_mask = content_token_mask.any(dim=1)
        residual_dtype = hidden_states.dtype
        adapter_dtype = self.query_norm.weight.dtype
        if not bool(target_mask.any().item()) or not bool(content_row_mask.any().item()):
            return ContentCrossAttentionOutput(
                delta=torch.zeros_like(hidden_states),
                attention_weights=None,
                stats={
                    "content_cross_attn_gate_mean": float(torch.sigmoid(self.gate_logit.detach().float()).item()),
                    "content_cross_attn_delta_norm": 0.0,
                    "content_cross_attn_raw_delta_norm": 0.0,
                    "content_cross_attn_hidden_norm": 0.0,
                    "content_cross_attn_delta_ratio": 0.0,
                    "content_cross_attn_valid_tokens": 0.0,
                },
            )
        query = self.query_down(self.query_norm(hidden_states.to(dtype=adapter_dtype)))
        memory = self.memory_down(
            self.memory_norm(content_memory.to(device=hidden_states.device, dtype=adapter_dtype))
        )
        safe_content_mask = content_token_mask.clone()
        empty_rows = ~safe_content_mask.any(dim=1)
        if bool(empty_rows.any().item()) and int(content_memory.shape[1]) > 0:
            safe_content_mask[empty_rows, 0] = True
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=~safe_content_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        raw_delta = self.out(attn_out).to(dtype=residual_dtype)
        gate = torch.sigmoid(self.gate_logit).to(device=hidden_states.device, dtype=residual_dtype)
        residual_gate = (
            target_mask.unsqueeze(-1).to(dtype=residual_dtype)
            * content_row_mask.to(device=hidden_states.device, dtype=residual_dtype).view(batch_size, 1, 1)
            * gate
        )
        delta = torch.as_tensor(self.output_scale, device=hidden_states.device, dtype=residual_dtype) * residual_gate * raw_delta
        stats: dict[str, float] = {
            "content_cross_attn_gate_mean": float(gate.detach().float().item()),
            "content_cross_attn_output_scale": float(self.output_scale),
        }
        with torch.no_grad():
            hidden_norm = hidden_states.detach().float().norm(dim=-1)
            raw_delta_norm = raw_delta.detach().float().norm(dim=-1)
            effective_delta_norm = delta.detach().float().norm(dim=-1)
            valid_targets = target_mask.to(device=hidden_states.device).bool()
            mean_hidden_norm = hidden_norm.masked_select(valid_targets).mean()
            mean_raw_delta_norm = raw_delta_norm.masked_select(valid_targets).mean()
            mean_delta_norm = effective_delta_norm.masked_select(valid_targets).mean()
            probs = attn_weights.detach().float()
            probs = probs.masked_fill(~safe_content_mask[:, None, None, :], 0.0)
            denom = probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
            probs = probs / denom
            entropy = -(probs.clamp_min(1.0e-9).log() * probs).sum(dim=-1)
            peak = probs.max(dim=-1).values
            stats.update(
                {
                    "content_cross_attn_hidden_norm": float(mean_hidden_norm.item()),
                    "content_cross_attn_raw_delta_norm": float(mean_raw_delta_norm.item()),
                    "content_cross_attn_delta_norm": float(mean_delta_norm.item()),
                    "content_cross_attn_delta_ratio": float(
                        (mean_delta_norm / mean_hidden_norm.clamp_min(1.0e-8)).item()
                    ),
                    "content_cross_attn_valid_tokens": float(content_token_mask.detach().float().sum(dim=1).mean().item()),
                    "content_cross_attn_attn_entropy": float(entropy[:, :, valid_targets.any(dim=0)].mean().item())
                    if bool(valid_targets.any().item())
                    else 0.0,
                    "content_cross_attn_attn_peak_mean": float(peak[:, :, valid_targets.any(dim=0)].mean().item())
                    if bool(valid_targets.any().item())
                    else 0.0,
                }
            )
        return ContentCrossAttentionOutput(delta=delta, attention_weights=attn_weights, stats=stats)


class ContentPhonemeClassifierHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, *, adapter_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        if int(vocab_size) <= 1:
            raise ValueError("vocab_size must be > 1")
        adapter_dim = max(1, int(adapter_dim))
        self.vocab_size = int(vocab_size)
        self.net = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), adapter_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(adapter_dim, self.vocab_size),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.net(memory)


def compute_guided_attention_loss(
    attentions: list[torch.Tensor],
    target_mask: torch.Tensor,
    content_mask: torch.Tensor | None,
    *,
    band_frames: int = 3,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if not attentions:
        return None, {"content_guided_attn_layers": 0.0}
    states: list[torch.Tensor] = []
    stats: dict[str, float] = {"content_guided_attn_layers": float(len(attentions))}
    for attn in attentions:
        if attn.dim() != 4:
            continue
        probs = attn.float()
        device = probs.device
        batch_size, _heads, target_len, source_len = probs.shape
        cur_target_mask = target_mask.to(device=device).bool()
        if cur_target_mask.shape != (batch_size, target_len):
            cur_target_mask = cur_target_mask[:, -target_len:]
        if content_mask is None:
            cur_content_mask = torch.ones(batch_size, source_len, dtype=torch.bool, device=device)
        else:
            cur_content_mask = content_mask.to(device=device).bool()
            if cur_content_mask.shape != (batch_size, source_len):
                cur_content_mask = cur_content_mask[:, :source_len]
                if cur_content_mask.shape[1] < source_len:
                    pad = torch.zeros(batch_size, source_len - cur_content_mask.shape[1], dtype=torch.bool, device=device)
                    cur_content_mask = torch.cat([cur_content_mask, pad], dim=1)
        probs = probs.masked_fill(~cur_content_mask[:, None, None, :], 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        target_rank = cur_target_mask.long().cumsum(dim=1).sub(1).clamp_min(0).float()
        target_count = cur_target_mask.sum(dim=1).clamp_min(1).float()
        content_count = cur_content_mask.sum(dim=1).clamp_min(1).float()
        target_denom = (target_count - 1).clamp_min(1.0)
        expected = target_rank / target_denom.unsqueeze(1) * (content_count - 1).clamp_min(0.0).unsqueeze(1)
        src_pos = torch.arange(source_len, device=device, dtype=torch.float32).view(1, 1, source_len)
        outside = (src_pos - expected.unsqueeze(-1)).abs() > float(max(0, int(band_frames)))
        outside = outside & cur_content_mask.unsqueeze(1)
        outside_mass = (probs * outside[:, None, :, :].to(dtype=probs.dtype)).sum(dim=-1)
        valid = cur_target_mask[:, None, :].expand_as(outside_mass)
        if bool(valid.any().item()):
            states.append(outside_mass.masked_select(valid).mean())
    if not states:
        return None, stats
    loss = torch.stack(states).mean()
    stats["content_guided_attn_loss"] = float(loss.detach().item())
    stats["content_guided_attn_band_frames"] = float(max(0, int(band_frames)))
    return loss, stats


def compute_phoneme_classifier_loss(
    logits: torch.Tensor,
    token_ids: torch.Tensor | None,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if token_ids is None:
        return None, {"content_phoneme_classifier_missing": 1.0}
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, S, V], got {tuple(logits.shape)}")
    token_ids = token_ids.to(device=logits.device).long()
    if token_ids.dim() != 2 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError(f"token_ids must be [B, L], got {tuple(token_ids.shape)}")
    if token_mask is None:
        mask = token_ids.ge(0)
    else:
        mask = token_mask.to(device=logits.device).bool()
        if mask.shape != token_ids.shape:
            raise ValueError(f"token_mask shape {tuple(mask.shape)} does not match token_ids {tuple(token_ids.shape)}")
    target_len = int(token_ids.shape[1])
    if int(logits.shape[1]) != target_len:
        logits_t = logits.transpose(1, 2).float()
        logits = F.interpolate(logits_t, size=target_len, mode="linear", align_corners=False).transpose(1, 2)
    valid = mask & token_ids.ge(0) & token_ids.lt(int(logits.shape[-1]))
    if not bool(valid.any().item()):
        return None, {"content_phoneme_classifier_valid_tokens": 0.0}
    safe_token_ids = token_ids.masked_fill(~valid, -100)
    loss = F.cross_entropy(
        logits.float().reshape(-1, int(logits.shape[-1])),
        safe_token_ids.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(token_ids)
    loss = loss.masked_select(valid).mean()
    with torch.no_grad():
        pred = logits.detach().float().argmax(dim=-1)
        acc = (pred[valid] == token_ids[valid]).float().mean()
    return loss, {
        "content_phoneme_classifier_loss": float(loss.detach().item()),
        "content_phoneme_classifier_acc": float(acc.item()),
        "content_phoneme_classifier_valid_tokens": float(valid.detach().float().sum().item()),
    }
