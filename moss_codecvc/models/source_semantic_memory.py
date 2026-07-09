from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from moss_codecvc.modes import VC_MODE_NO_TEXT, VC_MODE_TEXT


@dataclass
class SourceSemanticMemoryState:
    memory: torch.Tensor
    mask: torch.Tensor | None
    stats: dict[str, float]


@dataclass
class SourceSemanticAdapterOutput:
    hidden_states: torch.Tensor
    attention_weights: torch.Tensor | None
    stats: dict[str, float]


def _valid_num_heads(adapter_dim: int, num_heads: int) -> int:
    num_heads = max(1, int(num_heads))
    if int(adapter_dim) % num_heads != 0:
        return 1
    return num_heads


def _sinusoidal_position_encoding(
    seq_len: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if seq_len <= 0 or dim <= 0:
        return torch.empty(seq_len, dim, device=device, dtype=dtype)
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, dim))
    )
    pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    if dim > 1:
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


class SourceSemanticMemoryEncoder(nn.Module):
    """Project frozen SSL source features into MOSS-TTS hidden space."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        dropout: float = 0.1,
        position_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.position_scale = float(position_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

    def forward(
        self,
        source_semantic_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SourceSemanticMemoryState:
        if source_semantic_features.dim() != 3:
            raise ValueError(
                "source_semantic_features must be [B, S, D], "
                f"got {tuple(source_semantic_features.shape)}"
            )
        if int(source_semantic_features.shape[-1]) != self.input_dim:
            raise ValueError(
                f"source_semantic_features dim={source_semantic_features.shape[-1]} "
                f"does not match input_dim={self.input_dim}"
            )
        module_dtype = self.net[0].weight.dtype
        memory = self.net(source_semantic_features.to(dtype=module_dtype))
        if self.position_scale > 0.0:
            pos = _sinusoidal_position_encoding(
                int(memory.shape[1]),
                int(memory.shape[2]),
                device=memory.device,
                dtype=memory.dtype,
            )
            memory = memory + float(self.position_scale) * pos.unsqueeze(0)
        mask = None if attention_mask is None else attention_mask.to(device=memory.device).bool()
        if mask is not None:
            memory = memory.masked_fill(~mask.unsqueeze(-1), 0.0)
        valid = (
            torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
            if mask is None
            else mask
        )
        per_sample_available = valid.any(dim=1).float()
        stats = {
            "semantic_memory_available_ratio": float(per_sample_available.detach().mean().item()),
            "semantic_memory_position_scale": float(self.position_scale),
            "semantic_memory_feature_norm": float(
                memory.detach().float().norm(dim=-1).masked_select(valid).mean().item()
            )
            if bool(valid.any().item())
            else 0.0,
        }
        return SourceSemanticMemoryState(memory=memory, mask=mask, stats=stats)


def _deduplicate_token_batch(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    padding_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = token_ids.long()
    if attention_mask is None:
        attention_mask = token_ids.ne(int(padding_id))
    else:
        attention_mask = attention_mask.to(device=token_ids.device).bool()
    batch_tokens: list[torch.Tensor] = []
    max_len = 1
    for batch_idx in range(token_ids.shape[0]):
        cur = token_ids[batch_idx].masked_select(attention_mask[batch_idx])
        if cur.numel() == 0:
            cur = token_ids.new_full((1,), int(padding_id))
        else:
            keep = torch.ones_like(cur, dtype=torch.bool)
            if cur.numel() > 1:
                keep[1:] = cur[1:] != cur[:-1]
            cur = cur.masked_select(keep)
        batch_tokens.append(cur)
        max_len = max(max_len, int(cur.numel()))
    padded = token_ids.new_full((token_ids.shape[0], max_len), int(padding_id))
    mask = torch.zeros((token_ids.shape[0], max_len), dtype=torch.bool, device=token_ids.device)
    for batch_idx, cur in enumerate(batch_tokens):
        padded[batch_idx, : cur.numel()] = cur
        mask[batch_idx, : cur.numel()] = cur.ne(int(padding_id))
    return padded, mask


class SourceTokenMemoryEncoder(nn.Module):
    """Embed source content token ids into the target-only content memory space."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        padding_id: int = 0,
        dropout: float = 0.1,
        position_scale: float = 0.0,
        dedup_units: bool = False,
    ) -> None:
        super().__init__()
        if int(vocab_size) <= 1:
            raise ValueError("vocab_size must be > 1 for SourceTokenMemoryEncoder")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.padding_id = int(padding_id)
        self.position_scale = float(position_scale)
        self.dedup_units = bool(dedup_units)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.padding_id)
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SourceSemanticMemoryState:
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be [B, S], got {tuple(token_ids.shape)}")
        token_ids = token_ids.to(device=self.embedding.weight.device).long()
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=token_ids.device).bool()
            if attention_mask.shape != token_ids.shape:
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} does not match token_ids {tuple(token_ids.shape)}"
                )
        if self.dedup_units:
            token_ids, attention_mask = _deduplicate_token_batch(
                token_ids,
                attention_mask,
                padding_id=self.padding_id,
            )
        mask = token_ids.ne(self.padding_id) if attention_mask is None else attention_mask
        valid_tokens = token_ids.masked_select(mask)
        if valid_tokens.numel() > 0:
            min_id = int(valid_tokens.min().item())
            max_id = int(valid_tokens.max().item())
            if min_id < 0 or max_id >= self.vocab_size:
                raise ValueError(
                    f"source content token ids out of range for vocab_size={self.vocab_size}: "
                    f"min={min_id} max={max_id}"
                )
        safe_ids = token_ids.clamp(min=0, max=self.vocab_size - 1)
        memory = self.net(self.embedding(safe_ids))
        if self.position_scale > 0.0:
            pos = _sinusoidal_position_encoding(
                int(memory.shape[1]),
                int(memory.shape[2]),
                device=memory.device,
                dtype=memory.dtype,
            )
            memory = memory + float(self.position_scale) * pos.unsqueeze(0)
        memory = memory.masked_fill(~mask.unsqueeze(-1), 0.0)
        per_sample_available = mask.any(dim=1).float()
        valid_count = mask.sum(dim=1).float()
        stats = {
            "semantic_memory_available_ratio": float(per_sample_available.detach().mean().item()),
            "semantic_memory_position_scale": float(self.position_scale),
            "semantic_memory_token_length_mean": float(valid_count.detach().mean().item()),
            "semantic_memory_dedup_units": 1.0 if self.dedup_units else 0.0,
            "semantic_memory_feature_norm": float(
                memory.detach().float().norm(dim=-1).masked_select(mask).mean().item()
            )
            if bool(mask.any().item())
            else 0.0,
        }
        return SourceSemanticMemoryState(memory=memory, mask=mask, stats=stats)


class SourceCodecBottleneckMemoryEncoder(nn.Module):
    """Compress source codec embeddings into a content/prosody memory bottleneck."""

    def __init__(
        self,
        hidden_size: int,
        *,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
        position_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        bottleneck_dim = max(1, min(int(bottleneck_dim), int(hidden_size)))
        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.position_scale = float(position_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.bottleneck_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

    def forward(
        self,
        source_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SourceSemanticMemoryState:
        if source_embeddings.dim() != 3:
            raise ValueError(f"source_embeddings must be [B, S, H], got {tuple(source_embeddings.shape)}")
        if int(source_embeddings.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"source embedding dim={source_embeddings.shape[-1]} does not match hidden_size={self.hidden_size}"
            )
        module_dtype = self.net[0].weight.dtype
        memory = self.net(source_embeddings.to(device=self.net[0].weight.device, dtype=module_dtype))
        if self.position_scale > 0.0:
            pos = _sinusoidal_position_encoding(
                int(memory.shape[1]),
                int(memory.shape[2]),
                device=memory.device,
                dtype=memory.dtype,
            )
            memory = memory + float(self.position_scale) * pos.unsqueeze(0)
        mask = None
        if attention_mask is not None:
            mask = attention_mask.to(device=memory.device).bool()
            if mask.shape != memory.shape[:2]:
                raise ValueError(f"attention_mask shape {tuple(mask.shape)} does not match memory {tuple(memory.shape[:2])}")
            memory = memory.masked_fill(~mask.unsqueeze(-1), 0.0)
        valid = (
            torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
            if mask is None
            else mask
        )
        per_sample_available = valid.any(dim=1).float()
        stats = {
            "semantic_memory_available_ratio": float(per_sample_available.detach().mean().item()),
            "semantic_memory_position_scale": float(self.position_scale),
            "semantic_memory_codec_bottleneck_dim": float(self.bottleneck_dim),
            "semantic_memory_token_length_mean": float(valid.sum(dim=1).detach().float().mean().item()),
            "semantic_memory_feature_norm": float(
                memory.detach().float().norm(dim=-1).masked_select(valid).mean().item()
            )
            if bool(valid.any().item())
            else 0.0,
        }
        return SourceSemanticMemoryState(memory=memory, mask=mask, stats=stats)


class SourceSemanticAdapter(nn.Module):
    """Target-only gated cross-attention from target hidden states to source semantic memory."""

    MODE_TO_ID = {
        VC_MODE_TEXT: 1,
        VC_MODE_NO_TEXT: 2,
    }

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        adapter_dim: int = 256,
        dropout: float = 0.0,
        init_gate: float = -2.0,
        no_text_gate: float = 1.0,
        text_gate: float = 0.0,
        allow_learned_text_gate: bool = False,
        monotonic_bias_strength: float = 0.0,
        monotonic_bias_width: float = 0.25,
    ) -> None:
        super().__init__()
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        num_heads = _valid_num_heads(adapter_dim, num_heads)
        self.hidden_size = int(hidden_size)
        self.adapter_dim = int(adapter_dim)
        self.num_heads = int(num_heads)
        self.no_text_gate = float(no_text_gate)
        self.text_gate = float(text_gate)
        self.allow_learned_text_gate = bool(allow_learned_text_gate)
        self.monotonic_bias_strength = float(monotonic_bias_strength)
        self.monotonic_bias_width = max(1.0e-4, float(monotonic_bias_width))
        self.monotonic_release_after_progress = False
        self.monotonic_release_start = 1.0
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.memory_norm = nn.LayerNorm(self.hidden_size)
        self.query_down = nn.Linear(self.hidden_size, self.adapter_dim)
        self.memory_down = nn.Linear(self.hidden_size, self.adapter_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.adapter_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.adapter_dim),
            nn.Linear(self.adapter_dim, self.hidden_size),
            nn.Dropout(float(dropout)),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(init_gate), dtype=torch.float32))
        if self.allow_learned_text_gate:
            self.text_gate_logit = nn.Parameter(torch.tensor(float(init_gate), dtype=torch.float32))
        else:
            self.register_parameter("text_gate_logit", None)

    def _mode_gate(self, vc_mode_id: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if vc_mode_id is None:
            return torch.full((batch_size,), self.no_text_gate, device=device, dtype=dtype)
        mode_ids = vc_mode_id.to(device=device).long().view(-1)
        if mode_ids.numel() == 1 and batch_size > 1:
            mode_ids = mode_ids.expand(batch_size)
        if mode_ids.numel() != batch_size:
            mode_ids = mode_ids[:batch_size]
            if mode_ids.numel() < batch_size:
                mode_ids = F.pad(mode_ids, (0, batch_size - mode_ids.numel()), value=0)
        no_text_id = int(self.MODE_TO_ID[VC_MODE_NO_TEXT])
        text_id = int(self.MODE_TO_ID[VC_MODE_TEXT])
        gate = torch.zeros((batch_size,), device=device, dtype=dtype)
        gate = torch.where(mode_ids.eq(no_text_id), torch.as_tensor(self.no_text_gate, device=device, dtype=dtype), gate)
        text_gate = (
            torch.sigmoid(self.text_gate_logit).to(device=device, dtype=dtype)
            if self.text_gate_logit is not None
            else torch.as_tensor(self.text_gate, device=device, dtype=dtype)
        )
        gate = torch.where(mode_ids.eq(text_id), text_gate, gate)
        return gate.clamp(min=0.0, max=1.0)

    def _build_monotonic_attention_bias(
        self,
        target_mask: torch.Tensor,
        source_semantic_mask: torch.Tensor | None,
        *,
        source_len: int,
        dtype: torch.dtype,
        target_progress: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        strength = float(self.monotonic_bias_strength)
        if strength <= 0.0:
            return None
        batch_size, _target_len = target_mask.shape
        source_len = int(source_len)
        if source_len <= 0:
            return None
        device = target_mask.device
        if target_progress is None:
            target_count = target_mask.sum(dim=1).clamp(min=1)
            target_order = target_mask.float().cumsum(dim=1) - 1.0
            target_pos = target_order / target_count.float().sub(1.0).clamp_min(1.0).unsqueeze(1)
        else:
            target_pos = target_progress.to(device=device, dtype=torch.float32)
            if target_pos.shape != target_mask.shape:
                if target_pos.dim() != 2 or target_pos.shape[0] != batch_size:
                    raise ValueError(
                        f"target_progress shape {tuple(target_pos.shape)} cannot align to "
                        f"target_mask {tuple(target_mask.shape)}"
                    )
                if target_pos.shape[1] < target_mask.shape[1]:
                    target_pos = F.pad(target_pos, (target_mask.shape[1] - target_pos.shape[1], 0), value=0.0)
                elif target_pos.shape[1] > target_mask.shape[1]:
                    target_pos = target_pos[:, -target_mask.shape[1] :]
        target_pos = target_pos.clamp(0.0, 1.0).to(device=device, dtype=torch.float32)

        src_index = torch.arange(source_len, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
        if source_semantic_mask is not None:
            valid_count = source_semantic_mask.to(device=device).bool().sum(dim=1).clamp(min=1)
            src_pos = src_index / valid_count.float().sub(1.0).clamp_min(1.0).unsqueeze(1)
        else:
            denom = max(1, source_len - 1)
            src_pos = src_index / float(denom)
        src_pos = src_pos.clamp(0.0, 1.0)

        dist = (src_pos[:, None, :] - target_pos[:, :, None]) / float(self.monotonic_bias_width)
        bias = -0.5 * strength * dist.pow(2)
        if bool(getattr(self, "monotonic_release_after_progress", False)):
            release_start = float(getattr(self, "monotonic_release_start", 1.0))
            release_start = min(max(release_start, 0.0), 1.0)
            if release_start >= 1.0:
                release_gate = target_pos.lt(1.0).to(dtype=bias.dtype)
            else:
                release_gate = (1.0 - target_pos).div(max(1.0e-6, 1.0 - release_start)).clamp(0.0, 1.0)
                release_gate = release_gate.to(dtype=bias.dtype)
            bias = bias * release_gate[:, :, None]
        bias = torch.where(target_mask[:, :, None], bias, torch.zeros_like(bias))
        return bias.to(dtype=dtype).repeat_interleave(self.num_heads, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_semantic_memory: torch.Tensor,
        target_mask: torch.Tensor,
        source_semantic_mask: torch.Tensor | None = None,
        vc_mode_id: torch.Tensor | None = None,
        target_progress: torch.Tensor | None = None,
    ) -> SourceSemanticAdapterOutput:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, T, H], got {tuple(hidden_states.shape)}")
        if source_semantic_memory.dim() != 3:
            raise ValueError(
                "source_semantic_memory must be [B, S, H], "
                f"got {tuple(source_semantic_memory.shape)}"
            )
        batch_size, target_len, _ = hidden_states.shape
        if source_semantic_memory.shape[0] != batch_size:
            raise ValueError("source_semantic_memory batch size mismatch")
        target_mask = target_mask.to(device=hidden_states.device).bool()
        if target_mask.shape != hidden_states.shape[:2]:
            raise ValueError(f"target_mask shape {tuple(target_mask.shape)} does not match {tuple(hidden_states.shape[:2])}")
        residual_dtype = hidden_states.dtype
        adapter_dtype = self.query_norm.weight.dtype
        adapter_hidden = hidden_states.to(dtype=adapter_dtype)
        memory = source_semantic_memory.to(device=hidden_states.device, dtype=adapter_dtype)
        memory_mask = None
        if source_semantic_mask is not None:
            memory_mask = source_semantic_mask.to(device=hidden_states.device).bool()
        if memory_mask is not None and not bool(memory_mask.any().item()):
            return SourceSemanticAdapterOutput(hidden_states=hidden_states, attention_weights=None, stats={})
        mode_gate = self._mode_gate(vc_mode_id, batch_size, hidden_states.device, adapter_dtype)
        learned_gate = torch.sigmoid(self.gate_logit).to(device=hidden_states.device, dtype=adapter_dtype)
        combined_gate = mode_gate * learned_gate
        if not bool(target_mask.any().item()) or float(combined_gate.detach().float().max().item()) <= 0.0:
            return SourceSemanticAdapterOutput(
                hidden_states=hidden_states,
                attention_weights=None,
                stats={
                    "source_semantic_gate_mean": float(combined_gate.detach().float().mean().item()),
                    "source_semantic_no_text_gate_mean": float(mode_gate.detach().float().mean().item()),
                },
            )

        query = self.query_down(self.query_norm(adapter_hidden))
        key_value = self.memory_down(self.memory_norm(memory))
        attn_mask = self._build_monotonic_attention_bias(
            target_mask,
            memory_mask,
            source_len=int(key_value.shape[1]),
            dtype=query.dtype,
            target_progress=target_progress,
        )
        key_padding_mask = None if memory_mask is None else ~memory_mask
        if attn_mask is not None and key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(dtype=query.dtype).masked_fill(
                key_padding_mask,
                -1.0e4,
            )
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        delta = self.out(attn_out).to(dtype=residual_dtype)
        residual_gate = (
            target_mask.unsqueeze(-1).to(dtype=residual_dtype)
            * combined_gate.to(dtype=residual_dtype).view(batch_size, 1, 1)
        )
        updated = hidden_states + residual_gate * delta

        stats = {
            "source_semantic_gate_mean": float(combined_gate.detach().float().mean().item()),
            "source_semantic_no_text_gate_mean": float(mode_gate.detach().float().mean().item()),
            "source_semantic_monotonic_bias_strength": float(self.monotonic_bias_strength),
            "source_semantic_monotonic_bias_width": float(self.monotonic_bias_width),
            "source_semantic_monotonic_release_after_progress": float(
                bool(getattr(self, "monotonic_release_after_progress", False))
            ),
            "source_semantic_monotonic_release_start": float(getattr(self, "monotonic_release_start", 1.0)),
        }
        with torch.no_grad():
            token_mask = target_mask.to(device=hidden_states.device).bool()
            token_count = token_mask.sum().clamp(min=1)
            hidden_norm = hidden_states.detach().float().norm(dim=-1)
            raw_delta_norm = delta.detach().float().norm(dim=-1)
            effective_delta = (residual_gate * delta).detach().float()
            effective_delta_norm = effective_delta.norm(dim=-1)
            mean_hidden_norm = hidden_norm.masked_select(token_mask).mean()
            mean_effective_delta_norm = effective_delta_norm.masked_select(token_mask).mean()
            prompt_mask = ~token_mask
            prompt_delta_norm = (
                effective_delta_norm.masked_select(prompt_mask).mean()
                if bool(prompt_mask.any().item())
                else torch.zeros((), device=effective_delta_norm.device)
            )
            stats["source_semantic_hidden_norm"] = float(mean_hidden_norm.item())
            stats["source_semantic_raw_delta_norm"] = float(raw_delta_norm.masked_select(token_mask).mean().item())
            stats["source_semantic_delta_norm"] = float(mean_effective_delta_norm.item())
            stats["source_semantic_delta_ratio"] = float(
                (mean_effective_delta_norm / mean_hidden_norm.clamp_min(1.0e-6)).item()
            )
            stats["source_semantic_delta_max"] = float(effective_delta_norm.masked_select(token_mask).max().item())
            stats["source_semantic_prompt_delta_norm"] = float(prompt_delta_norm.item())
            stats["source_semantic_target_fraction"] = float(
                token_count.float().div(max(1, int(target_mask.numel()))).item()
            )

            probs = attn_weights.detach().float()
            if memory_mask is not None:
                valid_source = memory_mask.to(device=probs.device).bool()
                probs = probs.masked_fill(~valid_source[:, None, None, :], 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
                stats["source_semantic_memory_valid_ratio"] = float(valid_source.float().mean().item())
            else:
                valid_source = torch.ones(probs.shape[0], probs.shape[-1], dtype=torch.bool, device=probs.device)
            probs = probs.clamp_min(1.0e-9)
            entropy = -(probs * probs.log()).sum(dim=-1)
            denom = torch.log(torch.tensor(float(max(2, probs.shape[-1])), device=probs.device))
            entropy = entropy / denom.clamp_min(1.0e-6)
            target_weight = target_mask[:, None, :].to(device=entropy.device, dtype=entropy.dtype)
            target_head_count = (target_weight.sum() * probs.shape[1]).clamp(min=1.0)
            if bool(target_mask.any().item()):
                stats["source_semantic_attn_entropy"] = float(
                    (entropy * target_weight).sum().item() / target_head_count.item()
                )
                peak = probs.max(dim=-1).values
                stats["source_semantic_attn_peak_mean"] = float((peak * target_weight).sum().item() / target_head_count.item())

                source_positions = torch.linspace(0.0, 1.0, probs.shape[-1], device=probs.device, dtype=probs.dtype)
                expected_pos = (probs * source_positions.view(1, 1, 1, -1)).sum(dim=-1)
                stats["source_semantic_attn_expected_pos_mean"] = float(
                    (expected_pos * target_weight).sum().item() / target_head_count.item()
                )

                if target_progress is None:
                    per_sample_target_count = target_mask.sum(dim=1).clamp(min=1)
                    target_order = target_mask.float().cumsum(dim=1) - 1.0
                    target_rel_pos = target_order / (
                        per_sample_target_count.float().sub(1.0).clamp_min(1.0).unsqueeze(1)
                    )
                else:
                    target_rel_pos = target_progress.to(device=probs.device, dtype=probs.dtype)
                    if target_rel_pos.shape != target_mask.shape:
                        if target_rel_pos.shape[1] < target_mask.shape[1]:
                            target_rel_pos = F.pad(
                                target_rel_pos,
                                (target_mask.shape[1] - target_rel_pos.shape[1], 0),
                                value=0.0,
                            )
                        elif target_rel_pos.shape[1] > target_mask.shape[1]:
                            target_rel_pos = target_rel_pos[:, -target_mask.shape[1] :]
                target_rel_pos = target_rel_pos.to(device=probs.device, dtype=probs.dtype).clamp(0.0, 1.0)

                def _segment_expected(name: str, segment_mask: torch.Tensor) -> None:
                    segment_weight = segment_mask[:, None, :].to(device=probs.device, dtype=probs.dtype)
                    denom_value = (segment_weight.sum() * probs.shape[1]).clamp(min=1.0)
                    if float(segment_weight.sum().item()) > 0.0:
                        stats[name] = float((expected_pos * segment_weight).sum().item() / denom_value.item())

                begin_mask = token_mask & target_rel_pos.le(1.0 / 3.0)
                mid_mask = token_mask & target_rel_pos.gt(1.0 / 3.0) & target_rel_pos.le(2.0 / 3.0)
                end_mask = token_mask & target_rel_pos.gt(2.0 / 3.0)
                _segment_expected("source_semantic_attn_expected_pos_begin", begin_mask)
                _segment_expected("source_semantic_attn_expected_pos_mid", mid_mask)
                _segment_expected("source_semantic_attn_expected_pos_end", end_mask)

                expected_per_token = expected_pos.mean(dim=1)
                rel_values = target_rel_pos.masked_select(token_mask)
                expected_values = expected_per_token.masked_select(token_mask)
                if rel_values.numel() > 1:
                    rel_centered = rel_values - rel_values.mean()
                    exp_centered = expected_values - expected_values.mean()
                    stats["source_semantic_attn_expected_pos_slope"] = float(
                        (rel_centered * exp_centered).mean().div(rel_centered.pow(2).mean().clamp_min(1.0e-6)).item()
                    )

                per_sample_head_target_count = (
                    target_mask.sum(dim=1).to(device=probs.device, dtype=probs.dtype) * probs.shape[1]
                ).clamp(min=1.0)
                source_mass = (probs * target_weight.unsqueeze(-1)).sum(dim=(1, 2)) / per_sample_head_target_count.unsqueeze(1)
                valid_count = valid_source.sum(dim=1).clamp(min=1).to(dtype=source_mass.dtype)
                coverage_threshold = 0.5 / valid_count.unsqueeze(1)
                covered = source_mass.gt(coverage_threshold) & valid_source
                stats["source_semantic_attn_coverage"] = float(
                    (covered.sum(dim=1).to(dtype=source_mass.dtype) / valid_count).mean().item()
                )
        return SourceSemanticAdapterOutput(hidden_states=updated, attention_weights=attn_weights, stats=stats)
