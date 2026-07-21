from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TimbreMemoryState:
    timbre_tokens: torch.Tensor
    timbre_mask: torch.Tensor | None = None


def _valid_num_heads(adapter_dim: int, num_heads: int) -> int:
    num_heads = max(1, int(num_heads))
    if adapter_dim % num_heads != 0:
        return 1
    return num_heads


def _mask_padded(hidden_states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden_states
    return hidden_states.masked_fill(~mask.bool().unsqueeze(-1), 0.0)


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, expansion: int = 4) -> None:
        super().__init__()
        inner_dim = max(dim, int(dim) * int(expansion))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class MoEFeedForwardModule(nn.Module):
    """Small dense-gated MoE FFN for optional TTE ablations."""

    def __init__(self, dim: int, dropout: float = 0.0, num_experts: int = 4, expansion: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, max(dim, dim * expansion)),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(dim, dim * expansion), dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden_states)
        weights = torch.softmax(self.gate(normalized), dim=-1)
        expert_outputs = torch.stack([expert(normalized) for expert in self.experts], dim=-2)
        mixed = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        return self.dropout(mixed)


class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.channel_norm = nn.GroupNorm(1, dim)
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual_dtype = hidden_states.dtype
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.glu(self.pointwise_in(hidden_states), dim=1)
        hidden_states = self.depthwise(hidden_states)
        hidden_states = self.channel_norm(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.pointwise_out(hidden_states).transpose(1, 2)
        hidden_states = self.dropout(hidden_states).to(dtype=residual_dtype)
        return _mask_padded(hidden_states, mask)


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        conv_kernel_size: int = 7,
        use_moe_ffn: bool = False,
    ) -> None:
        super().__init__()
        ffn_cls = MoEFeedForwardModule if use_moe_ffn else FeedForwardModule
        self.ffn1 = ffn_cls(dim, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=_valid_num_heads(dim, num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = ffn_cls(dim, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        key_padding_mask = None if mask is None else ~mask.bool()
        hidden_states = hidden_states + 0.5 * self.ffn1(hidden_states)
        attn_in = self.self_attn_norm(hidden_states)
        attn_out, _ = self.self_attn(
            query=attn_in,
            key=attn_in,
            value=attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.self_attn_dropout(attn_out)
        hidden_states = hidden_states + self.conv(hidden_states, mask)
        hidden_states = hidden_states + 0.5 * self.ffn2(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        return _mask_padded(hidden_states, mask)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        conv_kernel_size: int = 7,
        use_moe_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                    use_moe_ffn=use_moe_ffn,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        return hidden_states


class ReferenceCodecTimbreMemory(nn.Module):
    """Compresses target-timbre reference codec embeddings into K memory tokens.

    The module intentionally consumes only the S2/reference codec stream. Source
    codec tokens never enter this encoder.
    """

    def __init__(
        self,
        hidden_size: int,
        num_memory_tokens: int = 8,
        num_heads: int = 8,
        adapter_dim: int = 256,
        dropout: float = 0.0,
        encoder_type: str = "conformer",
        encoder_layers: int = 2,
        conv_kernel_size: int = 7,
        speaker_embedding_dim: int = 0,
        speaker_conditioning: bool = True,
    ) -> None:
        super().__init__()
        if num_memory_tokens <= 0:
            raise ValueError("num_memory_tokens must be positive")
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        num_heads = _valid_num_heads(adapter_dim, num_heads)
        encoder_type = str(encoder_type).strip().lower()
        if encoder_type not in {"perceiver", "transformer", "conformer", "moe_conformer"}:
            raise ValueError(
                "encoder_type must be one of: perceiver, transformer, conformer, moe_conformer"
            )
        self.hidden_size = int(hidden_size)
        self.adapter_dim = int(adapter_dim)
        self.num_memory_tokens = int(num_memory_tokens)
        self.encoder_type = encoder_type
        self.speaker_embedding_dim = int(speaker_embedding_dim)
        self.speaker_conditioning = bool(speaker_conditioning and self.speaker_embedding_dim > 0)
        self.query = nn.Parameter(torch.empty(num_memory_tokens, adapter_dim))
        position = torch.arange(num_memory_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, adapter_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / adapter_dim)
        )
        query_pos_embedding = torch.zeros(num_memory_tokens, adapter_dim)
        query_pos_embedding[:, 0::2] = torch.sin(position * div_term)
        query_pos_embedding[:, 1::2] = torch.cos(position * div_term[: query_pos_embedding[:, 1::2].shape[1]])
        self.register_buffer("query_pos_embedding", query_pos_embedding)
        self.ref_norm = nn.LayerNorm(hidden_size)
        self.ref_down = nn.Linear(hidden_size, adapter_dim)
        self.ref_encoder: nn.Module | None
        if encoder_type == "perceiver":
            self.ref_encoder = None
        elif encoder_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=adapter_dim,
                nhead=num_heads,
                dim_feedforward=adapter_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.ref_encoder = nn.TransformerEncoder(
                layer,
                num_layers=max(1, int(encoder_layers)),
                norm=nn.LayerNorm(adapter_dim),
            )
        else:
            self.ref_encoder = ConformerEncoder(
                dim=adapter_dim,
                num_layers=max(1, int(encoder_layers)),
                num_heads=num_heads,
                dropout=dropout,
                conv_kernel_size=conv_kernel_size,
                use_moe_ffn=encoder_type == "moe_conformer",
            )
        self.speaker_token = (
            nn.Sequential(
                nn.LayerNorm(self.speaker_embedding_dim),
                nn.Linear(self.speaker_embedding_dim, adapter_dim),
                nn.SiLU(),
                nn.Linear(adapter_dim, adapter_dim),
            )
            if self.speaker_conditioning
            else None
        )
        # Do not normalize the learnable pooling queries: LayerNorm erased the
        # magnitude signal needed to sharpen query/key attention scores.
        self.query_scale = nn.Parameter(torch.ones(1) * 5.0)
        self.pool = nn.MultiheadAttention(
            embed_dim=adapter_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, adapter_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim * 2, adapter_dim),
        )
        self.memory_up = nn.Linear(adapter_dim, hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)
        self._last_attention_entropy_normalized = 0.0
        self._last_attn_max_minus_mean = 0.0
        nn.init.normal_(self.query, mean=0.0, std=0.5)

    def forward(
        self,
        ref_embeddings: torch.Tensor,
        ref_mask: torch.Tensor | None = None,
        speaker_embedding: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
    ) -> TimbreMemoryState:
        if ref_embeddings.dim() != 3:
            raise ValueError(f"ref_embeddings must be [B, T, D], got {tuple(ref_embeddings.shape)}")
        batch_size = ref_embeddings.shape[0]
        ref_embeddings = ref_embeddings.to(dtype=self.ref_norm.weight.dtype)
        ref_embeddings = self.ref_down(self.ref_norm(ref_embeddings))
        if ref_mask is not None:
            if ref_mask.shape != ref_embeddings.shape[:2]:
                raise ValueError(
                    f"ref_mask shape {tuple(ref_mask.shape)} does not match ref embeddings {tuple(ref_embeddings.shape[:2])}"
                )
            ref_mask = ref_mask.bool().to(ref_embeddings.device)
        else:
            ref_mask = torch.ones(ref_embeddings.shape[:2], dtype=torch.bool, device=ref_embeddings.device)
        if self.speaker_token is not None and speaker_embedding is not None:
            if speaker_embedding.dim() != 2 or speaker_embedding.shape[0] != batch_size:
                raise ValueError(
                    f"speaker_embedding must be [B, D_spk], got {tuple(speaker_embedding.shape)}"
                )
            if speaker_embedding.shape[-1] != self.speaker_embedding_dim:
                raise ValueError(
                    f"speaker_embedding dim mismatch: got {speaker_embedding.shape[-1]}, expected {self.speaker_embedding_dim}"
                )
            speaker_token = self.speaker_token(speaker_embedding.to(ref_embeddings.device, dtype=ref_embeddings.dtype))
            if speaker_mask is None:
                speaker_mask = torch.ones(batch_size, dtype=torch.bool, device=ref_embeddings.device)
            else:
                speaker_mask = speaker_mask.bool().to(ref_embeddings.device)
            ref_embeddings = torch.cat([speaker_token.unsqueeze(1), ref_embeddings], dim=1) # E_ref + C_ref
            ref_mask = torch.cat([speaker_mask.unsqueeze(1), ref_mask], dim=1)
        ref_embeddings = _mask_padded(ref_embeddings, ref_mask)
        if self.ref_encoder is not None:
            if self.encoder_type == "transformer":
                ref_embeddings = self.ref_encoder(ref_embeddings, src_key_padding_mask=~ref_mask)
                ref_embeddings = _mask_padded(ref_embeddings, ref_mask)
            else:
                ref_embeddings = self.ref_encoder(ref_embeddings, ref_mask)
        query_with_pos = self.query + self.query_pos_embedding.to(
            device=self.query.device, dtype=self.query.dtype
        )
        query = (query_with_pos * self.query_scale).unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None
        if ref_mask is not None:
            key_padding_mask = ~ref_mask.bool()
        pooled, attention_weights = self.pool(
            query=query,
            key=ref_embeddings,
            value=ref_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        with torch.no_grad():
            probs = attention_weights.float().clamp_min(1.0e-8)
            entropy = -(probs * probs.log()).sum(dim=-1)
            self._last_attention_entropy_normalized = float(
                (entropy / torch.log(torch.tensor(float(ref_embeddings.shape[1]), device=entropy.device))).mean().item()
            )
            max_attn = attention_weights.float().max(dim=-1).values
            mean_attn = attention_weights.float().mean(dim=-1)
            self._last_attn_max_minus_mean = float((max_attn - mean_attn).mean().item())
        memory = self.final_norm(self.memory_up(pooled + self.out(pooled)))
        return TimbreMemoryState(timbre_tokens=memory, timbre_mask=None)


class TargetOnlyTimbreAdapter(nn.Module):
    """Gated residual adapter that injects timbre memory only on target positions."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        adapter_dim: int = 256,
        dropout: float = 0.0,
        init_gate: float = -4.0,
    ) -> None:
        super().__init__()
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        num_heads = _valid_num_heads(adapter_dim, num_heads)
        self.adapter_dim = int(adapter_dim)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.query_down = nn.Linear(hidden_size, adapter_dim)
        self.memory_down = nn.Linear(hidden_size, adapter_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=adapter_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, hidden_size),
        )
        self.gate = nn.Parameter(torch.full((1,), float(init_gate)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        timbre_tokens: torch.Tensor,
        target_position_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, S, D], got {tuple(hidden_states.shape)}")
        if timbre_tokens.dim() != 3:
            raise ValueError(f"timbre_tokens must be [B, K, D], got {tuple(timbre_tokens.shape)}")
        if target_position_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"target_position_mask shape {tuple(target_position_mask.shape)} does not match hidden states {tuple(hidden_states.shape[:2])}"
            )
        if not bool(target_position_mask.any().item()):
            return hidden_states

        hidden_dtype = hidden_states.dtype
        adapter_dtype = self.query_norm.weight.dtype
        adapter_hidden_states = hidden_states.to(dtype=adapter_dtype)
        adapter_timbre_tokens = timbre_tokens.to(device=hidden_states.device, dtype=adapter_dtype)
        attn_out, _ = self.cross_attn(
            query=self.query_down(self.query_norm(adapter_hidden_states)),
            key=self.memory_down(self.memory_norm(adapter_timbre_tokens)),
            value=self.memory_down(self.memory_norm(adapter_timbre_tokens)),
            need_weights=False,
        )
        update = self.residual(attn_out)
        scale = torch.sigmoid(self.gate).to(dtype=hidden_dtype)
        mask = target_position_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        return hidden_states + mask * scale * update.to(dtype=hidden_dtype)
