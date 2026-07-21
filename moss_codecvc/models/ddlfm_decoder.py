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
import torch.nn.functional as F
from torch import nn
from moss_codecvc.models.timbre_memory import ReferenceCodecTimbreMemory


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


def _apply_rotary_position_embedding(
    value: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float,
) -> torch.Tensor:
    """Apply parameter-free RoPE to ``[B,H,L,D]`` attention projections.

    ``positions`` may be the shared ``[L]`` position vector or a per-row
    ``[B,L]`` matrix.  RoPE is evaluated in float32 for numerical stability
    and cast back to the attention dtype.  If the head dimension is odd, the
    largest even prefix is rotated and the final channel is left unchanged.
    """

    if value.ndim != 4:
        raise ValueError(f"RoPE value must be [B,H,L,D], got {tuple(value.shape)}")
    batch, _heads, length, head_dim = value.shape
    rotary_dim = int(head_dim) - int(head_dim) % 2
    if rotary_dim == 0 or length == 0:
        return value
    positions = positions.to(device=value.device)
    if positions.ndim == 1:
        if int(positions.shape[0]) != int(length):
            raise ValueError(
                f"RoPE positions length {int(positions.shape[0])} does not match value length {length}"
            )
        positions = positions.unsqueeze(0).expand(batch, -1)
    elif positions.ndim == 2:
        if tuple(positions.shape) != (batch, length):
            raise ValueError(
                f"RoPE positions shape {tuple(positions.shape)} does not match {(batch, length)}"
            )
    else:
        raise ValueError(f"RoPE positions must be [L] or [B,L], got {tuple(positions.shape)}")

    inv_freq = torch.exp(
        -math.log(float(base))
        * torch.arange(0, rotary_dim, 2, device=value.device, dtype=torch.float32)
        / float(rotary_dim)
    )
    phase = positions.to(dtype=torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    cos = torch.repeat_interleave(phase.cos(), 2, dim=-1).unsqueeze(1).to(dtype=value.dtype)
    sin = torch.repeat_interleave(phase.sin(), 2, dim=-1).unsqueeze(1).to(dtype=value.dtype)

    rotated = value[..., :rotary_dim]
    even, odd = rotated[..., 0::2], rotated[..., 1::2]
    rotate_half = torch.stack((-odd, even), dim=-1).flatten(-2)
    rotated = rotated * cos + rotate_half * sin
    if rotary_dim == head_dim:
        return rotated
    return torch.cat([rotated, value[..., rotary_dim:]], dim=-1)


class DDLFMAdaLNBlock(nn.Module):
    """Self-attention + semantic cross-attention + FFN with two AdaLN paths.

    Batch-47 keeps the historical ``condition`` projection as a compatibility
    surface for the cross branch, and adds independent ``adaln_self`` and
    ``adaln_ffn`` projections.  The latter are the two explicit AdaLN paths
    requested by Fix E.  Their final biases initialize the *actual*
    multiplicative scale at one (the residual ``scale`` parameter is zero) and
    shifts at zero.  Self/FFN residuals are active at unit strength; the
    condition gate only controls how much speaker/time modulation is applied.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        condition_size: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        cross_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        num_heads = int(num_heads)
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope_base = float(rope_base)
        if not math.isfinite(self.rope_base) or self.rope_base <= 1.0:
            raise ValueError("rope_base must be finite and greater than 1")
        self.cross_gate_init = float(cross_gate_init)
        if not math.isfinite(self.cross_gate_init):
            raise ValueError("cross_gate_init must be finite")
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
        # AdaLN-Zero-style start.  Self-attention and FFN begin as identity;
        # the semantic cross branch may use a small explicit nonzero gate.
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        with torch.no_grad():
            self.condition[-1].bias[5 * hidden_size : 6 * hidden_size].fill_(
                self.cross_gate_init
            )

        # Independent AdaLN projections before self-attention and FFN.  The
        # zero residual-scale bias means ``1 + scale == 1`` at initialization;
        # this is the numerically stable interpretation of scale(init)=1.0.
        self.adaln_self = nn.Sequential(
            nn.LayerNorm(int(condition_size)),
            nn.Linear(int(condition_size), hidden_size * 3),
        )
        self.adaln_ffn = nn.Sequential(
            nn.LayerNorm(int(condition_size)),
            nn.Linear(int(condition_size), hidden_size * 3),
        )
        for projection in (self.adaln_self[-1], self.adaln_ffn[-1]):
            nn.init.normal_(projection.weight, mean=0.0, std=0.01)
            nn.init.zeros_(projection.bias)
        # Speaker-only FiLM/AdaLN branch.  It bypasses the time+speaker sum so
        # LayerNorm on the shared condition cannot wash out speaker identity.
        self.speaker_adaln = nn.Sequential(
            nn.LayerNorm(int(condition_size)),
            nn.Linear(int(condition_size), hidden_size * 6),
        )
        nn.init.normal_(self.speaker_adaln[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.speaker_adaln[-1].bias)

    def _cross_attention_with_rope(
        self,
        query: torch.Tensor,
        semantic: torch.Tensor,
        *,
        semantic_mask: torch.Tensor,
        target_positions: torch.Tensor,
        semantic_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run cross-attention with RoPE on projected target Q and semantic K.

        The projections and output layer remain the original
        :class:`~torch.nn.MultiheadAttention` parameters.  This keeps state
        dict compatibility while making the semantic memory explicitly
        order-sensitive.
        """

        batch, target_len, hidden = query.shape
        if semantic.ndim != 3 or int(semantic.shape[0]) != batch or int(semantic.shape[2]) != hidden:
            raise ValueError(
                f"semantic must be [B,S,{hidden}] for cross-attention, got {tuple(semantic.shape)}"
            )
        semantic_len = int(semantic.shape[1])
        if tuple(semantic_mask.shape) != (batch, semantic_len):
            raise ValueError(
                f"semantic_mask shape {tuple(semantic_mask.shape)} does not match semantic"
            )

        q_weight, k_weight, v_weight = self.cross_attn.in_proj_weight.chunk(3, dim=0)
        if self.cross_attn.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.cross_attn.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, q_weight, q_bias)
        k = F.linear(semantic, k_weight, k_bias)
        v = F.linear(semantic, v_weight, v_bias)

        q = q.view(batch, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, semantic_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, semantic_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = _apply_rotary_position_embedding(q, target_positions, base=self.rope_base)
        k = _apply_rotary_position_embedding(k, semantic_positions, base=self.rope_base)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        valid_keys = semantic_mask[:, None, None, :]
        scores = scores.masked_fill(~valid_keys, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention.masked_fill(~valid_keys, 0.0)
        if float(self.cross_attn.dropout) > 0.0:
            attention = F.dropout(
                attention,
                p=float(self.cross_attn.dropout),
                training=self.training,
            )
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch, target_len, hidden)
        return self.cross_attn.out_proj(output)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        *,
        target_mask: torch.Tensor,
        semantic: torch.Tensor,
        semantic_mask: torch.Tensor,
        target_positions: torch.Tensor | None = None,
        semantic_positions: torch.Tensor | None = None,
        condition_gate_scale: float = 1.0,
        speaker_condition: torch.Tensor | None = None,
        speaker_condition_scale: float = 1.0,
    ) -> torch.Tensor:
        gate_scale = float(condition_gate_scale)
        if not math.isfinite(gate_scale) or gate_scale < 0.0:
            raise ValueError("condition_gate_scale must be finite and non-negative")
        params = self.condition(condition).chunk(9, dim=-1)
        shift_self, scale_self, gate_self = params[0:3]
        shift_cross, scale_cross, gate_cross = params[3:6]
        shift_ffn, scale_ffn, gate_ffn = params[6:9]
        self_extra = self.adaln_self(condition).chunk(3, dim=-1)
        ffn_extra = self.adaln_ffn(condition).chunk(3, dim=-1)
        if speaker_condition is None:
            speaker_extra = [torch.zeros_like(self_extra[0]) for _ in range(6)]
        else:
            if speaker_condition.shape != condition.shape:
                raise ValueError("speaker_condition must match condition shape")
            speaker_scale = float(speaker_condition_scale)
            if not math.isfinite(speaker_scale) or speaker_scale < 0.0:
                raise ValueError("speaker_condition_scale must be finite and non-negative")
            speaker_extra = [value * speaker_scale for value in self.speaker_adaln(speaker_condition).chunk(6, dim=-1)]
        # Each extra AdaLN has a shift/scale pair for the attention/FFN input;
        # the second pair is a residual-gate correction.  At gate_scale=0 all
        # conditional effects vanish, which is important for CFG baselines.
        shift_self = gate_scale * (shift_self + self_extra[0] + speaker_extra[0])
        scale_self = gate_scale * (scale_self + self_extra[1] + speaker_extra[1])
        gate_self = gate_scale * (gate_self + self_extra[2] + speaker_extra[2])
        shift_cross = gate_scale * shift_cross
        scale_cross = gate_scale * scale_cross
        gate_cross = gate_scale * gate_cross
        shift_ffn = gate_scale * (shift_ffn + ffn_extra[0] + speaker_extra[3])
        scale_ffn = gate_scale * (scale_ffn + ffn_extra[1] + speaker_extra[4])
        gate_ffn = gate_scale * (gate_ffn + ffn_extra[2] + speaker_extra[5])

        h = _modulate(self.norm_self(x), shift_self, scale_self)
        h = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=~target_mask,
            need_weights=False,
        )[0]
        x = x + (1.0 + gate_self).unsqueeze(1) * h

        h = _modulate(self.norm_cross(x), shift_cross, scale_cross)
        if target_positions is None:
            target_positions = torch.arange(x.shape[1], device=x.device)
        if semantic_positions is None:
            semantic_positions = torch.arange(semantic.shape[1], device=semantic.device)
        h = self._cross_attention_with_rope(
            h,
            semantic,
            semantic_mask=semantic_mask,
            target_positions=target_positions,
            semantic_positions=semantic_positions,
        )
        x = x + gate_cross.unsqueeze(1) * h

        h = _modulate(self.norm_ffn(x), shift_ffn, scale_ffn)
        x = x + (1.0 + gate_ffn).unsqueeze(1) * self.ffn(h)
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
        rope_base: float = 10000.0,
        cross_gate_init: float = 0.0,
        num_timbre_tokens: int = 32,
        speaker_condition_scale: float = 4.0,
        speaker_input_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.semantic_dim = int(semantic_dim)
        self.speaker_dim = int(speaker_dim)
        self.hidden_size = int(hidden_size)
        self.num_modalities = int(num_modalities)
        self.num_timbre_tokens = int(num_timbre_tokens)
        self.speaker_condition_scale = float(speaker_condition_scale)
        self.speaker_input_scale = float(speaker_input_scale)
        if self.num_modalities < 1:
            raise ValueError("num_modalities must be positive")
        if not math.isfinite(self.speaker_condition_scale) or self.speaker_condition_scale < 0.0:
            raise ValueError("speaker_condition_scale must be finite and non-negative")
        if not math.isfinite(self.speaker_input_scale) or self.speaker_input_scale < 0.0:
            raise ValueError("speaker_input_scale must be finite and non-negative")

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
        # Make zero-speaker CFG truly unconditional: all affine biases on the
        # speaker path start at zero, so a zero embedding produces zero.
        for module in self.speaker_proj:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        self.time_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.speaker_input_proj = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.normal_(self.speaker_input_proj.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.speaker_input_proj.bias)
        self.modality_embedding = nn.Embedding(self.num_modalities, self.semantic_dim)
        self.timbre_memory = ReferenceCodecTimbreMemory(
            hidden_size=self.hidden_size,
            num_memory_tokens=self.num_timbre_tokens,
            num_heads=8,
            adapter_dim=256,
            dropout=0.0,
            encoder_type="conformer",
            encoder_layers=2,
            conv_kernel_size=7,
            speaker_embedding_dim=self.speaker_dim,
            speaker_conditioning=True,
        )
        self.layers = nn.ModuleList(
            [
                DDLFMAdaLNBlock(
                    self.hidden_size,
                    int(num_heads),
                    int(ffn_size),
                    self.hidden_size,
                    dropout=float(dropout),
                    rope_base=float(rope_base),
                    cross_gate_init=float(cross_gate_init),
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

    @torch.no_grad()
    def conditioning_diagnostics(
        self,
        t: torch.Tensor,
        speaker: torch.Tensor,
        gate_scale: float = 1.0,
    ) -> dict[str, object]:
        """Return JSON-safe AdaLN modulation statistics without side effects.

        ``scale`` is the raw AdaLN output used by :func:`_modulate`; the
        actual multiplicative factor is explicitly reported as ``1 + scale``.
        Cross-gate statistics include ``gate_scale`` because that is the
        effective semantic/speaker residual multiplier used by
        :meth:`forward` during conditioning warmup.  Self/FFN gates are not
        scaled by this control.
        """

        if t.ndim != 1:
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        batch = int(t.shape[0])
        if speaker.ndim != 2 or tuple(speaker.shape) != (batch, self.speaker_dim):
            raise ValueError(f"speaker must be [B,{self.speaker_dim}], got {tuple(speaker.shape)}")
        gate_scale = float(gate_scale)
        if not math.isfinite(gate_scale) or gate_scale < 0.0:
            raise ValueError("gate_scale must be finite and non-negative")

        device = self.input_proj.weight.device
        dtype = self.input_proj.weight.dtype
        position = _sinusoidal_time_embedding(t.to(device=device), self.hidden_size).to(dtype=dtype)
        time_condition = self.time_proj(position)
        speaker_condition = self.speaker_proj(speaker.to(device=device, dtype=dtype))
        condition = time_condition + speaker_condition
        layer_rows: list[dict[str, object]] = []
        for layer_index, layer in enumerate(self.layers):
            params = layer.condition(condition).chunk(9, dim=-1)
            self_extra = layer.adaln_self(condition).chunk(3, dim=-1)
            ffn_extra = layer.adaln_ffn(condition).chunk(3, dim=-1)
            params = list(params)
            params[0] = params[0] + float(gate_scale) * self_extra[0]
            params[1] = params[1] + float(gate_scale) * self_extra[1]
            params[2] = params[2] + float(gate_scale) * self_extra[2]
            params[6] = params[6] + float(gate_scale) * ffn_extra[0]
            params[7] = params[7] + float(gate_scale) * ffn_extra[1]
            params[8] = params[8] + float(gate_scale) * ffn_extra[2]
            row: dict[str, object] = {"layer": int(layer_index)}
            for branch_index, branch_name in enumerate(("self", "cross", "ffn")):
                shift, scale, gate = params[branch_index * 3 : branch_index * 3 + 3]
                if branch_name == "cross":
                    gate = gate * gate_scale
                multiplier = 1.0 + scale
                row[branch_name] = {
                    "shift_mean_abs": float(shift.float().abs().mean().item()),
                    "shift_max_abs": float(shift.float().abs().max().item()),
                    "shift_p95_abs": float(torch.quantile(shift.float().abs(), 0.95).item()),
                    "scale_mean_abs": float(scale.float().abs().mean().item()),
                    "scale_max_abs": float(scale.float().abs().max().item()),
                    "multiplicative_scale_formula": "1 + scale",
                    "multiplicative_scale_mean": float(multiplier.float().mean().item()),
                    "multiplicative_scale_min": float(multiplier.float().min().item()),
                    "multiplicative_scale_max": float(multiplier.float().max().item()),
                    "multiplicative_scale_p01": float(torch.quantile(multiplier.float(), 0.01).item()),
                    "multiplicative_scale_p99": float(torch.quantile(multiplier.float(), 0.99).item()),
                    "multiplicative_scale_negative_fraction": float(
                        (multiplier.float() < 0.0).float().mean().item()
                    ),
                    "gate_mean_abs": float(gate.float().abs().mean().item()),
                    "gate_max_abs": float(gate.float().abs().max().item()),
                    "gate_p95_abs": float(torch.quantile(gate.float().abs(), 0.95).item()),
                }
            layer_rows.append(row)
        return {
            "gate_scale": gate_scale,
            "gate_scale_applies_to": "cross",
            "modulate_multiplicative_scale_formula": "1 + scale",
            "time_condition_norm_mean": float(
                time_condition.float().norm(dim=-1).mean().item()
            ),
            "speaker_condition_norm_mean": float(
                speaker_condition.float().norm(dim=-1).mean().item()
            ),
            "speaker_to_time_condition_norm_ratio": float(
                (
                    speaker_condition.float().norm(dim=-1)
                    / time_condition.float().norm(dim=-1).clamp_min(1.0e-8)
                ).mean().item()
            ),
            "layers": layer_rows,
        }

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        semantic: torch.Tensor,
        speaker: torch.Tensor,
        *,
        prompt_zq: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        semantic_mask: torch.Tensor | None = None,
        semantic_modality: torch.Tensor | None = None,
        condition_gate_scale: float = 1.0,
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
        if prompt_zq is None:
            prompt_zq = x_t.new_zeros((batch, 1, self.latent_dim))
            prompt_mask = torch.zeros((batch, 1), dtype=torch.bool, device=x_t.device)
        if prompt_mask is None:
            prompt_mask = torch.ones(prompt_zq.shape[:2], dtype=torch.bool, device=x_t.device)
        timbre_state = self.timbre_memory(
            prompt_zq.to(dtype=x_t.dtype),
            prompt_mask,
            speaker_embedding=speaker,
        )
        timbre_tokens = timbre_state.timbre_tokens.to(dtype=x_t.dtype)
        timbre_mask = torch.ones(
            batch, timbre_tokens.shape[1], dtype=torch.bool, device=x_t.device
        )
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
        time_condition = self.time_proj(pos)
        speaker_condition = self.speaker_proj(speaker.to(dtype=x_t.dtype))
        condition = time_condition + speaker_condition
        x = self.input_proj(x_t)
        x = x + float(condition_gate_scale) * self.speaker_input_scale * self.speaker_input_proj(
            speaker_condition
        ).unsqueeze(1)
        # Dynamic sinusoidal frame positions avoid a fixed maximum sequence length.
        frame_pos = _sinusoidal_time_embedding(
            torch.arange(target_len, device=x.device, dtype=torch.float32),
            self.hidden_size,
        ).to(dtype=x.dtype)
        x = (x + frame_pos.unsqueeze(0)).masked_fill(~target_mask.unsqueeze(-1), 0.0)

        semantic = semantic.to(dtype=x.dtype)
        semantic = semantic + self.modality_embedding(semantic_modality).unsqueeze(1).to(dtype=x.dtype)
        semantic = self.semantic_proj(semantic).masked_fill(~semantic_mask.unsqueeze(-1), 0.0)
        semantic = torch.cat([timbre_tokens, semantic], dim=1)
        semantic_mask = torch.cat([timbre_mask, semantic_mask], dim=1)
        target_positions = torch.arange(target_len, device=x.device)
        semantic_positions = torch.arange(semantic.shape[1], device=x.device)
        for layer in self.layers:
            x = layer(
                x,
                condition,
                target_mask=target_mask,
                semantic=semantic,
                semantic_mask=semantic_mask,
                target_positions=target_positions,
                semantic_positions=semantic_positions,
                condition_gate_scale=condition_gate_scale,
                speaker_condition=speaker_condition,
                speaker_condition_scale=self.speaker_condition_scale,
            )
        velocity = self.output_proj(self.output_norm(x)).masked_fill(~target_mask.unsqueeze(-1), 0.0)
        return DDLFMOutput(velocity=velocity)


__all__ = ["DDLFMDecoder", "DDLFMAdaLNBlock", "DDLFMOutput"]
