from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from moss_codecvc.roles import NUM_ROLES, REF_CODEC, SOURCE_CODEC, TARGET_CODEC, TEXT_OR_OTHER

from .timbre_memory import ReferenceCodecTimbreMemory


def _valid_num_heads(adapter_dim: int, num_heads: int) -> int:
    num_heads = max(1, int(num_heads))
    if adapter_dim % num_heads != 0:
        return 1
    return num_heads


def _logit_from_prob(value: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    value = value.clamp(min=eps, max=1.0 - eps)
    return torch.log(value / (1.0 - value))


def role_gate_prior(n_vq: int) -> torch.Tensor:
    """Soft initialization prior for role-aware VQ routing gates."""

    prior = torch.full((NUM_ROLES, int(n_vq)), 0.98, dtype=torch.float32)
    prior[TEXT_OR_OTHER].fill_(0.995)
    prior[TARGET_CODEC].fill_(0.98)
    for idx in range(int(n_vq)):
        if idx < max(1, int(n_vq) // 2):
            prior[SOURCE_CODEC, idx] = 0.85
        else:
            prior[SOURCE_CODEC, idx] = 0.60
        if idx < max(1, int(n_vq) // 4):
            prior[REF_CODEC, idx] = 0.65
        else:
            prior[REF_CODEC, idx] = 0.90
    return prior


def target_head_gate_prior(n_vq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial source-prosody/ref-timbre preferences for target RVQ heads."""

    n_vq = int(n_vq)
    prosody = torch.empty(n_vq, dtype=torch.float32)
    timbre = torch.empty(n_vq, dtype=torch.float32)
    for idx in range(n_vq):
        if idx < max(1, n_vq // 2):
            prosody[idx] = 0.65
        else:
            prosody[idx] = 0.45
        if idx < max(1, n_vq // 4):
            timbre[idx] = 0.45
        else:
            timbre[idx] = 0.65
    return prosody, timbre


@dataclass
class RoutingLossState:
    loss: torch.Tensor
    stats: dict[str, float]


class RoleCodecRouter(nn.Module):
    """Applies learnable per-role, per-codebook gates to MOSS audio embeddings."""

    def __init__(self, n_vq: int, *, num_roles: int = NUM_ROLES) -> None:
        super().__init__()
        if int(num_roles) != NUM_ROLES:
            raise ValueError(f"num_roles must be {NUM_ROLES}, got {num_roles}")
        self.n_vq = int(n_vq)
        prior = role_gate_prior(self.n_vq)
        self.register_buffer("gate_prior", prior, persistent=True)
        self.gate_logits = nn.Parameter(_logit_from_prob(prior))

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def _validate_role_ids(self, input_ids: torch.Tensor, role_ids: torch.Tensor | None) -> torch.Tensor | None:
        if role_ids is None:
            return None
        if role_ids.shape != input_ids.shape[:2]:
            raise ValueError(f"role_ids shape {tuple(role_ids.shape)} does not match input_ids {tuple(input_ids.shape[:2])}")
        return role_ids.to(device=input_ids.device, dtype=torch.long).clamp(min=0, max=NUM_ROLES - 1)

    def compute_audio_embeddings(
        self,
        base_model: nn.Module,
        input_ids: torch.LongTensor,
        role_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        role_ids = self._validate_role_ids(input_ids, role_ids)
        embeds = None
        gates = self.gates()
        if role_ids is not None:
            gates = gates.to(device=input_ids.device)
            position_gates = gates.index_select(0, role_ids.reshape(-1)).view(*role_ids.shape, self.n_vq)
        else:
            position_gates = None

        for idx, embed_layer in enumerate(base_model.emb_ext):
            cur = embed_layer(input_ids[..., idx + 1])
            if position_gates is not None:
                cur = cur * position_gates[..., idx].to(dtype=cur.dtype).unsqueeze(-1)
            embeds = cur if embeds is None else embeds + cur
        if embeds is None:
            raise RuntimeError("No audio embedding layers found on base model.")
        return embeds

    def compute_input_embeddings(
        self,
        base_model: nn.Module,
        input_ids: torch.LongTensor,
        role_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        text_embeds = base_model.get_input_embeddings()(input_ids[..., 0])
        return text_embeds + self.compute_audio_embeddings(base_model, input_ids, role_ids)

    def regularization_loss(self) -> RoutingLossState:
        gates = self.gates()
        prior = self.gate_prior.to(device=gates.device, dtype=gates.dtype)
        prior_loss = F.mse_loss(gates, prior)
        source_ref_gap = (gates[SOURCE_CODEC] - gates[REF_CODEC]).abs().mean()
        no_collapse = F.relu(0.05 - gates).mean() + F.relu(gates - 0.995).mean()
        loss = prior_loss + 0.01 * no_collapse
        stats = {
            "role_route_loss": float(loss.detach().item()),
            "role_gate_mean": float(gates.detach().mean().item()),
            "source_gate_mean": float(gates[SOURCE_CODEC].detach().mean().item()),
            "ref_gate_mean": float(gates[REF_CODEC].detach().mean().item()),
            "target_gate_mean": float(gates[TARGET_CODEC].detach().mean().item()),
            "source_ref_gate_gap": float(source_ref_gap.detach().item()),
        }
        return RoutingLossState(loss=loss, stats=stats)


class SourceProsodyEncoder(nn.Module):
    """Compresses routed source codec embeddings into compact prosody memory."""

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
    ) -> None:
        super().__init__()
        self.memory = ReferenceCodecTimbreMemory(
            hidden_size=hidden_size,
            num_memory_tokens=num_memory_tokens,
            num_heads=num_heads,
            adapter_dim=adapter_dim,
            dropout=dropout,
            encoder_type=encoder_type,
            encoder_layers=encoder_layers,
            conv_kernel_size=conv_kernel_size,
            speaker_embedding_dim=0,
            speaker_conditioning=False,
        )

    def forward(self, source_embeddings: torch.Tensor, source_mask: torch.Tensor | None) -> torch.Tensor:
        if source_embeddings.dim() != 3:
            raise ValueError(f"source_embeddings must be [B, T, D], got {tuple(source_embeddings.shape)}")
        if source_mask is None:
            source_mask = torch.ones(source_embeddings.shape[:2], dtype=torch.bool, device=source_embeddings.device)
        else:
            source_mask = source_mask.to(device=source_embeddings.device).bool()
        if source_mask.shape != source_embeddings.shape[:2]:
            raise ValueError(
                f"source_mask shape {tuple(source_mask.shape)} does not match source embeddings {tuple(source_embeddings.shape[:2])}"
            )

        empty_rows = ~source_mask.any(dim=1)
        if bool(empty_rows.any().item()):
            source_embeddings = source_embeddings.clone()
            source_mask = source_mask.clone()
            source_embeddings[empty_rows, 0] = 0
            source_mask[empty_rows, 0] = True
        return self.memory(source_embeddings, source_mask).timbre_tokens


class MemoryDeltaAdapter(nn.Module):
    """Cross-attention adapter that returns a residual delta without applying it."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        adapter_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        num_heads = _valid_num_heads(adapter_dim, num_heads)
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

    def forward(self, hidden_states: torch.Tensor, memory_tokens: torch.Tensor | None) -> torch.Tensor:
        if memory_tokens is None:
            return torch.zeros_like(hidden_states)
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, T, D], got {tuple(hidden_states.shape)}")
        if memory_tokens.dim() != 3:
            raise ValueError(f"memory_tokens must be [B, K, D], got {tuple(memory_tokens.shape)}")
        hidden_dtype = hidden_states.dtype
        adapter_dtype = self.query_norm.weight.dtype
        query_states = hidden_states.to(dtype=adapter_dtype)
        memory_states = memory_tokens.to(device=hidden_states.device, dtype=adapter_dtype)
        attn_out, _ = self.cross_attn(
            query=self.query_down(self.query_norm(query_states)),
            key=self.memory_down(self.memory_norm(memory_states)),
            value=self.memory_down(self.memory_norm(memory_states)),
            need_weights=False,
        )
        return self.residual(attn_out).to(dtype=hidden_dtype)


class PerCodebookTargetHeadRouter(nn.Module):
    """Applies source/ref memory deltas with learnable per-audio-head gates."""

    def __init__(
        self,
        hidden_size: int,
        n_vq: int,
        *,
        num_heads: int = 8,
        adapter_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_vq = int(n_vq)
        self.prosody_adapter = MemoryDeltaAdapter(
            hidden_size=hidden_size,
            num_heads=num_heads,
            adapter_dim=adapter_dim,
            dropout=dropout,
        )
        self.timbre_adapter = MemoryDeltaAdapter(
            hidden_size=hidden_size,
            num_heads=num_heads,
            adapter_dim=adapter_dim,
            dropout=dropout,
        )
        prosody_prior, timbre_prior = target_head_gate_prior(self.n_vq)
        self.register_buffer("prosody_gate_prior", prosody_prior, persistent=True)
        self.register_buffer("timbre_gate_prior", timbre_prior, persistent=True)
        self.prosody_gate_logits = nn.Parameter(_logit_from_prob(prosody_prior))
        self.timbre_gate_logits = nn.Parameter(_logit_from_prob(timbre_prior))

    def prosody_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.prosody_gate_logits)

    def timbre_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.timbre_gate_logits)

    def routed_logits(
        self,
        *,
        hidden_states_for_heads: list[torch.Tensor],
        lm_heads: nn.ModuleList,
        target_position_mask: torch.Tensor | None,
        prosody_tokens: torch.Tensor | None,
        timbre_tokens: torch.Tensor | None,
        prosody_batch_gate: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if len(hidden_states_for_heads) != self.n_vq + 1:
            raise ValueError(
                f"hidden_states_for_heads must contain {self.n_vq + 1} tensors, got {len(hidden_states_for_heads)}"
            )
        query_states = hidden_states_for_heads[0]
        if target_position_mask is None:
            mask = torch.ones(query_states.shape[:2], dtype=query_states.dtype, device=query_states.device)
        else:
            if target_position_mask.shape[1] != query_states.shape[1]:
                target_position_mask = target_position_mask[:, -query_states.shape[1] :]
            mask = target_position_mask.to(device=query_states.device, dtype=query_states.dtype)
        mask = mask.unsqueeze(-1)
        delta_prosody = self.prosody_adapter(query_states, prosody_tokens)
        if prosody_batch_gate is not None:
            gate = prosody_batch_gate.to(device=query_states.device, dtype=query_states.dtype).view(-1, 1, 1)
            if gate.shape[0] != delta_prosody.shape[0]:
                raise ValueError(
                    f"prosody_batch_gate batch={gate.shape[0]} does not match hidden batch={delta_prosody.shape[0]}"
                )
            delta_prosody = delta_prosody * gate
        delta_timbre = self.timbre_adapter(query_states, timbre_tokens)
        prosody_gates = self.prosody_gates().to(device=query_states.device, dtype=query_states.dtype)
        timbre_gates = self.timbre_gates().to(device=query_states.device, dtype=query_states.dtype)

        logits = [lm_heads[0](hidden_states_for_heads[0])]
        for idx in range(self.n_vq):
            head_hidden = hidden_states_for_heads[idx + 1]
            routed_hidden = head_hidden + mask * (
                prosody_gates[idx] * delta_prosody + timbre_gates[idx] * delta_timbre
            )
            cur_logits = lm_heads[idx + 1](routed_hidden)
            cur_logits[..., -1] = float("-inf")
            logits.append(cur_logits)
        return logits

    def regularization_loss(self) -> RoutingLossState:
        prosody_gates = self.prosody_gates()
        timbre_gates = self.timbre_gates()
        prosody_prior = self.prosody_gate_prior.to(device=prosody_gates.device, dtype=prosody_gates.dtype)
        timbre_prior = self.timbre_gate_prior.to(device=timbre_gates.device, dtype=timbre_gates.dtype)
        prior_loss = F.mse_loss(prosody_gates, prosody_prior) + F.mse_loss(timbre_gates, timbre_prior)
        no_collapse = (
            F.relu(0.05 - prosody_gates).mean()
            + F.relu(prosody_gates - 0.995).mean()
            + F.relu(0.05 - timbre_gates).mean()
            + F.relu(timbre_gates - 0.995).mean()
        )
        loss = prior_loss + 0.01 * no_collapse
        stats = {
            "target_head_route_loss": float(loss.detach().item()),
            "prosody_head_gate_mean": float(prosody_gates.detach().mean().item()),
            "timbre_head_gate_mean": float(timbre_gates.detach().mean().item()),
            "head_gate_gap": float((prosody_gates - timbre_gates).abs().detach().mean().item()),
        }
        return RoutingLossState(loss=loss, stats=stats)
