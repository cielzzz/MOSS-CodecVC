from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SemanticProgressState:
    loss: torch.Tensor | None
    stats: dict[str, float]


def _pearson_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    if pred.numel() < 2:
        return None
    pred = pred.float()
    target = target.float()
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm()
    if float(denom.detach().item()) <= 1.0e-8:
        return None
    return (pred * target).sum() / denom.clamp_min(1.0e-8)


def compute_semantic_attention_progress(
    attention_weights: list[torch.Tensor],
    target_mask: torch.Tensor,
    source_mask: torch.Tensor | None = None,
) -> SemanticProgressState:
    """Compute linear target-to-source progress from semantic cross-attention.

    Args:
        attention_weights: list of [B, heads, T_target_seq, S_source] tensors.
        target_mask: [B, T_target_seq] positions where target codec is generated.
        source_mask: optional [B, S_source] valid source semantic frames.
    """

    valid_attn = [attn for attn in attention_weights if torch.is_tensor(attn) and attn.dim() == 4]
    if not valid_attn:
        return SemanticProgressState(loss=None, stats={})
    target_mask = target_mask.to(device=valid_attn[0].device).bool()
    source_mask = None if source_mask is None else source_mask.to(device=valid_attn[0].device).bool()
    losses: list[torch.Tensor] = []
    corrs: list[torch.Tensor] = []
    revisit_terms: list[torch.Tensor] = []
    sample_count = 0
    for attn in valid_attn:
        probs = attn.float().mean(dim=1).clamp_min(1.0e-9)  # [B, T, S]
        if probs.shape[:2] != target_mask.shape:
            cur_target_mask = target_mask[:, -probs.shape[1] :]
        else:
            cur_target_mask = target_mask
        cur_source_mask = source_mask
        if cur_source_mask is not None and cur_source_mask.shape[1] != probs.shape[-1]:
            cur_source_mask = cur_source_mask[:, : probs.shape[-1]]
            if cur_source_mask.shape[1] < probs.shape[-1]:
                cur_source_mask = F.pad(cur_source_mask, (0, probs.shape[-1] - cur_source_mask.shape[1]), value=False)
        if cur_source_mask is not None:
            probs = probs.masked_fill(~cur_source_mask[:, None, :], 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        src_pos = torch.linspace(0.0, 1.0, probs.shape[-1], device=probs.device, dtype=probs.dtype)
        pred_progress = (probs * src_pos.view(1, 1, -1)).sum(dim=-1)
        for batch_idx in range(probs.shape[0]):
            idx = cur_target_mask[batch_idx].nonzero(as_tuple=False).flatten()
            if idx.numel() < 2:
                continue
            pred = pred_progress[batch_idx, idx]
            gt = torch.linspace(0.0, 1.0, idx.numel(), device=pred.device, dtype=pred.dtype)
            losses.append(F.smooth_l1_loss(pred, gt, reduction="mean"))
            corr = _pearson_corr(pred.detach(), gt.detach())
            if corr is not None:
                corrs.append(corr)
            deltas = pred[1:] - pred[:-1]
            revisit_terms.append(F.relu(-deltas).mean())
            sample_count += 1
    if not losses:
        return SemanticProgressState(loss=None, stats={})
    loss = torch.stack(losses).mean()
    stats = {
        "source_semantic_progress_loss": float(loss.detach().item()),
        "source_semantic_progress_samples": float(sample_count),
    }
    if corrs:
        stats["source_semantic_progress_corr"] = float(torch.stack(corrs).mean().detach().item())
    if revisit_terms:
        stats["source_semantic_revisit_score"] = float(torch.stack(revisit_terms).mean().detach().item())
    return SemanticProgressState(loss=loss, stats=stats)
