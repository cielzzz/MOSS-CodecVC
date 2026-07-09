from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class AuxiliaryLossState:
    loss: torch.Tensor | None
    stats: dict[str, float]


def _valid_mask(mask: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(values.shape, dtype=torch.bool, device=values.device)
    if mask.shape != values.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match values {tuple(values.shape)}")
    return mask.to(device=values.device).bool()


def _resample_1d(values: torch.Tensor, target_len: int, *, binary: bool = False) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"values must be [B, T], got {tuple(values.shape)}")
    if values.shape[1] == target_len:
        return values
    if values.shape[1] <= 0:
        return values.new_zeros((values.shape[0], target_len))
    mode = "nearest" if binary else "linear"
    kwargs = {} if binary else {"align_corners": False}
    out = F.interpolate(values.float().unsqueeze(1), size=int(target_len), mode=mode, **kwargs).squeeze(1)
    if binary:
        out = (out >= 0.5).to(dtype=values.dtype)
    return out.to(device=values.device)


def _masked_standardize(values: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (values * mask_f).sum(dim=1, keepdim=True) / denom
    var = (((values - mean) * mask_f) ** 2).sum(dim=1, keepdim=True) / denom
    return (values - mean) / torch.sqrt(var + eps)


def _masked_mean_loss(loss_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    if loss_values.shape != mask.shape:
        raise ValueError(f"loss shape {tuple(loss_values.shape)} does not match mask {tuple(mask.shape)}")
    mask_f = mask.to(device=loss_values.device, dtype=loss_values.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    if not bool(mask.any().item()):
        return None
    return (loss_values * mask_f).sum() / denom


class ProsodyHead(nn.Module):
    """Predicts fixed prosody features from teacher-forced target hidden states."""

    def __init__(self, hidden_size: int, adapter_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        self.frame = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.logf0 = nn.Linear(adapter_dim, 1)
        self.voiced = nn.Linear(adapter_dim, 1)
        self.energy = nn.Linear(adapter_dim, 1)
        self.pause = nn.Linear(adapter_dim, 1)
        self.duration = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, 1),
        )

    def forward(self, target_hidden: torch.Tensor, target_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if target_hidden.dim() != 3:
            raise ValueError(f"target_hidden must be [B, T, D], got {tuple(target_hidden.shape)}")
        hidden_dtype = target_hidden.dtype
        head_dtype = self.frame[0].weight.dtype
        hidden = target_hidden.to(dtype=head_dtype)
        frame = self.frame(hidden)
        if target_mask is None:
            mask = torch.ones(target_hidden.shape[:2], dtype=target_hidden.dtype, device=target_hidden.device)
        else:
            mask = target_mask.to(device=target_hidden.device, dtype=target_hidden.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (hidden * mask.unsqueeze(-1).to(dtype=head_dtype)).sum(dim=1) / denom.to(dtype=head_dtype)
        return {
            "logf0": self.logf0(frame).squeeze(-1).to(dtype=hidden_dtype),
            "voiced_logit": self.voiced(frame).squeeze(-1).to(dtype=hidden_dtype),
            "energy": self.energy(frame).squeeze(-1).to(dtype=hidden_dtype),
            "pause_logit": self.pause(frame).squeeze(-1).to(dtype=hidden_dtype),
            "log_duration": self.duration(pooled).squeeze(-1).to(dtype=hidden_dtype),
        }


class ContentEmbeddingHead(nn.Module):
    """Predicts a fixed utterance-level content embedding from target hidden states."""

    def __init__(self, hidden_size: int, embedding_dim: int, adapter_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        if int(embedding_dim) <= 0:
            raise ValueError("embedding_dim must be positive")
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, int(embedding_dim)),
        )

    def forward(self, target_hidden: torch.Tensor, target_mask: torch.Tensor | None = None) -> torch.Tensor:
        if target_hidden.dim() != 3:
            raise ValueError(f"target_hidden must be [B, T, D], got {tuple(target_hidden.shape)}")
        head_dtype = self.net[0].weight.dtype
        hidden = target_hidden.to(dtype=head_dtype)
        if target_mask is None:
            mask = torch.ones(target_hidden.shape[:2], dtype=target_hidden.dtype, device=target_hidden.device)
        else:
            mask = target_mask.to(device=target_hidden.device, dtype=target_hidden.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (hidden * mask.unsqueeze(-1).to(dtype=head_dtype)).sum(dim=1) / denom.to(dtype=head_dtype)
        return F.normalize(self.net(pooled).float(), dim=-1)


class ContentTokenHead(nn.Module):
    """Predicts frame-level semantic/content token IDs from target hidden states."""

    def __init__(self, hidden_size: int, vocab_size: int, adapter_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        if int(vocab_size) <= 0:
            raise ValueError("vocab_size must be positive")
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        self.vocab_size = int(vocab_size)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, self.vocab_size),
        )

    def forward(self, target_hidden: torch.Tensor) -> torch.Tensor:
        if target_hidden.dim() != 3:
            raise ValueError(f"target_hidden must be [B, T, D], got {tuple(target_hidden.shape)}")
        hidden_dtype = target_hidden.dtype
        head_dtype = self.net[0].weight.dtype
        return self.net(target_hidden.to(dtype=head_dtype)).to(dtype=hidden_dtype)


class ContentCTCHead(nn.Module):
    """Predicts text/content-token logits for CTC sequence supervision."""

    def __init__(self, hidden_size: int, vocab_size: int, adapter_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        if int(vocab_size) <= 1:
            raise ValueError("CTC vocab_size must be greater than 1")
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        self.vocab_size = int(vocab_size)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, self.vocab_size),
        )

    def forward(self, target_hidden: torch.Tensor) -> torch.Tensor:
        if target_hidden.dim() != 3:
            raise ValueError(f"target_hidden must be [B, T, D], got {tuple(target_hidden.shape)}")
        hidden_dtype = target_hidden.dtype
        head_dtype = self.net[0].weight.dtype
        return self.net(target_hidden.to(dtype=head_dtype)).to(dtype=hidden_dtype)


class SemanticFeatureHead(nn.Module):
    """Predicts continuous semantic features from target hidden states."""

    def __init__(self, hidden_size: int, feature_dim: int, adapter_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, self.feature_dim),
        )

    def forward(self, target_hidden: torch.Tensor) -> torch.Tensor:
        if target_hidden.dim() != 3:
            raise ValueError(f"target_hidden must be [B, T, D], got {tuple(target_hidden.shape)}")
        hidden_dtype = target_hidden.dtype
        head_dtype = self.net[0].weight.dtype
        return self.net(target_hidden.to(dtype=head_dtype)).to(dtype=hidden_dtype)


class ProgressStopHead(nn.Module):
    """Predicts monotonic progress bins and utterance stop probability."""

    def __init__(self, hidden_size: int, num_bins: int = 32, adapter_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        if int(num_bins) <= 1:
            raise ValueError("num_bins must be greater than 1")
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        self.num_bins = int(num_bins)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.progress = nn.Linear(adapter_dim, self.num_bins)
        self.stop = nn.Linear(adapter_dim, 1)

    def forward(self, target_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if target_hidden.dim() != 3:
            raise ValueError(f"target_hidden must be [B, T, D], got {tuple(target_hidden.shape)}")
        hidden_dtype = target_hidden.dtype
        head_dtype = self.net[0].weight.dtype
        features = self.net(target_hidden.to(dtype=head_dtype))
        return {
            "progress_logits": self.progress(features).to(dtype=hidden_dtype),
            "stop_logit": self.stop(features).squeeze(-1).to(dtype=hidden_dtype),
        }


class SourceCodecContentHead(nn.Module):
    """Predicts selected source codec codebooks from target hidden states.

    This is a lightweight no-extra-data content proxy. It should be used with
    small weights because source codec tokens also carry speaker/timbre cues.
    """

    def __init__(
        self,
        hidden_size: int,
        codebooks: list[int],
        audio_vocab_size: int,
        adapter_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not codebooks:
            raise ValueError("codebooks must be non-empty")
        if int(audio_vocab_size) <= 0:
            raise ValueError("audio_vocab_size must be positive")
        adapter_dim = max(1, min(int(adapter_dim), int(hidden_size)))
        self.codebooks = [int(item) for item in codebooks]
        self.audio_vocab_size = int(audio_vocab_size)
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(adapter_dim, self.audio_vocab_size) for _ in self.codebooks])

    def forward(self, target_hidden: torch.Tensor) -> list[torch.Tensor]:
        if target_hidden.dim() != 3:
            raise ValueError(f"target_hidden must be [B, T, D], got {tuple(target_hidden.shape)}")
        hidden_dtype = target_hidden.dtype
        head_dtype = self.shared[0].weight.dtype
        features = self.shared(target_hidden.to(dtype=head_dtype))
        return [head(features).to(dtype=hidden_dtype) for head in self.heads]


def compute_prosody_proxy_loss(
    predictions: dict[str, torch.Tensor],
    *,
    source_logf0: torch.Tensor | None = None,
    source_logf0_mask: torch.Tensor | None = None,
    source_voiced_mask: torch.Tensor | None = None,
    source_energy: torch.Tensor | None = None,
    source_energy_mask: torch.Tensor | None = None,
    source_pause_mask: torch.Tensor | None = None,
    source_duration: torch.Tensor | None = None,
    source_duration_mask: torch.Tensor | None = None,
    f0_weight: float = 1.0,
    voiced_weight: float = 0.5,
    energy_weight: float = 0.5,
    pause_weight: float = 1.0,
    duration_weight: float = 0.5,
    normalize_f0: bool = True,
    normalize_energy: bool = True,
) -> AuxiliaryLossState:
    target_len = int(predictions["logf0"].shape[1])
    terms: list[torch.Tensor] = []
    stats: dict[str, float] = {}

    if f0_weight > 0 and source_logf0 is not None:
        gt = _resample_1d(source_logf0.to(device=predictions["logf0"].device).float(), target_len)
        valid = _resample_1d(_valid_mask(source_logf0_mask, source_logf0).float().to(gt.device), target_len, binary=True).bool()
        if source_voiced_mask is not None:
            voiced = _resample_1d(source_voiced_mask.to(device=gt.device).float(), target_len, binary=True).bool()
            valid = valid & voiced
        pred = predictions["logf0"].float()
        if normalize_f0:
            pred = _masked_standardize(pred, valid)
            gt = _masked_standardize(gt, valid)
        term = _masked_mean_loss(F.smooth_l1_loss(pred, gt, reduction="none"), valid)
        if term is not None:
            terms.append(float(f0_weight) * term)
            stats["prosody_f0_loss"] = float(term.detach().item())

    if voiced_weight > 0 and source_voiced_mask is not None:
        gt = _resample_1d(source_voiced_mask.to(device=predictions["voiced_logit"].device).float(), target_len, binary=True)
        valid = torch.ones_like(gt, dtype=torch.bool)
        term = _masked_mean_loss(
            F.binary_cross_entropy_with_logits(predictions["voiced_logit"].float(), gt.float(), reduction="none"),
            valid,
        )
        if term is not None:
            terms.append(float(voiced_weight) * term)
            stats["prosody_voiced_loss"] = float(term.detach().item())

    if energy_weight > 0 and source_energy is not None:
        gt = _resample_1d(source_energy.to(device=predictions["energy"].device).float(), target_len)
        valid = _resample_1d(_valid_mask(source_energy_mask, source_energy).float().to(gt.device), target_len, binary=True).bool()
        pred = predictions["energy"].float()
        if normalize_energy:
            pred = _masked_standardize(pred, valid)
            gt = _masked_standardize(gt, valid)
        term = _masked_mean_loss(F.smooth_l1_loss(pred, gt, reduction="none"), valid)
        if term is not None:
            terms.append(float(energy_weight) * term)
            stats["prosody_energy_loss"] = float(term.detach().item())

    if pause_weight > 0 and source_pause_mask is not None:
        gt = _resample_1d(source_pause_mask.to(device=predictions["pause_logit"].device).float(), target_len, binary=True)
        valid = torch.ones_like(gt, dtype=torch.bool)
        term = _masked_mean_loss(
            F.binary_cross_entropy_with_logits(predictions["pause_logit"].float(), gt.float(), reduction="none"),
            valid,
        )
        if term is not None:
            terms.append(float(pause_weight) * term)
            stats["prosody_pause_loss"] = float(term.detach().item())

    if duration_weight > 0 and source_duration is not None:
        target = torch.log1p(source_duration.to(device=predictions["log_duration"].device).float().clamp(min=0.0))
        if source_duration_mask is None:
            valid = torch.isfinite(target)
        else:
            valid = source_duration_mask.to(device=target.device).bool() & torch.isfinite(target)
        if bool(valid.any().item()):
            pred = predictions["log_duration"].float()
            term = F.smooth_l1_loss(pred[valid], target[valid], reduction="mean")
            terms.append(float(duration_weight) * term)
            stats["prosody_duration_loss"] = float(term.detach().item())

    if not terms:
        return AuxiliaryLossState(loss=None, stats={})
    loss = torch.stack(terms).sum()
    stats["prosody_loss_raw"] = float(loss.detach().item())
    return AuxiliaryLossState(loss=loss, stats=stats)


def compute_content_embedding_loss(
    prediction: torch.Tensor | None,
    positive_content_embedding: torch.Tensor | None,
    positive_content_embedding_mask: torch.Tensor | None = None,
    *,
    stat_prefix: str = "content_embedding",
) -> AuxiliaryLossState:
    if prediction is None or positive_content_embedding is None:
        return AuxiliaryLossState(loss=None, stats={})
    target = positive_content_embedding.to(device=prediction.device).float()
    if target.dim() != 2 or target.shape != prediction.shape:
        raise ValueError(f"content embedding shape {tuple(target.shape)} does not match prediction {tuple(prediction.shape)}")
    mask = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
    if positive_content_embedding_mask is not None:
        mask = positive_content_embedding_mask.to(device=target.device).bool()
    if not bool(mask.any().item()):
        return AuxiliaryLossState(loss=None, stats={})
    target = F.normalize(target, dim=-1)
    cosine = F.cosine_similarity(prediction.float(), target, dim=-1)
    loss = (1.0 - cosine[mask]).mean()
    return AuxiliaryLossState(
        loss=loss,
        stats={
            f"{stat_prefix}_loss": float(loss.detach().item()),
            f"{stat_prefix}_cos": float(cosine[mask].detach().mean().item()),
        },
    )


def _resample_token_ids(values: torch.Tensor, target_len: int) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"values must be [B, T], got {tuple(values.shape)}")
    if values.shape[1] == target_len:
        return values.long()
    if values.shape[1] <= 0:
        return values.new_zeros((values.shape[0], target_len), dtype=torch.long)
    out = F.interpolate(values.float().unsqueeze(1), size=int(target_len), mode="nearest").squeeze(1)
    return out.round().long().to(device=values.device)


def compute_content_token_loss(
    logits: torch.Tensor | None,
    content_ids: torch.Tensor | None,
    content_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
    stat_prefix: str = "content_token",
) -> AuxiliaryLossState:
    if logits is None or content_ids is None:
        return AuxiliaryLossState(loss=None, stats={})
    if logits.dim() != 3:
        raise ValueError(f"content token logits must be [B, T, V], got {tuple(logits.shape)}")
    if content_ids.dim() != 2:
        raise ValueError(f"content_ids must be [B, T_content], got {tuple(content_ids.shape)}")
    target_len = int(logits.shape[1])
    device = logits.device
    target = _resample_token_ids(content_ids.to(device=device), target_len)
    if content_mask is None:
        valid = target >= 0
    else:
        valid = _resample_1d(content_mask.to(device=device).float(), target_len, binary=True).bool()
        valid = valid & (target >= 0)
    target = target.masked_fill(~valid, int(ignore_index))
    if not bool(valid.any().item()):
        return AuxiliaryLossState(loss=None, stats={})
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        target.reshape(-1),
        ignore_index=int(ignore_index),
    )
    pred = logits.detach().float().argmax(dim=-1)
    acc = (pred[valid] == target[valid]).float().mean()
    return AuxiliaryLossState(
        loss=loss,
        stats={
            f"{stat_prefix}_loss": float(loss.detach().item()),
            f"{stat_prefix}_acc": float(acc.detach().item()),
            f"{stat_prefix}_valid_frames": float(valid.detach().float().sum().item()),
        },
    )


def compute_content_ctc_loss(
    logits: torch.Tensor | None,
    target_mask: torch.Tensor | None,
    token_ids: torch.Tensor | None,
    token_mask: torch.Tensor | None = None,
    *,
    blank_id: int = 0,
    stat_prefix: str = "content_ctc",
) -> AuxiliaryLossState:
    if logits is None or target_mask is None or token_ids is None:
        return AuxiliaryLossState(loss=None, stats={})
    if logits.dim() != 3:
        raise ValueError(f"CTC logits must be [B, T, V], got {tuple(logits.shape)}")
    if token_ids.dim() != 2:
        raise ValueError(f"CTC token_ids must be [B, S], got {tuple(token_ids.shape)}")
    device = logits.device
    mask = target_mask.to(device=device).bool()
    if mask.shape != logits.shape[:2]:
        raise ValueError(f"target_mask shape {tuple(mask.shape)} does not match logits {tuple(logits.shape[:2])}")
    input_lengths = mask.sum(dim=1).long()
    ids = token_ids.to(device=device).long()
    if token_mask is None:
        valid = ids.ne(int(blank_id)) & ids.ge(0)
    else:
        valid = token_mask.to(device=device).bool() & ids.ge(0) & ids.ne(int(blank_id))
    target_lengths = valid.sum(dim=1).long()
    batch_valid = (input_lengths > 0) & (target_lengths > 0) & (target_lengths <= input_lengths)
    if not bool(batch_valid.any().item()):
        return AuxiliaryLossState(loss=None, stats={})
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1).contiguous()
    flat_targets = []
    for batch_idx in torch.nonzero(batch_valid, as_tuple=False).flatten().tolist():
        flat_targets.append(ids[batch_idx][valid[batch_idx]])
    targets = torch.cat(flat_targets, dim=0)
    ctc = nn.CTCLoss(blank=int(blank_id), reduction="mean", zero_infinity=True)
    loss = ctc(
        log_probs[:, batch_valid],
        targets,
        input_lengths[batch_valid],
        target_lengths[batch_valid],
    )
    posterior_stats: dict[str, float] = {}
    if 0 <= int(blank_id) < int(logits.shape[-1]):
        with torch.no_grad():
            blank_probs = torch.softmax(logits.detach().float(), dim=-1)[..., int(blank_id)]
            valid_blank_probs = blank_probs[mask]
            if valid_blank_probs.numel() > 0:
                blank_mean = float(valid_blank_probs.mean().item())
                posterior_stats[f"{stat_prefix}_blank_posterior_mean"] = blank_mean
                posterior_stats[f"{stat_prefix}_nonblank_posterior_mean"] = 1.0 - blank_mean
    return AuxiliaryLossState(
        loss=loss,
        stats={
            f"{stat_prefix}_loss": float(loss.detach().item()),
            f"{stat_prefix}_valid_samples": float(batch_valid.detach().float().sum().item()),
            f"{stat_prefix}_input_len_mean": float(input_lengths[batch_valid].detach().float().mean().item()),
            f"{stat_prefix}_target_len_mean": float(target_lengths[batch_valid].detach().float().mean().item()),
            **posterior_stats,
        },
    )


def _resample_features(values: torch.Tensor, target_len: int) -> torch.Tensor:
    if values.dim() != 3:
        raise ValueError(f"features must be [B, T, D], got {tuple(values.shape)}")
    if values.shape[1] == target_len:
        return values
    if values.shape[1] <= 0:
        return values.new_zeros((values.shape[0], target_len, values.shape[2]))
    out = F.interpolate(
        values.float().transpose(1, 2),
        size=int(target_len),
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    return out.to(device=values.device)


def compute_semantic_feature_loss(
    prediction: torch.Tensor | None,
    semantic_features: torch.Tensor | None,
    semantic_mask: torch.Tensor | None = None,
    *,
    loss_type: str = "cosine",
    stat_prefix: str = "semantic_feature",
) -> AuxiliaryLossState:
    if prediction is None or semantic_features is None:
        return AuxiliaryLossState(loss=None, stats={})
    if prediction.dim() != 3:
        raise ValueError(f"prediction must be [B, T, D], got {tuple(prediction.shape)}")
    if semantic_features.dim() != 3:
        raise ValueError(f"semantic_features must be [B, T_sem, D], got {tuple(semantic_features.shape)}")
    if semantic_features.shape[2] != prediction.shape[2]:
        raise ValueError(
            f"semantic feature dim {semantic_features.shape[2]} does not match prediction dim {prediction.shape[2]}"
        )
    target_len = int(prediction.shape[1])
    target = _resample_features(semantic_features.to(device=prediction.device).float(), target_len)
    if semantic_mask is None:
        valid = torch.ones(target.shape[:2], dtype=torch.bool, device=prediction.device)
    else:
        valid = _resample_1d(semantic_mask.to(device=prediction.device).float(), target_len, binary=True).bool()
    if not bool(valid.any().item()):
        return AuxiliaryLossState(loss=None, stats={})
    pred = prediction.float()
    kind = str(loss_type or "cosine").lower()
    if kind == "mse":
        values = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
    else:
        values = 1.0 - F.cosine_similarity(pred, target, dim=-1)
    loss = _masked_mean_loss(values, valid)
    if loss is None:
        return AuxiliaryLossState(loss=None, stats={})
    return AuxiliaryLossState(
        loss=loss,
        stats={
            f"{stat_prefix}_loss": float(loss.detach().item()),
            f"{stat_prefix}_valid_frames": float(valid.detach().float().sum().item()),
        },
    )


def compute_progress_stop_loss(
    predictions: dict[str, torch.Tensor] | None,
    target_mask: torch.Tensor | None,
    *,
    progress_weight: float = 1.0,
    stop_weight: float = 1.0,
    stat_prefix: str = "content_order",
) -> AuxiliaryLossState:
    if predictions is None or target_mask is None:
        return AuxiliaryLossState(loss=None, stats={})
    progress_logits = predictions.get("progress_logits")
    stop_logit = predictions.get("stop_logit")
    if progress_logits is None or stop_logit is None:
        return AuxiliaryLossState(loss=None, stats={})
    if progress_logits.dim() != 3:
        raise ValueError(f"progress_logits must be [B, T, C], got {tuple(progress_logits.shape)}")
    if stop_logit.dim() != 2:
        raise ValueError(f"stop_logit must be [B, T], got {tuple(stop_logit.shape)}")
    mask = target_mask.to(device=progress_logits.device).bool()
    if mask.shape != progress_logits.shape[:2]:
        raise ValueError(f"target_mask shape {tuple(mask.shape)} does not match progress {tuple(progress_logits.shape[:2])}")
    if not bool(mask.any().item()):
        return AuxiliaryLossState(loss=None, stats={})

    device = progress_logits.device
    num_bins = int(progress_logits.shape[-1])
    progress_target = torch.zeros(mask.shape, dtype=torch.long, device=device)
    stop_target = torch.zeros(mask.shape, dtype=stop_logit.dtype, device=device)
    valid_samples = 0
    lengths = mask.sum(dim=1).long()
    for batch_idx, length in enumerate(lengths.detach().cpu().tolist()):
        if int(length) <= 0:
            continue
        valid_samples += 1
        positions = torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
        if int(length) == 1:
            bins = torch.zeros((1,), dtype=torch.long, device=device)
        else:
            bins = torch.linspace(0, num_bins - 1, steps=int(length), device=device).round().long()
        progress_target[batch_idx, positions] = bins.clamp(min=0, max=num_bins - 1)
        stop_target[batch_idx, positions[-1]] = 1.0

    if valid_samples <= 0:
        return AuxiliaryLossState(loss=None, stats={})

    terms: list[torch.Tensor] = []
    stats: dict[str, float] = {}
    if progress_weight > 0:
        progress_loss = F.cross_entropy(
            progress_logits.float().reshape(-1, num_bins),
            progress_target.reshape(-1),
            reduction="none",
        ).reshape(mask.shape)
        progress_term = _masked_mean_loss(progress_loss, mask)
        if progress_term is not None:
            terms.append(float(progress_weight) * progress_term)
            stats[f"{stat_prefix}_progress_loss"] = float(progress_term.detach().item())
            pred_bins = progress_logits.detach().float().argmax(dim=-1)
            stats[f"{stat_prefix}_progress_acc"] = float((pred_bins[mask] == progress_target[mask]).float().mean().item())
    if stop_weight > 0:
        stop_loss = F.binary_cross_entropy_with_logits(stop_logit.float(), stop_target.float(), reduction="none")
        stop_term = _masked_mean_loss(stop_loss, mask)
        if stop_term is not None:
            terms.append(float(stop_weight) * stop_term)
            stats[f"{stat_prefix}_stop_loss"] = float(stop_term.detach().item())
            pred_stop = torch.sigmoid(stop_logit.detach().float())
            stats[f"{stat_prefix}_stop_pos_prob"] = float(pred_stop[stop_target.bool()].mean().item()) if bool(stop_target.bool().any().item()) else 0.0
            stats[f"{stat_prefix}_stop_neg_prob"] = float(pred_stop[mask & ~stop_target.bool()].mean().item()) if bool((mask & ~stop_target.bool()).any().item()) else 0.0
            last5_values: list[torch.Tensor] = []
            middle5_values: list[torch.Tensor] = []
            for batch_idx, length in enumerate(lengths.detach().cpu().tolist()):
                if int(length) <= 0:
                    continue
                positions = torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
                cur_len = int(positions.numel())
                last5_positions = positions[max(0, cur_len - 5):]
                middle_start = max(0, min(cur_len - 5, cur_len // 2 - 2))
                middle5_positions = positions[middle_start:middle_start + min(5, cur_len)]
                if last5_positions.numel() > 0:
                    last5_values.append(pred_stop[batch_idx, last5_positions].reshape(-1))
                if middle5_positions.numel() > 0:
                    middle5_values.append(pred_stop[batch_idx, middle5_positions].reshape(-1))
            if last5_values and middle5_values:
                last5_mean = torch.cat(last5_values).mean()
                middle5_mean = torch.cat(middle5_values).mean()
                stats[f"{stat_prefix}_stop_last5_prob"] = float(last5_mean.item())
                stats[f"{stat_prefix}_stop_middle5_prob"] = float(middle5_mean.item())
                stats[f"{stat_prefix}_stop_last5_minus_middle5"] = float((last5_mean - middle5_mean).item())
    if not terms:
        return AuxiliaryLossState(loss=None, stats={})
    loss = torch.stack(terms).sum()
    stats[f"{stat_prefix}_loss_raw"] = float(loss.detach().item())
    stats[f"{stat_prefix}_valid_samples"] = float(valid_samples)
    stats[f"{stat_prefix}_target_len_mean"] = float(lengths[mask.any(dim=1)].float().mean().item())
    return AuxiliaryLossState(loss=loss, stats=stats)


def compute_source_codec_content_loss(
    logits: list[torch.Tensor] | None,
    codebooks: list[int],
    source_codes: torch.Tensor | None,
    source_mask: torch.Tensor | None = None,
    *,
    audio_pad_code: int | None = None,
) -> AuxiliaryLossState:
    if not logits or source_codes is None:
        return AuxiliaryLossState(loss=None, stats={})
    if source_codes.dim() != 3:
        raise ValueError(f"source_codes must be [B, T, n_vq], got {tuple(source_codes.shape)}")
    if len(logits) != len(codebooks):
        raise ValueError(f"logits/codebooks length mismatch: {len(logits)} != {len(codebooks)}")
    target_len = int(logits[0].shape[1])
    device = logits[0].device
    if source_mask is None:
        if audio_pad_code is None:
            valid = torch.ones(source_codes.shape[:2], dtype=torch.bool, device=source_codes.device)
        else:
            valid = (source_codes != int(audio_pad_code)).any(dim=-1)
    else:
        valid = source_mask.to(device=source_codes.device).bool()
    valid = _resample_1d(valid.float().to(device=device), target_len, binary=True).bool()

    terms: list[torch.Tensor] = []
    stats: dict[str, float] = {}
    for idx, (cur_logits, codebook) in enumerate(zip(logits, codebooks)):
        if cur_logits.dim() != 3:
            raise ValueError(f"codec content logits must be [B, T, V], got {tuple(cur_logits.shape)}")
        target = _resample_token_ids(source_codes[..., int(codebook)].to(device=device), target_len)
        target = target.masked_fill(~valid, -100)
        term = F.cross_entropy(
            cur_logits.reshape(-1, cur_logits.shape[-1]).float(),
            target.reshape(-1),
            ignore_index=-100,
        )
        if torch.isfinite(term):
            terms.append(term)
            stats[f"content_source_codec_cb{int(codebook)}_loss"] = float(term.detach().item())
    if not terms:
        return AuxiliaryLossState(loss=None, stats={})
    loss = torch.stack(terms).mean()
    stats["content_source_codec_loss"] = float(loss.detach().item())
    stats["content_source_codec_valid_frames"] = float(valid.detach().float().sum().item())
    return AuxiliaryLossState(loss=loss, stats=stats)
