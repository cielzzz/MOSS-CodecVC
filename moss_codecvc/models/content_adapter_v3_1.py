"""Content Adapter used by the ver3.1 DDLFM semantic path.

The adapter consumes a frame sequence from WavLM (the v1 manifests currently
contain ``microsoft/wavlm-base-plus`` layer-9 features, 768 dimensions) and
maps it to a 12.5-Hz, 512-dimensional semantic memory.  The downsampler is a
causal-free stride-4 convolution, matching the 50-Hz -> 12.5-Hz contract.

The classifier head is deliberately optional.  During the Step-3 probe it is
used for the existing manifest ``content_token_ids`` labels.  Those labels are
SentencePiece content tokens rather than MFA phoneme IDs; callers should keep
the recorded ``label_source`` distinction in reports instead of calling the
result phoneme accuracy unless real phoneme alignments are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .content_cross_attn import ContentConformerBlock
from .content_semantic_heads import ContentCTCHead


@dataclass
class ContentAdapterOutput:
    semantic: torch.Tensor
    semantic_mask: torch.Tensor
    logits: torch.Tensor | None


def _downsample_mask(mask: torch.Tensor, *, kernel_size: int, stride: int) -> torch.Tensor:
    """Return the valid windows produced by a no-padding Conv1d."""

    if mask.dim() != 2:
        raise ValueError(f"mask must be [B,T], got {tuple(mask.shape)}")
    if mask.shape[1] < int(kernel_size):
        return mask.new_zeros((mask.shape[0], 0), dtype=torch.bool)
    pooled = F.avg_pool1d(
        mask.to(dtype=torch.float32).unsqueeze(1),
        kernel_size=int(kernel_size),
        stride=int(stride),
        padding=0,
        ceil_mode=False,
    ).squeeze(1)
    return pooled.ge(1.0 - 1.0e-6)


class ContentAdapterV31(nn.Module):
    """WavLM layer-9 -> 12.5-Hz semantic adapter for ver3.1."""

    def __init__(
        self,
        input_dim: int,
        semantic_dim: int = 512,
        *,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        conv_kernel_size: int = 7,
        downsample_kernel_size: int = 4,
        downsample_stride: int = 4,
        vocab_size: int = 0,
        classifier_adapter_dim: int = 256,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        semantic_dim = int(semantic_dim)
        if input_dim <= 0 or semantic_dim <= 0:
            raise ValueError("input_dim and semantic_dim must be positive")
        if int(downsample_stride) <= 0 or int(downsample_kernel_size) <= 0:
            raise ValueError("downsample kernel/stride must be positive")
        self.input_dim = input_dim
        self.semantic_dim = semantic_dim
        self.num_layers = max(0, int(num_layers))
        self.downsample_kernel_size = int(downsample_kernel_size)
        self.downsample_stride = int(downsample_stride)
        self.vocab_size = int(vocab_size)

        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, semantic_dim),
            nn.Dropout(float(dropout)),
        )
        self.downsample = nn.Conv1d(
            semantic_dim,
            semantic_dim,
            kernel_size=self.downsample_kernel_size,
            stride=self.downsample_stride,
            padding=0,
        )
        self.layers = nn.ModuleList(
            [
                ContentConformerBlock(
                    semantic_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(semantic_dim)
        self.classifier = (
            ContentCTCHead(
                semantic_dim,
                int(vocab_size),
                adapter_dim=classifier_adapter_dim,
                dropout=dropout,
            )
            if int(vocab_size) > 1
            else None
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_logits: bool | None = None,
    ) -> ContentAdapterOutput:
        """Project WavLM features to the 12.5-Hz semantic memory.

        ``return_logits`` is accepted as a compatibility keyword for early
        Step-3 smoke scripts.  The adapter always returns a
        :class:`ContentAdapterOutput`; callers that do not need the
        auxiliary classifier can simply ignore ``.logits``.  Keeping the
        keyword harmless is preferable to silently maintaining two model
        implementations with incompatible checkpoints.
        """
        if features.dim() != 3:
            raise ValueError(f"features must be [B,T,D], got {tuple(features.shape)}")
        if int(features.shape[-1]) != self.input_dim:
            raise ValueError(
                f"feature dim={features.shape[-1]} does not match input_dim={self.input_dim}"
            )
        module_dtype = self.input[0].weight.dtype
        x = self.input(features.to(dtype=module_dtype))
        if mask is None:
            in_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        else:
            in_mask = mask.to(device=features.device).bool()
            if in_mask.shape != features.shape[:2]:
                raise ValueError(
                    f"mask shape {tuple(in_mask.shape)} does not match {tuple(features.shape[:2])}"
                )
        x = self.downsample(x.transpose(1, 2)).transpose(1, 2)
        out_mask = _downsample_mask(
            in_mask,
            kernel_size=self.downsample_kernel_size,
            stride=self.downsample_stride,
        )
        if out_mask.shape[1] != x.shape[1]:
            # This should only be reachable for a backend-specific Conv1d
            # shape change; fail loudly rather than silently misalign labels.
            raise RuntimeError(
                f"downsample mask length {out_mask.shape[1]} != output length {x.shape[1]}"
            )
        x = x.masked_fill(~out_mask.unsqueeze(-1), 0.0)
        for layer in self.layers:
            x = layer(x, out_mask)
        x = self.output_norm(x).masked_fill(~out_mask.unsqueeze(-1), 0.0)
        logits = self.classifier(x) if self.classifier is not None else None
        return ContentAdapterOutput(semantic=x, semantic_mask=out_mask, logits=logits)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        int(parameter.numel())
        for parameter in module.parameters()
        if (parameter.requires_grad or not trainable_only)
    )


__all__ = [
    "ContentAdapterOutput",
    "ContentAdapterV31",
    "_downsample_mask",
    "count_parameters",
]
