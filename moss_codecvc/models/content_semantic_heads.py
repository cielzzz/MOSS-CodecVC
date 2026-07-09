from __future__ import annotations

from .auxiliary_losses import ContentCTCHead, ContentEmbeddingHead, ContentTokenHead, SemanticFeatureHead

CTCContentHead = ContentCTCHead

__all__ = [
    "ContentCTCHead",
    "CTCContentHead",
    "ContentEmbeddingHead",
    "ContentTokenHead",
    "SemanticFeatureHead",
]
