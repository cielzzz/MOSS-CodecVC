from __future__ import annotations

from .speaker_encoder import (
    FrozenSeedTTSEvalECAPAEncoder,
    FrozenSpeakerEmbeddingLoader,
    FrozenSpeechBrainECAPAEncoder,
    build_frozen_speaker_encoder,
)
from .moss_codecvc_wrapper import (
    MossCodecVCTimbreMemoryWrapper,
    MossCodecVCTimbreSFTDataset,
    SOURCE_CONTINUOUS_MEMORY_TYPES,
    TimbreMemoryConfig,
    is_continuous_source_content_memory_type,
    normalize_source_content_memory_type,
)
from .speaker_cross_attn import SpeakerCrossAttentionLayer, SpeakerTokenProjector
from .auxiliary_losses import (
    ContentCTCHead,
    ContentEmbeddingHead,
    ContentTokenHead,
    ProsodyHead,
    SemanticFeatureHead,
    SourceCodecContentHead,
)
from .role_routing import PerCodebookTargetHeadRouter, RoleCodecRouter, SourceProsodyEncoder
from .source_semantic_memory import (
    SourceCodecBottleneckMemoryEncoder,
    SourceSemanticAdapter,
    SourceSemanticMemoryEncoder,
    SourceTokenMemoryEncoder,
)

__all__ = [
    "MossCodecVCTimbreMemoryWrapper",
    "MossCodecVCTimbreSFTDataset",
    "TimbreMemoryConfig",
    "SOURCE_CONTINUOUS_MEMORY_TYPES",
    "normalize_source_content_memory_type",
    "is_continuous_source_content_memory_type",
    "RoleCodecRouter",
    "SourceProsodyEncoder",
    "PerCodebookTargetHeadRouter",
    "SourceSemanticMemoryEncoder",
    "SourceTokenMemoryEncoder",
    "SourceCodecBottleneckMemoryEncoder",
    "SourceSemanticAdapter",
    "ProsodyHead",
    "ContentEmbeddingHead",
    "ContentCTCHead",
    "ContentTokenHead",
    "SemanticFeatureHead",
    "SourceCodecContentHead",
    "SpeakerCrossAttentionLayer",
    "SpeakerTokenProjector",
    "FrozenSpeakerEmbeddingLoader",
    "FrozenSpeechBrainECAPAEncoder",
    "FrozenSeedTTSEvalECAPAEncoder",
    "build_frozen_speaker_encoder",
]
