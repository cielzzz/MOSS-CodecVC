"""Audio-domain helpers for MOSS-CodecVC."""

from .decode_latents import decode_latents
from .zq_normalization import (
    CHANNEL_STATS_SCHEMA,
    DEFAULT_STD_FLOOR,
    ZQChannelStatsAccumulator,
    ZQNormalizer,
    denormalize_zq,
    load_zq_channel_stats,
    normalize_zq,
    normalization_max_abs_error,
    sha256_file,
    validate_channel_stats,
)

__all__ = [
    "CHANNEL_STATS_SCHEMA",
    "DEFAULT_STD_FLOOR",
    "ZQChannelStatsAccumulator",
    "ZQNormalizer",
    "decode_latents",
    "denormalize_zq",
    "load_zq_channel_stats",
    "normalize_zq",
    "normalization_max_abs_error",
    "sha256_file",
    "validate_channel_stats",
]
