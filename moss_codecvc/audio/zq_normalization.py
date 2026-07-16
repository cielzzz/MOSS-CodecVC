"""Per-channel normalization helpers for decoder-domain MOSS latents.

The DDLFM target is stored in native decoder layout ``[D, T]`` per ``.npy``
file, while the DiT uses ``[B, T, D]``.  This module deliberately requires the
caller to state the channel dimension when normalizing a tensor; silently
guessing between these layouts is an easy way to train against transposed
statistics.

``ZQChannelStatsAccumulator`` is used by
``scripts/ver3_1/compute_zq_channel_stats.py``.  It accumulates one pass over
the saved ``.npy`` targets using float64 sums, and can be checkpointed between
manifest rows.  The resulting payload is a plain dictionary so it remains
easy to inspect with ``torch.load`` and can be consumed by future training and
inference code without importing this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


CHANNEL_STATS_SCHEMA = "ver3_1_zq_channel_stats_v1"
DEFAULT_STD_FLOOR = 1.0e-6


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return a streaming SHA256 for a control-plane file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(int(chunk_size)):
            digest.update(block)
    return digest.hexdigest()


def _as_channel_vector(
    value: Any,
    *,
    name: str,
    latent_dim: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device="cpu").flatten()
    if tensor.ndim != 1 or int(tensor.numel()) != int(latent_dim):
        raise ValueError(
            f"{name} must have shape [{int(latent_dim)}], got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def validate_channel_stats(
    stats: Mapping[str, Any],
    *,
    latent_dim: int | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate and normalize a saved channel-statistics payload.

    The returned dictionary keeps the original metadata and replaces mean/std
    vectors with CPU float64 tensors.  ``require_complete=False`` is useful
    for inspecting a progress checkpoint; normalizing latents always requires
    a complete ``mean`` and positive ``std`` vector.
    """

    if not isinstance(stats, Mapping):
        raise TypeError(f"channel stats must be a mapping, got {type(stats).__name__}")
    schema = str(stats.get("schema") or stats.get("schema_version") or "")
    if schema and schema != CHANNEL_STATS_SCHEMA:
        raise ValueError(f"unsupported channel stats schema: {schema!r}")
    inferred = stats.get("latent_dim")
    if inferred is None:
        if stats.get("mean") is not None:
            inferred = int(torch.as_tensor(stats["mean"]).numel())
        elif latent_dim is not None:
            inferred = int(latent_dim)
    if inferred is None or int(inferred) <= 0:
        raise ValueError("channel stats do not contain a positive latent_dim")
    dim = int(inferred)
    if latent_dim is not None and dim != int(latent_dim):
        raise ValueError(f"stats latent_dim={dim} does not match requested {latent_dim}")

    result = dict(stats)
    result["schema"] = schema or CHANNEL_STATS_SCHEMA
    result["latent_dim"] = dim
    if require_complete and (
        bool(stats.get("partial", False)) or str(stats.get("status") or "completed") != "completed"
    ):
        raise ValueError(
            f"channel stats are not complete: status={stats.get('status')!r}, "
            f"partial={stats.get('partial')!r}"
        )
    if require_complete and (stats.get("mean") is None or stats.get("std") is None):
        raise ValueError("complete channel stats require both mean and std")
    if stats.get("mean") is not None:
        result["mean"] = _as_channel_vector(stats["mean"], name="mean", latent_dim=dim)
    if stats.get("std") is not None:
        result["std"] = _as_channel_vector(stats["std"], name="std", latent_dim=dim)
        if bool(torch.any(result["std"] <= 0)):
            raise ValueError("std must be strictly positive after applying std_floor")
    if stats.get("raw_std") is not None:
        result["raw_std"] = _as_channel_vector(stats["raw_std"], name="raw_std", latent_dim=dim)
    if stats.get("count") is not None:
        count = torch.as_tensor(stats["count"], dtype=torch.long, device="cpu").flatten()
        if count.numel() != dim or bool(torch.any(count <= 0)):
            raise ValueError(f"count must contain {dim} positive per-channel counts")
        result["count"] = count
    return result


def load_zq_channel_stats(
    path: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Load and validate ``channel_stats.pt`` produced by the stats script."""

    payload = torch.load(str(Path(path).expanduser().resolve()), map_location="cpu", weights_only=False)
    return validate_channel_stats(payload, require_complete=require_complete)


def _broadcast_channel_vector(
    value: torch.Tensor,
    *,
    ndim: int,
    channel_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dim = int(channel_dim)
    if dim < 0:
        dim += int(ndim)
    if dim < 0 or dim >= int(ndim):
        raise ValueError(f"channel_dim={channel_dim} is invalid for ndim={ndim}")
    shape = [1] * int(ndim)
    shape[dim] = int(value.numel())
    return value.to(device=device, dtype=dtype).reshape(shape)


def _prepare_stats_for_latents(
    latents: torch.Tensor,
    stats: Mapping[str, Any] | str | Path,
    *,
    channel_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"latents must be a torch.Tensor, got {type(latents).__name__}")
    if not latents.is_floating_point():
        raise TypeError(f"latents must be floating point, got {latents.dtype}")
    if latents.ndim < 2:
        raise ValueError(f"latents must have at least 2 dimensions, got {tuple(latents.shape)}")
    resolved = load_zq_channel_stats(stats) if isinstance(stats, (str, Path)) else validate_channel_stats(stats)
    dim = int(channel_dim)
    if dim < 0:
        dim += latents.ndim
    if dim < 0 or dim >= latents.ndim:
        raise ValueError(f"channel_dim={channel_dim} is invalid for shape={tuple(latents.shape)}")
    if int(latents.shape[dim]) != int(resolved["latent_dim"]):
        raise ValueError(
            f"latent channel dimension {int(latents.shape[dim])} does not match stats "
            f"latent_dim={int(resolved['latent_dim'])}"
        )
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("latents contain non-finite values")
    mean = _broadcast_channel_vector(
        resolved["mean"],
        ndim=latents.ndim,
        channel_dim=dim,
        device=latents.device,
        dtype=latents.dtype,
    )
    std = _broadcast_channel_vector(
        resolved["std"],
        ndim=latents.ndim,
        channel_dim=dim,
        device=latents.device,
        dtype=latents.dtype,
    )
    return mean, std


def normalize_zq(
    latents: torch.Tensor,
    stats: Mapping[str, Any] | str | Path,
    *,
    channel_dim: int = 1,
) -> torch.Tensor:
    """Apply per-channel ``(zq - mean) / std``.

    ``channel_dim=1`` is the native ``[B,D,T]`` convention and
    ``channel_dim=-1`` is the DiT ``[B,T,D]`` convention.  The output keeps
    the input shape, device and dtype.
    """

    mean, std = _prepare_stats_for_latents(latents, stats, channel_dim=channel_dim)
    return (latents - mean) / std


def denormalize_zq(
    latents: torch.Tensor,
    stats: Mapping[str, Any] | str | Path,
    *,
    channel_dim: int = 1,
) -> torch.Tensor:
    """Invert :func:`normalize_zq` using the exact same channel convention."""

    mean, std = _prepare_stats_for_latents(latents, stats, channel_dim=channel_dim)
    return latents * std + mean


def normalization_max_abs_error(
    latents: torch.Tensor,
    stats: Mapping[str, Any] | str | Path,
    *,
    channel_dim: int = 1,
) -> float:
    """Return the max absolute normalize/inverse-normalize round-trip error."""

    restored = denormalize_zq(normalize_zq(latents, stats, channel_dim=channel_dim), stats, channel_dim=channel_dim)
    return float((restored - latents).abs().max().item())


class ZQNormalizer(nn.Module):
    """Cached module form of the per-channel transform.

    Mean/std are persistent buffers, so a DDLFM checkpoint remains
    self-describing even if the external stats file moves.  Callers still
    record the stats-file SHA in the run configuration for provenance.
    """

    def __init__(self, stats: Mapping[str, Any] | str | Path) -> None:
        super().__init__()
        resolved = load_zq_channel_stats(stats) if isinstance(stats, (str, Path)) else validate_channel_stats(stats)
        self.latent_dim = int(resolved["latent_dim"])
        self.std_floor = float(resolved.get("std_floor", DEFAULT_STD_FLOOR))
        self.register_buffer("mean", resolved["mean"].to(dtype=torch.float32), persistent=True)
        self.register_buffer("std", resolved["std"].to(dtype=torch.float32), persistent=True)

    def _vectors(self, latents: torch.Tensor, channel_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(latents, torch.Tensor) or not latents.is_floating_point():
            raise TypeError("latents must be a floating-point torch.Tensor")
        dim = int(channel_dim)
        if dim < 0:
            dim += latents.ndim
        if dim < 0 or dim >= latents.ndim:
            raise ValueError(f"channel_dim={channel_dim} is invalid for shape={tuple(latents.shape)}")
        if int(latents.shape[dim]) != self.latent_dim:
            raise ValueError(
                f"latent channel dimension {int(latents.shape[dim])} does not match {self.latent_dim}"
            )
        shape = [1] * latents.ndim
        shape[dim] = self.latent_dim
        mean = self.mean.to(dtype=latents.dtype).reshape(shape)
        std = self.std.to(dtype=latents.dtype).reshape(shape)
        return mean, std

    def normalize(self, latents: torch.Tensor, *, channel_dim: int = 1) -> torch.Tensor:
        mean, std = self._vectors(latents, channel_dim)
        return (latents - mean) / std

    def denormalize(self, latents: torch.Tensor, *, channel_dim: int = 1) -> torch.Tensor:
        mean, std = self._vectors(latents, channel_dim)
        return latents * std + mean

    def forward(self, latents: torch.Tensor, *, channel_dim: int = 1) -> torch.Tensor:
        return self.normalize(latents, channel_dim=channel_dim)


class ZQChannelStatsAccumulator:
    """Single-pass float64 accumulator over ``[D,T]`` latent arrays."""

    def __init__(self, latent_dim: int | None = None) -> None:
        self.latent_dim = int(latent_dim) if latent_dim is not None else None
        if self.latent_dim is not None and self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.sum: np.ndarray | None = None
        self.sum_squares: np.ndarray | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None
        self.frame_count = 0
        self.row_count = 0

    def _initialize(self, dim: int) -> None:
        if self.latent_dim is None:
            self.latent_dim = int(dim)
        if int(self.latent_dim) != int(dim):
            raise ValueError(f"latent dim changed from {self.latent_dim} to {dim}")
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_squares = np.zeros(dim, dtype=np.float64)
        self.minimum = np.full(dim, np.inf, dtype=np.float64)
        self.maximum = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, value: np.ndarray, *, chunk_frames: int = 8192) -> None:
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(f"zq array must have native [D,T] shape, got {array.shape}")
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"zq array must be floating point, got {array.dtype}")
        dim, frames = map(int, array.shape)
        if dim <= 0 or frames <= 0:
            raise ValueError(f"zq array dimensions must be positive, got {array.shape}")
        if self.sum is None:
            self._initialize(dim)
        assert self.sum is not None and self.sum_squares is not None
        assert self.minimum is not None and self.maximum is not None
        chunk_size = max(1, int(chunk_frames))
        for start in range(0, frames, chunk_size):
            chunk = np.asarray(array[:, start : start + chunk_size], dtype=np.float64)
            if not np.isfinite(chunk).all():
                raise ValueError("zq array contains non-finite values")
            self.sum += chunk.sum(axis=1, dtype=np.float64)
            self.sum_squares += np.square(chunk, dtype=np.float64).sum(axis=1, dtype=np.float64)
            self.minimum = np.minimum(self.minimum, chunk.min(axis=1))
            self.maximum = np.maximum(self.maximum, chunk.max(axis=1))
        self.frame_count += frames
        self.row_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "sum": self.sum,
            "sum_squares": self.sum_squares,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "frame_count": int(self.frame_count),
            "row_count": int(self.row_count),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ZQChannelStatsAccumulator":
        result = cls(state.get("latent_dim"))
        dim = result.latent_dim
        if dim is None:
            raise ValueError("checkpoint is missing latent_dim")
        for key in ("sum", "sum_squares", "minimum", "maximum"):
            value = np.asarray(state.get(key), dtype=np.float64).reshape(-1)
            if value.size != dim:
                raise ValueError(f"checkpoint {key} has {value.size} values, expected {dim}")
            setattr(result, key, value.copy())
        result.frame_count = int(state.get("frame_count", 0))
        result.row_count = int(state.get("row_count", 0))
        if result.frame_count <= 0 or result.row_count < 0:
            raise ValueError("checkpoint has invalid frame/row counts")
        return result

    def finalize(
        self,
        *,
        std_floor: float = DEFAULT_STD_FLOOR,
        metadata: Mapping[str, Any] | None = None,
        partial: bool = False,
    ) -> dict[str, Any]:
        if self.sum is None or self.sum_squares is None or self.minimum is None or self.maximum is None:
            raise ValueError("cannot finalize an empty accumulator")
        floor = float(std_floor)
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError(f"std_floor must be finite and positive, got {std_floor}")
        count = np.full(int(self.latent_dim), int(self.frame_count), dtype=np.int64)
        mean = self.sum / float(self.frame_count)
        variance = self.sum_squares / float(self.frame_count) - np.square(mean)
        variance = np.maximum(variance, 0.0)
        raw_std = np.sqrt(variance)
        std = np.maximum(raw_std, floor)
        payload: dict[str, Any] = {
            "schema": CHANNEL_STATS_SCHEMA,
            "status": "partial" if partial else "completed",
            "partial": bool(partial),
            "created_at_unix": time.time(),
            "latent_dim": int(self.latent_dim),
            "frame_count": int(self.frame_count),
            "row_count": int(self.row_count),
            "value_count": int(self.frame_count * int(self.latent_dim)),
            "dtype": "float32_input_float64_accumulator",
            "frame_rate_hz": 12.5,
            "std_floor": floor,
            "mean": torch.from_numpy(mean.copy()),
            "std": torch.from_numpy(std.copy()),
            "raw_std": torch.from_numpy(raw_std.copy()),
            "count": torch.from_numpy(count),
            "sum": torch.from_numpy(self.sum.copy()),
            "sum_squares": torch.from_numpy(self.sum_squares.copy()),
            "min": torch.from_numpy(self.minimum.copy()),
            "max": torch.from_numpy(self.maximum.copy()),
        }
        if metadata:
            payload.update(dict(metadata))
        # Validate before publishing so a malformed payload cannot become the
        # canonical stats file.
        validate_channel_stats(payload, require_complete=not partial)
        return payload


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Atomically publish a torch payload, preserving an interrupted run."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def json_safe_stats(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert tensor-heavy stats into a compact JSON-auditable mapping."""

    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    return convert(dict(payload))


def atomic_json_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Atomically write a JSON audit companion."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(json_safe_stats(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
