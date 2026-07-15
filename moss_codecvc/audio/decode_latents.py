from __future__ import annotations

from typing import Any

import torch


def _resolve_tokenizer_model(model: Any) -> Any:
    """Resolve a MOSS audio-tokenizer model from common lightweight wrappers."""

    candidate = model
    seen: set[int] = set()
    while id(candidate) not in seen:
        seen.add(id(candidate))
        if hasattr(candidate, "decoder"):
            return candidate
        if hasattr(candidate, "module"):
            candidate = candidate.module
            continue
        if hasattr(candidate, "model"):
            candidate = candidate.model
            continue
        break
    raise TypeError(
        "model must be a MOSS audio-tokenizer model, or a wrapper exposing it "
        "through `.model`/`.module`; no decoder ModuleList was found"
    )


def _normalize_lengths(latents: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    batch_size, _, max_frames = latents.shape
    if lengths is None:
        return torch.full(
            (batch_size,),
            max_frames,
            dtype=torch.long,
            device=latents.device,
        )

    if not isinstance(lengths, torch.Tensor):
        raise TypeError(f"lengths must be a torch.Tensor, got {type(lengths).__name__}")
    if lengths.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"lengths must use an integer dtype, got {lengths.dtype}")
    normalized = lengths.to(dtype=torch.long, device=latents.device)
    if normalized.ndim != 1 or normalized.shape[0] != batch_size:
        raise ValueError(
            "lengths must be a 1-D latent-frame length tensor with one value per "
            f"batch item; got shape={tuple(normalized.shape)}, batch_size={batch_size}"
        )
    if bool(torch.any(normalized <= 0)):
        raise ValueError("all latent-frame lengths must be positive")
    if bool(torch.any(normalized > max_frames)):
        raise ValueError(
            f"latent-frame lengths cannot exceed latents.shape[-1]={max_frames}"
        )
    return normalized


@torch.inference_mode()
def decode_latents(
    model: Any,
    latents: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode decoder-domain MOSS latents without passing through the quantizer.

    Args:
        model: A loaded MOSS audio-tokenizer model, or a wrapper whose ``model``
            or ``module`` attribute resolves to one.
        latents: Decoder-domain dequantized latents in native ``(B, D, T)``
            layout. For the current MOSS Audio Tokenizer checkpoint this is
            ``D=768`` at 12.5 Hz. This function deliberately does not accept the
            DiT-friendly ``(B, T, D)`` layout to avoid an ambiguous transpose.
        lengths: Valid latent-frame counts with shape ``(B,)``. If omitted, all
            ``T`` frames are treated as valid.

    Returns:
        ``(waveform, waveform_lengths)`` matching the tokenizer's internal
        decoder output convention. ``waveform`` is normally ``(B, 1, samples)``
        and ``waveform_lengths`` is ``(B,)`` in waveform samples.

    Notes:
        The caller is responsible for placing ``latents`` on the same device
        and in a compatible dtype as the frozen tokenizer decoder. No
        quantization, dequantization, normalization, masking, or transpose is
        performed here.
    """

    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"latents must be a torch.Tensor, got {type(latents).__name__}")
    if latents.ndim != 3:
        raise ValueError(f"latents must have native (B, D, T) shape, got {tuple(latents.shape)}")
    if not latents.is_floating_point():
        raise TypeError(f"latents must be floating point, got dtype={latents.dtype}")
    if any(int(size) <= 0 for size in latents.shape):
        raise ValueError(f"latents dimensions must be non-empty, got {tuple(latents.shape)}")
    if not bool(torch.all(torch.isfinite(latents))):
        raise ValueError("latents must contain only finite values")

    tokenizer_model = _resolve_tokenizer_model(model)
    quantizer_kwargs = getattr(getattr(tokenizer_model, "config", None), "quantizer_kwargs", None)
    if isinstance(quantizer_kwargs, dict):
        expected_dim = quantizer_kwargs.get("output_dim") or quantizer_kwargs.get("input_dim")
        if expected_dim is not None and int(latents.shape[1]) != int(expected_dim):
            raise ValueError(
                "latents must use native decoder (B, D, T) layout; "
                f"expected D={int(expected_dim)}, got shape={tuple(latents.shape)}"
            )
    decoder = tokenizer_model.decoder
    if not hasattr(decoder, "__iter__"):
        raise TypeError("resolved tokenizer decoder must be an iterable module container")

    decoded_lengths = _normalize_lengths(latents, lengths)
    waveform = latents
    for decoder_module in decoder:
        waveform, decoded_lengths = decoder_module(waveform, decoded_lengths)

    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise RuntimeError(
            "MOSS decoder returned an invalid waveform tensor; expected (B, C, samples), "
            f"got {type(waveform).__name__} shape={getattr(waveform, 'shape', None)}"
        )
    decoded_lengths = torch.as_tensor(
        decoded_lengths,
        dtype=torch.long,
        device=waveform.device,
    )
    if decoded_lengths.ndim != 1 or decoded_lengths.shape[0] != waveform.shape[0]:
        raise RuntimeError(
            "MOSS decoder returned invalid waveform lengths: "
            f"shape={tuple(decoded_lengths.shape)}, batch_size={waveform.shape[0]}"
        )
    if bool(torch.any(decoded_lengths <= 0)) or bool(torch.any(decoded_lengths > waveform.shape[-1])):
        raise RuntimeError(
            "MOSS decoder returned waveform lengths outside the decoded tensor bounds: "
            f"max_samples={waveform.shape[-1]}, lengths={decoded_lengths.tolist()}"
        )
    return waveform, decoded_lengths
