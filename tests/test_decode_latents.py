from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from moss_codecvc.audio.decode_latents import decode_latents


class _ScaleLength(nn.Module):
    def __init__(self, value_scale: float, length_scale: int) -> None:
        super().__init__()
        self.value_scale = value_scale
        self.length_scale = length_scale

    def forward(self, values: torch.Tensor, lengths: torch.Tensor):
        values = values.repeat_interleave(self.length_scale, dim=-1)
        return values * self.value_scale, lengths * self.length_scale


def _fake_model() -> SimpleNamespace:
    return SimpleNamespace(decoder=nn.ModuleList([_ScaleLength(2.0, 2), _ScaleLength(3.0, 4)]))


def test_decode_latents_runs_decoder_modules_without_quantizer() -> None:
    latents = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    waveform, lengths = decode_latents(_fake_model(), latents, torch.tensor([5, 3]))
    expected = latents.repeat_interleave(2, dim=-1).repeat_interleave(4, dim=-1) * 6.0
    torch.testing.assert_close(waveform, expected)
    assert lengths.tolist() == [40, 24]


def test_decode_latents_resolves_model_wrapper_and_defaults_lengths() -> None:
    latents = torch.ones(1, 3, 4)
    wrapped = SimpleNamespace(model=_fake_model())
    waveform, lengths = decode_latents(wrapped, latents)
    assert waveform.shape == (1, 3, 32)
    torch.testing.assert_close(waveform, torch.full_like(waveform, 6.0))
    assert lengths.tolist() == [32]


@pytest.mark.parametrize(
    ("latents", "lengths", "error_type", "message"),
    [
        (torch.ones(3, 4), None, ValueError, r"\(B, D, T\)"),
        (torch.ones(1, 3, 4, dtype=torch.long), None, TypeError, "floating point"),
        (torch.ones(2, 3, 4), torch.tensor([4]), ValueError, "one value per batch"),
        (torch.ones(1, 3, 4), torch.tensor([4.0]), TypeError, "integer dtype"),
        (torch.ones(1, 3, 4), torch.tensor([0]), ValueError, "positive"),
        (torch.ones(1, 3, 4), torch.tensor([5]), ValueError, "cannot exceed"),
        (torch.full((1, 3, 4), float("nan")), None, ValueError, "finite values"),
    ],
)
def test_decode_latents_rejects_invalid_inputs(
    latents: torch.Tensor,
    lengths: torch.Tensor | None,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        decode_latents(_fake_model(), latents, lengths)


def test_decode_latents_requires_decoder_model() -> None:
    with pytest.raises(TypeError, match="no decoder ModuleList"):
        decode_latents(object(), torch.ones(1, 3, 4))
