from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from moss_codecvc.audio.decode_latents import decode_latents
from moss_codecvc.audio.zq_normalization import (
    ZQChannelStatsAccumulator,
    ZQNormalizer,
    denormalize_zq,
    load_zq_channel_stats,
    normalize_zq,
    normalization_max_abs_error,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ver3_1/compute_zq_channel_stats.py"
SPEC = importlib.util.spec_from_file_location("compute_zq_channel_stats_v3_1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compute_stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute_stats)


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


def _stats_for(value: np.ndarray) -> dict[str, object]:
    accumulator = ZQChannelStatsAccumulator(value.shape[0])
    accumulator.update(value, chunk_frames=2)
    return accumulator.finalize(std_floor=1.0e-6)


def test_normalize_inverse_supports_native_and_dit_layouts() -> None:
    native = torch.tensor(
        [[[1.0, 2.0, 3.0], [10.0, 14.0, 18.0]],
         [[4.0, 5.0, 6.0], [22.0, 26.0, 30.0]]],
        dtype=torch.float32,
    )
    stats = _stats_for(native.permute(1, 0, 2).reshape(2, -1).numpy())
    # Build a non-degenerate, explicit 2-channel stats payload.  The values
    # above are only used to ensure both layouts are exercised.
    stats = {
        "schema": "ver3_1_zq_channel_stats_v1",
        "latent_dim": 2,
        "mean": torch.tensor([2.0, 14.0], dtype=torch.float64),
        "std": torch.tensor([1.0, 4.0], dtype=torch.float64),
    }
    normalized_native = normalize_zq(native, stats, channel_dim=1)
    restored_native = denormalize_zq(normalized_native, stats, channel_dim=1)
    assert normalization_max_abs_error(native, stats, channel_dim=1) < 1.0e-6
    torch.testing.assert_close(restored_native, native, atol=1.0e-6, rtol=0.0)

    dit = native.transpose(1, 2).contiguous()
    normalized_dit = normalize_zq(dit, stats, channel_dim=-1)
    restored_dit = denormalize_zq(normalized_dit, stats, channel_dim=-1)
    torch.testing.assert_close(restored_dit, dit, atol=1.0e-6, rtol=0.0)

    module = ZQNormalizer(stats)
    torch.testing.assert_close(
        module.denormalize(module(dit, channel_dim=-1), channel_dim=-1),
        dit,
        atol=1.0e-6,
        rtol=0.0,
    )
    assert set(module.state_dict()) == {"mean", "std"}


def test_decode_after_normalize_inverse_matches_direct_decode() -> None:
    native = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5) / 7.0
    stats = {
        "schema": "ver3_1_zq_channel_stats_v1",
        "latent_dim": 3,
        "mean": torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64),
        "std": torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64),
    }
    normalized = normalize_zq(native, stats, channel_dim=1)
    restored = denormalize_zq(normalized, stats, channel_dim=1)
    direct_audio, direct_lengths = decode_latents(_fake_model(), native, torch.tensor([5, 3]))
    restored_audio, restored_lengths = decode_latents(_fake_model(), restored, torch.tensor([5, 3]))
    torch.testing.assert_close(restored_audio, direct_audio, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(restored_lengths, direct_lengths)


def test_accumulator_uses_per_channel_counts_and_std_floor() -> None:
    first = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]], dtype=np.float32)
    second = np.array([[4.0], [5.0]], dtype=np.float32)
    accumulator = ZQChannelStatsAccumulator(2)
    accumulator.update(first, chunk_frames=1)
    accumulator.update(second, chunk_frames=1)
    stats = accumulator.finalize(std_floor=0.1)
    torch.testing.assert_close(stats["mean"], torch.tensor([2.5, 5.0], dtype=torch.float64))
    assert float(stats["raw_std"][1]) == 0.0
    assert float(stats["std"][1]) == 0.1
    assert stats["count"].tolist() == [4, 4]
    assert stats["frame_count"] == 4
    assert stats["row_count"] == 2


def test_stats_script_partial_then_resume(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for index, frames in enumerate((2, 3, 1, 4)):
        value = np.arange(3 * frames, dtype=np.float32).reshape(3, frames) + float(index)
        target = tmp_path / f"zq-{index}.npy"
        np.save(target, value)
        rows.append(
            {
                "output_path": str(target),
                "latent_dim": 3,
                "frame_rate_hz": 12.5,
                "num_frames": frames,
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "channel_stats.pt"

    assert compute_stats.main(
        [
            "--manifest", str(manifest),
            "--output", str(output),
            "--latent-dim", "3",
            "--max-rows", "2",
            "--checkpoint-every-rows", "1",
        ]
    ) == 0
    partial = torch.load(output, map_location="cpu", weights_only=False)
    assert partial["status"] == "partial"
    with pytest.raises(ValueError, match="not complete"):
        load_zq_channel_stats(output)
    progress = output.with_suffix(".progress.pt")
    assert progress.is_file()

    assert compute_stats.main(
        [
            "--manifest", str(manifest),
            "--output", str(output),
            "--latent-dim", "3",
            "--resume",
            "--checkpoint-every-rows", "1",
        ]
    ) == 0
    complete = torch.load(output, map_location="cpu", weights_only=False)
    assert complete["status"] == "completed"
    assert complete["partial"] is False
    assert complete["row_count"] == 4
    assert complete["frame_count"] == 10
    assert not progress.exists()
    expected = np.concatenate([np.load(row["output_path"]) for row in rows], axis=1)
    expected_stats = _stats_for(expected)
    torch.testing.assert_close(complete["mean"], expected_stats["mean"])
    torch.testing.assert_close(complete["std"], expected_stats["std"])
