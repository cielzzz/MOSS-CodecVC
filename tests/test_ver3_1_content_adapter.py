from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from moss_codecvc.models.content_adapter_v3_1 import ContentAdapterV31, _downsample_mask, count_parameters


def test_downsample_mask_matches_stride_four_conv() -> None:
    mask = torch.tensor([[True] * 10 + [False] * 2, [True] * 4 + [False] * 8])
    result = _downsample_mask(mask, kernel_size=4, stride=4)
    assert result.tolist() == [[True, True, False], [True, False, False]]


def test_adapter_outputs_12_5_hz_shape_and_mask() -> None:
    model = ContentAdapterV31(
        input_dim=768,
        semantic_dim=32,
        num_layers=1,
        num_heads=4,
        vocab_size=17,
    )
    features = torch.randn(2, 20, 768)
    mask = torch.tensor([[True] * 20, [True] * 13 + [False] * 7])
    output = model(features, mask)
    assert output.semantic.shape == (2, 5, 32)
    assert output.semantic_mask.tolist() == [[True] * 5, [True] * 3 + [False] * 2]
    assert output.logits is not None
    assert output.logits.shape == (2, 5, 17)
    assert count_parameters(model) > count_parameters(model, trainable_only=True) - 1


def test_adapter_rejects_mismatched_feature_dim() -> None:
    model = ContentAdapterV31(input_dim=8, semantic_dim=4, num_layers=0, vocab_size=0)
    with pytest.raises(ValueError, match="does not match input_dim"):
        model(torch.randn(1, 4, 7))


def test_semantic_extractor_atomic_npy_round_trip(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/ver3_1/extract_semantic_v3_1.py"
    spec = importlib.util.spec_from_file_location("extract_semantic_v31", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "a/b.npy"
    value = np.arange(15, dtype=np.float16).reshape(5, 3)
    size = module.atomic_npy(target, value)
    assert size == target.stat().st_size
    np.testing.assert_array_equal(np.load(target), value)
    assert not list(target.parent.glob("*.tmp-*"))
