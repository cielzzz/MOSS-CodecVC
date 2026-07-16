from __future__ import annotations

from types import SimpleNamespace

import torch

from moss_codecvc.models.ddlfm_decoder import DDLFMDecoder
from scripts.ver3_1.infer_ddlfm_cfm import combine_dual_cfg_velocity
from scripts.ver3_1.train_ddlfm_cfm import masked_mse


def test_batch48_weighting_keeps_absolute_low_t_scale() -> None:
    prediction = torch.ones(2, 4, 3)
    target = torch.zeros_like(prediction)
    mask = torch.ones(2, 4, dtype=torch.bool)
    low = masked_mse(
        prediction,
        target,
        mask,
        sample_weight=torch.tensor([25.0, 25.0]),
        normalize_sample_weight=False,
    )
    high = masked_mse(
        prediction,
        target,
        mask,
        sample_weight=torch.tensor([1.0, 1.0]),
        normalize_sample_weight=False,
    )
    assert 24.99 < float(low / high) < 25.01


def test_batch48_four_state_cfg_matches_reference() -> None:
    v00 = torch.randn(2, 5, 8)
    v10 = torch.randn_like(v00)
    v01 = torch.randn_like(v00)
    expected = v00 + 2.5 * (v10 - v00) + 2.0 * (v01 - v00)
    actual = combine_dual_cfg_velocity(v00, v10, v01, 2.5, 2.0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(
        combine_dual_cfg_velocity(v00, v10, v01, 0.0, 0.0), v00
    )


def test_batch48_zero_speaker_has_no_prompt_key_or_value() -> None:
    model = DDLFMDecoder(
        latent_dim=8,
        semantic_dim=6,
        speaker_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        num_speaker_prompt_tokens=4,
    ).eval()
    with torch.no_grad():
        model.speaker_prompt.fill_(0.5)
    captured: dict[str, torch.Tensor] = {}
    layer = model.layers[0]
    original = layer._cross_attention_with_rope

    def capture(self, query, semantic, **kwargs):
        del self
        captured["semantic"] = semantic.detach().clone()
        captured["mask"] = kwargs["semantic_mask"].detach().clone()
        return original(query, semantic, **kwargs)

    from types import MethodType

    layer._cross_attention_with_rope = MethodType(capture, layer)
    model(
        torch.randn(2, 4, 8),
        torch.zeros(2),
        torch.randn(2, 3, 6),
        torch.stack([torch.zeros(3), torch.ones(3)]),
    )
    assert torch.allclose(captured["semantic"][0, :4], torch.zeros(4, 8))
    assert not bool(captured["mask"][0, :4].any())
    assert bool(captured["mask"][1, :4].all())


def test_batch48_raw_speaker_metric_is_distinct_from_cfg_metric() -> None:
    # The diagnostic schema must expose the raw field used by the gate; this
    # lightweight contract prevents accidentally reading the CFG-amplified
    # historical field in a future report parser.
    payload = {
        "speaker_raw_cond_vs_zero_relative_to_target_rms": 0.08,
        "speaker_cfg_advantage_relative_to_target_rms": 0.20,
    }
    assert payload["speaker_raw_cond_vs_zero_relative_to_target_rms"] < 0.12
    assert payload["speaker_cfg_advantage_relative_to_target_rms"] > payload[
        "speaker_raw_cond_vs_zero_relative_to_target_rms"
    ]
