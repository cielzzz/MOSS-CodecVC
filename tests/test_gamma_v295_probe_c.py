from __future__ import annotations

import torch

from moss_codecvc.models.content_cross_attn import ContentConformerEncoder
from moss_codecvc.models.moss_codecvc_wrapper import (
    TimbreMemoryConfig,
    load_content_encoder_state_with_probe_migration,
)


def test_probe_c_encoder_has_four_conformer_layers() -> None:
    encoder = ContentConformerEncoder(
        input_dim=8,
        hidden_size=16,
        num_layers=4,
        num_heads=4,
        dropout=0.0,
        conv_kernel_size=7,
    )
    assert len(encoder.layers) == 4
    features = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    state = encoder(features, mask)
    assert state.memory.shape == (2, 5, 16)
    assert torch.isfinite(state.memory).all()


def test_probe_c_config_weights() -> None:
    config = TimbreMemoryConfig(
        content_cross_attn_enabled=True,
        content_encoder_layers=4,
        guided_attn_loss_weight=0.10,
        guided_attn_warmup_steps=0,
        phoneme_classifier_loss_weight=0.05,
        content_token_vocab_size=32,
    )
    assert config.content_encoder_layers == 4
    assert config.guided_attn_loss_weight == 0.10
    assert config.phoneme_classifier_loss_weight == 0.05


def test_probe_c_migrates_two_layers_and_rejects_unrelated_missing_keys() -> None:
    baseline = ContentConformerEncoder(input_dim=8, hidden_size=16, num_layers=2, num_heads=4)
    widened = ContentConformerEncoder(input_dim=8, hidden_size=16, num_layers=4, num_heads=4)
    result = load_content_encoder_state_with_probe_migration(widened, baseline.state_dict())
    assert result["initialized_new_layers"] == ["2", "3"]
    for key, value in baseline.state_dict().items():
        assert torch.equal(widened.state_dict()[key], value)

    broken = dict(baseline.state_dict())
    broken.pop("input.1.weight")
    try:
        load_content_encoder_state_with_probe_migration(
            ContentConformerEncoder(input_dim=8, hidden_size=16, num_layers=4, num_heads=4),
            broken,
        )
    except RuntimeError as exc:
        assert "missing non-migration keys" in str(exc)
    else:
        raise AssertionError("unrelated missing checkpoint keys must fail closed")


def test_probe_c_new_layers_receive_gradients() -> None:
    encoder = ContentConformerEncoder(input_dim=8, hidden_size=16, num_layers=4, num_heads=4)
    output = encoder(torch.randn(2, 6, 8)).memory
    loss = output.square().mean()
    loss.backward()
    assert encoder.layers[2].ff1[0].weight.grad is not None
    assert encoder.layers[3].ff2[3].weight.grad is not None
