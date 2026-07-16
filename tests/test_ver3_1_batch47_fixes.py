from __future__ import annotations

from types import SimpleNamespace

import torch

from moss_codecvc.models.ddlfm_decoder import DDLFMDecoder
from scripts.ver3_1.infer_ddlfm_cfm import combine_dual_cfg_velocity
from scripts.ver3_1.train_ddlfm_cfm import (
    DDLFMTrainModule,
    apply_semantic_dropout,
    cfm_loss_weights,
    masked_mse,
    sample_cfm_time,
)


def test_shift_low_power_four_meets_low_t_contract() -> None:
    torch.manual_seed(47)
    values = sample_cfm_time(
        100_000,
        device=torch.device("cpu"),
        schedule="shift_low",
        shift_power=4.0,
    )
    assert float((values < 0.3).float().mean()) > 0.70
    assert float(values.mean()) < 0.30


def test_cfm_weight_cap_is_real_for_single_example() -> None:
    low = cfm_loss_weights(
        torch.tensor([0.0]), mode="low_t", eps=0.02, cap=25.0, normalize=False
    )
    high = cfm_loss_weights(
        torch.tensor([0.9]), mode="low_t", eps=0.02, cap=25.0, normalize=False
    )
    assert float(low.item()) == 25.0
    assert float(low.item() / high.item()) > 20.0

    prediction = torch.ones(1, 2, 3)
    target = torch.zeros_like(prediction)
    mask = torch.ones(1, 2, dtype=torch.bool)
    low_loss = masked_mse(
        prediction, target, mask, sample_weight=low, normalize_sample_weight=False
    )
    high_loss = masked_mse(
        prediction, target, mask, sample_weight=high, normalize_sample_weight=False
    )
    assert float(low_loss) > float(high_loss)


def test_semantic_dropout_masks_complete_rows() -> None:
    semantic = torch.ones(8, 3, 4)
    mask = torch.ones(8, 3, dtype=torch.bool)
    torch.manual_seed(2)
    dropped, dropped_mask, drop_rows = apply_semantic_dropout(semantic, mask, 1.0)
    assert bool(drop_rows.all())
    assert torch.equal(dropped, torch.zeros_like(semantic))
    assert not bool(dropped_mask.any())


def test_dual_cfg_scale_one_is_fully_conditioned_identity() -> None:
    cond = torch.randn(2, 4, 6)
    speaker = torch.randn_like(cond)
    semantic = torch.randn_like(cond)
    torch.testing.assert_close(
        combine_dual_cfg_velocity(cond, speaker, semantic, 1.0, 1.0), cond
    )
    torch.testing.assert_close(
        combine_dual_cfg_velocity(cond, speaker, semantic, 0.0, 0.0),
        cond - (cond - semantic) - (cond - speaker),
    )


def test_batch47_decoder_has_four_speaker_prompts_and_zero_speaker_anchor() -> None:
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
    assert model.speaker_prompt.shape == (1, 4, 8)
    assert len(model.speaker_prompt_mlps) == 4
    zero = torch.zeros(2, 3)
    with torch.no_grad():
        values = [mlp(zero) for mlp in model.speaker_prompt_mlps]
    assert all(torch.allclose(value, torch.zeros_like(value)) for value in values)
    for layer in model.layers:
        for projection in (layer.adaln_self[-1], layer.adaln_ffn[-1]):
            hidden = layer.hidden_size
            torch.testing.assert_close(
                projection.bias[hidden : 2 * hidden],
                torch.zeros(hidden),
            )


def test_batch47_aux_endpoint_forward_has_finite_gradients() -> None:
    args = SimpleNamespace(
        latent_dim=8,
        semantic_dim=6,
        speaker_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        text_vocab_size=17,
        text_padding_id=0,
        smoke_small_model=False,
        cross_gate_init=0.05,
        num_speaker_prompt_tokens=4,
    )
    module = DDLFMTrainModule(args)
    with torch.no_grad():
        torch.nn.init.normal_(module.decoder.output_proj.weight, std=0.1)
    x = torch.randn(2, 5, 8)
    target = torch.randn_like(x)
    semantic = torch.randn(2, 5, 6)
    speaker = torch.randn(2, 3)
    prediction = module.decoder(
        x,
        torch.zeros(2),
        semantic,
        speaker,
        target_mask=torch.ones(2, 5, dtype=torch.bool),
        semantic_mask=torch.ones(2, 5, dtype=torch.bool),
        condition_gate_scale=1.0,
    ).velocity
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert module.decoder.speaker_prompt.grad is not None
    assert float(module.decoder.speaker_prompt.grad.abs().sum()) > 0.0
