from __future__ import annotations

from types import MethodType

import torch

from moss_codecvc.models.ddlfm_decoder import DDLFMAdaLNBlock, DDLFMDecoder


def _activate_cross_attention_for_regression(model: DDLFMDecoder) -> None:
    """Make the zero-initialized DiT expose semantic effects in unit tests."""

    with torch.no_grad():
        torch.nn.init.normal_(model.output_proj.weight, std=0.1)
        model.output_proj.bias.zero_()
        for layer in model.layers:
            hidden = layer.hidden_size
            # AdaLN-Zero starts every residual gate at zero.  Activate only
            # the semantic cross-attention gate for an inference-only test.
            layer.condition[-1].bias[5 * hidden : 6 * hidden].fill_(1.0)


def test_rotary_cross_attention_matches_mha_when_all_positions_are_zero() -> None:
    torch.manual_seed(5)
    block = DDLFMAdaLNBlock(
        hidden_size=16,
        num_heads=4,
        ffn_size=32,
        condition_size=16,
    ).eval()
    query = torch.randn(2, 5, 16)
    semantic = torch.randn(2, 7, 16)
    semantic_mask = torch.tensor(
        [[True] * 7, [True] * 4 + [False] * 3], dtype=torch.bool
    )
    reference = block.cross_attn(
        query,
        semantic,
        semantic,
        key_padding_mask=~semantic_mask,
        need_weights=False,
    )[0]
    actual = block._cross_attention_with_rope(
        query,
        semantic,
        semantic_mask=semantic_mask,
        target_positions=torch.zeros(5),
        semantic_positions=torch.zeros(7),
    )
    assert torch.allclose(actual, reference, atol=1.0e-6, rtol=1.0e-6)


def test_ddlfm_decoder_supports_variable_semantic_lengths_and_modalities() -> None:
    torch.manual_seed(7)
    model = DDLFMDecoder(
        latent_dim=16,
        semantic_dim=8,
        speaker_dim=4,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        ffn_size=32,
    )
    x = torch.randn(2, 11, 16)
    semantic = torch.randn(2, 6, 8)
    speaker = torch.randn(2, 4)
    target_mask = torch.tensor(
        [[True] * 11, [True] * 7 + [False] * 4], dtype=torch.bool
    )
    semantic_mask = torch.tensor(
        [[True] * 6, [True] * 3 + [False] * 3], dtype=torch.bool
    )
    out = model(
        x,
        torch.tensor([0.2, 0.8]),
        semantic,
        speaker,
        target_mask=target_mask,
        semantic_mask=semantic_mask,
        semantic_modality=torch.tensor([0, 1]),
    )
    assert out.velocity.shape == x.shape
    assert torch.isfinite(out.velocity).all()
    assert torch.allclose(out.velocity[1, 7:], torch.zeros_like(out.velocity[1, 7:]))


def test_ddlfm_decoder_cfm_backward_smoke() -> None:
    torch.manual_seed(11)
    model = DDLFMDecoder(
        latent_dim=8,
        semantic_dim=6,
        speaker_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
    )
    _activate_cross_attention_for_regression(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    target = torch.randn(2, 5, 8)
    noise = torch.randn_like(target)
    t = torch.tensor([0.25, 0.75])
    x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target
    semantic = torch.randn(2, 4, 6, requires_grad=True)
    speaker = torch.randn(2, 3, requires_grad=True)
    prediction = model(x_t, t, semantic, speaker).velocity
    loss = torch.nn.functional.mse_loss(prediction, target - noise)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert semantic.grad is not None and float(semantic.grad.abs().sum().item()) > 0.0
    assert speaker.grad is not None and float(speaker.grad.abs().sum().item()) > 0.0
    assert model.speaker_prompt.grad is not None
    assert float(model.speaker_prompt.grad.abs().sum().item()) > 0.0
    speaker_linear_grads = [
        module.weight.grad
        for module in model.speaker_proj
        if isinstance(module, torch.nn.Linear)
    ]
    assert all(grad is not None and float(grad.abs().sum().item()) > 0.0 for grad in speaker_linear_grads)
    cross_grad = model.layers[0].cross_attn.in_proj_weight.grad
    assert cross_grad is not None and float(cross_grad.abs().sum().item()) > 0.0
    optimizer.step()


def test_ddlfm_decoder_semantic_permutation_changes_velocity() -> None:
    """Regression: semantic K/V must no longer be permutation invariant."""

    torch.manual_seed(123)
    model = DDLFMDecoder(
        latent_dim=16,
        semantic_dim=8,
        speaker_dim=4,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        ffn_size=32,
    ).eval()
    _activate_cross_attention_for_regression(model)
    x = torch.randn(1, 9, 16)
    semantic = torch.randn(1, 7, 8)
    semantic_mask = torch.tensor([[True] * 5 + [False] * 2])
    speaker = torch.randn(1, 4)

    original = model(
        x,
        torch.tensor([0.3]),
        semantic,
        speaker,
        semantic_mask=semantic_mask,
    ).velocity
    permuted_semantic = semantic.clone()
    permuted_semantic[:, :5] = semantic[:, :5].flip(1)
    permuted = model(
        x,
        torch.tensor([0.3]),
        permuted_semantic,
        speaker,
        semantic_mask=semantic_mask,
    ).velocity
    max_diff = float((original - permuted).abs().max().item())
    assert max_diff > 1.0e-3

    # RoPE must not weaken padding semantics: changing masked K/V values has
    # no effect while permuting valid frames does.
    changed_padding = semantic.clone()
    changed_padding[:, 5:] = changed_padding[:, 5:] + 1000.0
    padding_output = model(
        x,
        torch.tensor([0.3]),
        changed_padding,
        speaker,
        semantic_mask=semantic_mask,
    ).velocity
    assert torch.allclose(original, padding_output, atol=1.0e-6, rtol=1.0e-6)


def test_ddlfm_decoder_condition_gate_scale_and_speaker_only_memory() -> None:
    torch.manual_seed(321)
    model = DDLFMDecoder(
        latent_dim=8,
        semantic_dim=6,
        speaker_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
    ).eval()
    _activate_cross_attention_for_regression(model)
    x = torch.randn(1, 5, 8).expand(2, -1, -1).clone()
    semantic = torch.randn(1, 4, 6).expand(2, -1, -1).clone()
    # All semantic frames are masked.  The decoder-prefixed speaker prompt is
    # still a valid K/V token, so this path must remain finite for CFG.
    semantic_mask = torch.zeros(2, 4, dtype=torch.bool)
    speaker = torch.randn(2, 3)
    t = torch.tensor([0.4, 0.4])

    enabled = model(
        x,
        t,
        semantic,
        speaker,
        semantic_mask=semantic_mask,
        condition_gate_scale=1.0,
    ).velocity
    assert torch.isfinite(enabled).all()
    assert not torch.allclose(enabled[0], enabled[1])

    disabled = model(
        x,
        t,
        semantic,
        speaker,
        semantic_mask=semantic_mask,
        condition_gate_scale=0.0,
    ).velocity
    disabled_with_other_conditions = model(
        x,
        t,
        semantic.flip(1),
        -speaker,
        semantic_mask=semantic_mask,
        condition_gate_scale=0.0,
    ).velocity
    assert torch.allclose(disabled, disabled_with_other_conditions, atol=1.0e-6, rtol=1.0e-6)


def test_ddlfm_decoder_short_row_is_invariant_to_batch_padding_length() -> None:
    torch.manual_seed(456)
    model = DDLFMDecoder(
        latent_dim=8,
        semantic_dim=6,
        speaker_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
    ).eval()
    _activate_cross_attention_for_regression(model)
    x = torch.randn(1, 5, 8)
    semantic = torch.randn(1, 3, 6)
    speaker = torch.randn(1, 3)
    single = model(x, torch.tensor([0.35]), semantic, speaker).velocity

    padded_semantic = torch.zeros(2, 7, 6)
    padded_semantic[0, :3] = semantic[0]
    padded_semantic[1] = torch.randn(7, 6)
    semantic_mask = torch.tensor(
        [[True] * 3 + [False] * 4, [True] * 7], dtype=torch.bool
    )
    batched = model(
        torch.cat([x, torch.randn_like(x)], dim=0),
        torch.tensor([0.35, 0.65]),
        padded_semantic,
        torch.cat([speaker, torch.randn_like(speaker)], dim=0),
        semantic_mask=semantic_mask,
    ).velocity
    assert torch.allclose(single[0], batched[0], atol=1.0e-6, rtol=1.0e-6)


def test_ddlfm_decoder_conditioning_diagnostics_reports_effective_adaln_values() -> None:
    model = DDLFMDecoder(
        latent_dim=8,
        semantic_dim=6,
        speaker_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
    )
    hidden = model.layers[0].hidden_size
    with torch.no_grad():
        bias = model.layers[0].condition[-1].bias
        bias[3 * hidden : 4 * hidden].fill_(-0.1)
        bias[4 * hidden : 5 * hidden].fill_(0.25)
        bias[5 * hidden : 6 * hidden].fill_(2.0)
    diagnostics = model.conditioning_diagnostics(
        torch.tensor([0.2, 0.8]),
        torch.randn(2, 3),
        gate_scale=0.5,
    )
    assert diagnostics["modulate_multiplicative_scale_formula"] == "1 + scale"
    assert diagnostics["gate_scale"] == 0.5
    cross = diagnostics["layers"][0]["cross"]
    assert abs(cross["shift_mean_abs"] - 0.1) < 1.0e-6
    assert abs(cross["scale_mean_abs"] - 0.25) < 1.0e-6
    assert abs(cross["multiplicative_scale_mean"] - 1.25) < 1.0e-6
    assert abs(cross["gate_mean_abs"] - 1.0) < 1.0e-6


def test_ddlfm_decoder_cross_gate_small_init_and_warmup_scope() -> None:
    model = DDLFMDecoder(
        latent_dim=8,
        semantic_dim=6,
        speaker_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        cross_gate_init=0.05,
    )
    hidden = model.layers[0].hidden_size
    bias = model.layers[0].condition[-1].bias
    expected = torch.zeros_like(bias)
    expected[5 * hidden : 6 * hidden] = 0.05
    assert torch.allclose(bias, expected)

    with torch.no_grad():
        bias[2 * hidden : 3 * hidden].fill_(0.2)
        bias[8 * hidden : 9 * hidden].fill_(0.3)
    diagnostics = model.conditioning_diagnostics(
        torch.tensor([0.5]),
        torch.randn(1, 3),
        gate_scale=0.0,
    )
    layer = diagnostics["layers"][0]
    assert abs(layer["self"]["gate_mean_abs"] - 0.2) < 1.0e-6
    assert layer["cross"]["gate_mean_abs"] == 0.0
    assert abs(layer["ffn"]["gate_mean_abs"] - 0.3) < 1.0e-6


def test_ddlfm_decoder_speaker_prompt_is_fixed_position_zero() -> None:
    torch.manual_seed(654)
    model = DDLFMDecoder(
        latent_dim=8,
        semantic_dim=6,
        speaker_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
    ).eval()
    layer = model.layers[0]
    captured: dict[str, torch.Tensor] = {}
    original = layer._cross_attention_with_rope

    def capture_cross_attention(
        self,
        query: torch.Tensor,
        semantic: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del self
        captured["semantic"] = semantic.detach().clone()
        captured["semantic_mask"] = kwargs["semantic_mask"].detach().clone()
        captured["semantic_positions"] = kwargs["semantic_positions"].detach().clone()
        return original(query, semantic, **kwargs)

    layer._cross_attention_with_rope = MethodType(capture_cross_attention, layer)
    speaker = torch.randn(2, 3)
    semantic_mask = torch.tensor(
        [[True] * 4, [True] * 2 + [False] * 2], dtype=torch.bool
    )
    model(
        torch.randn(2, 5, 8),
        torch.tensor([0.2, 0.8]),
        torch.randn(2, 4, 6),
        speaker,
        semantic_mask=semantic_mask,
    )

    expected_prompt = torch.cat(
        [
            (
                mlp(speaker) + model.speaker_prompt[:, index, :]
            ).unsqueeze(1)
            for index, mlp in enumerate(model.speaker_prompt_mlps)
        ],
        dim=1,
    )
    assert torch.allclose(captured["semantic"][:, :4], expected_prompt)
    assert captured["semantic"].shape[1] == 8
    assert captured["semantic_mask"].tolist() == [
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, False, False],
    ]
    assert captured["semantic_positions"].tolist() == [-4, -3, -2, -1, 0, 1, 2, 3]
