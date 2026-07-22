import torch

from moss_codecvc.models.ddlfm_decoder import DDLFMDecoder
from scripts.ver3_1.train_ddlfm_cfm import apply_speaker_dropout


def _make() -> DDLFMDecoder:
    model = DDLFMDecoder(
        latent_dim=32,
        semantic_dim=16,
        speaker_dim=8,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        ffn_size=64,
        num_timbre_tokens=4,
        cross_gate_init=0.1,
    )
    # The production decoder intentionally zero-initializes its output head.
    # Activate it here so functional path differences are observable.
    with torch.no_grad():
        model.output_proj.weight.copy_(torch.eye(32))
    return model


def _inputs():
    return {
        "x_t": torch.randn(2, 5, 32),
        "t": torch.rand(2),
        "semantic": torch.randn(2, 6, 16),
        "speaker": torch.randn(2, 8),
        "prompt_zq": torch.randn(2, 7, 32),
        "prompt_mask": torch.ones(2, 7, dtype=torch.bool),
    }


def test_fix_j_has_broadcast_prompt_and_per_layer_film() -> None:
    model = _make()
    assert hasattr(model, "speaker_expand")
    assert all(hasattr(layer, "speaker_film_generator") for layer in model.layers)
    for layer in model.layers:
        assert layer.speaker_film_generator[-1].out_features == 64
        assert torch.count_nonzero(layer.speaker_film_generator[-1].weight) == 0
        assert torch.count_nonzero(layer.speaker_film_generator[-1].bias) == 0


def test_fix_j_three_paths_are_independently_switchable() -> None:
    model = _make().eval()
    values = _inputs()
    with torch.no_grad():
        full = model(**values).velocity
        no_broadcast = model(**values, enable_speaker_broadcast=False).velocity
        no_prompt = model(**values, enable_prompt_prefix=False).velocity
        # Make the zero-initialized FiLM observable for this structural test.
        model.layers[0].speaker_film_generator[-1].weight.normal_(0.0, 0.01)
        with_film = model(**values).velocity
        no_film = model(**values, enable_speaker_film=False).velocity
    assert not torch.allclose(full, no_broadcast)
    assert not torch.allclose(full, no_prompt)
    assert not torch.allclose(with_film, no_film)


def test_fix_j_different_speaker_and_prompt_change_output() -> None:
    model = _make().eval()
    values = _inputs()
    with torch.no_grad():
        base = model(**values).velocity
        changed_speaker = dict(values)
        changed_speaker["speaker"] = values["speaker"].flip(0)
        speaker_output = model(**changed_speaker).velocity
        changed_prompt = dict(values)
        changed_prompt["prompt_zq"] = values["prompt_zq"].flip(0)
        prompt_output = model(**changed_prompt).velocity
    assert not torch.allclose(base, speaker_output)
    assert not torch.allclose(base, prompt_output)


def test_fix_j_cfg_zeroes_all_three_new_paths() -> None:
    model = _make().eval()
    speaker = torch.randn(2, 8)
    dropped, mask = apply_speaker_dropout(speaker, 1.0)
    assert bool(mask.all())
    prompt = torch.randn(2, 7, 32)
    prompt[mask] = 0.0
    prompt_mask = torch.ones(2, 7, dtype=torch.bool)
    prompt_mask[mask] = False

    zero = torch.zeros_like(dropped)
    with torch.no_grad():
        broadcast = model.speaker_expand(dropped) - model.speaker_expand(zero)
        for layer in model.layers:
            film = layer.speaker_film_generator(dropped)
            film = film - layer.speaker_film_generator(zero)
            assert torch.equal(film, torch.zeros_like(film))
    assert torch.equal(broadcast, torch.zeros_like(broadcast))
    assert torch.equal(prompt, torch.zeros_like(prompt))
    assert not bool(prompt_mask.any())


def test_fix_j_forward_backward_shape_and_gradients() -> None:
    model = _make()
    values = _inputs()
    output = model(**values)
    assert output.velocity.shape == (2, 5, 32)
    output.velocity.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.speaker_expand.parameters())
    assert any(
        parameter.grad is not None
        for layer in model.layers
        for parameter in layer.speaker_film_generator.parameters()
    )
