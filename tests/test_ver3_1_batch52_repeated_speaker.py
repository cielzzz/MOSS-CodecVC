import torch

from moss_codecvc.models.ddlfm_decoder import DDLFMDecoder


def _make():
    return DDLFMDecoder(
        latent_dim=32,
        semantic_dim=16,
        speaker_dim=8,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        ffn_size=64,
        num_timbre_tokens=4,
    )


def test_batch52_repeated_speaker_structure_and_parameter_count():
    model = _make()
    assert not hasattr(model, "timbre_memory")
    assert hasattr(model, "speaker_expand")
    assert 8 * 32 < sum(p.numel() for p in model.speaker_expand.parameters()) < 500_000


def test_batch52_different_speakers_create_different_prefixes():
    model = _make()
    speakers = torch.randn(2, 8)
    prefixes = model.speaker_expand(speakers)
    cosine = torch.nn.functional.cosine_similarity(prefixes[0], prefixes[1], dim=0)
    assert float(cosine) < 0.99


def test_batch52_forward_backward_and_unconditional_speaker():
    model = _make()
    x_t = torch.randn(2, 5, 32)
    semantic = torch.randn(2, 6, 16)
    speaker = torch.randn(2, 8)
    output = model(x_t, torch.rand(2), semantic, speaker)
    assert output.velocity.shape == (2, 5, 32)
    output.velocity.square().mean().backward()
    assert any(p.grad is not None for p in model.speaker_expand.parameters())

    with torch.no_grad():
        zero_prefix = model.speaker_expand(torch.zeros(1, 8))
    assert torch.allclose(zero_prefix, torch.zeros_like(zero_prefix), atol=1e-6)

