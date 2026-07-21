import torch

from moss_codecvc.models.ddlfm_decoder import DDLFMDecoder


def test_tte_hybrid_shapes_and_backward():
    model = DDLFMDecoder(
        latent_dim=64,
        semantic_dim=16,
        speaker_dim=8,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        ffn_size=128,
        num_timbre_tokens=4,
    )
    x_t = torch.randn(2, 5, 64)
    semantic = torch.randn(2, 6, 16)
    speaker = torch.randn(2, 8)
    prompt = torch.randn(2, 8, 64)
    output = model(
        x_t,
        torch.rand(2),
        semantic,
        speaker,
        prompt_zq=prompt,
        prompt_mask=torch.ones(2, 8, dtype=torch.bool),
    )
    assert output.velocity.shape == (2, 5, 64)
    output.velocity.square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.timbre_memory.parameters()
    )


def test_tte_unconditional_prompt_mask_is_supported():
    model = DDLFMDecoder(
        latent_dim=32,
        semantic_dim=8,
        speaker_dim=4,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        ffn_size=64,
        num_timbre_tokens=4,
    )
    output = model(
        torch.randn(1, 3, 32),
        torch.zeros(1),
        torch.randn(1, 4, 8),
        torch.zeros(1, 4),
        prompt_zq=torch.zeros(1, 8, 32),
        prompt_mask=torch.zeros(1, 8, dtype=torch.bool),
    )
    assert torch.isfinite(output.velocity).all()
