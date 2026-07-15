from __future__ import annotations

import torch

from moss_codecvc.models.ddlfm_decoder import DDLFMDecoder


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    target = torch.randn(2, 5, 8)
    noise = torch.randn_like(target)
    t = torch.tensor([0.25, 0.75])
    x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target
    semantic = torch.randn(2, 4, 6)
    speaker = torch.randn(2, 3)
    prediction = model(x_t, t, semantic, speaker).velocity
    loss = torch.nn.functional.mse_loss(prediction, target - noise)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
