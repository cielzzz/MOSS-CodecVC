import torch

from moss_codecvc.models.timbre_memory import ReferenceCodecTimbreMemory


def _make():
    return ReferenceCodecTimbreMemory(
        hidden_size=64,
        num_memory_tokens=32,
        adapter_dim=16,
        num_heads=4,
        speaker_embedding_dim=8,
        encoder_type="conformer",
        encoder_layers=1,
    )


def test_fix_g_speaker_conditioned_query_structure():
    module = _make()
    assert not hasattr(module, "query")
    assert hasattr(module, "query_generator")
    assert module.query_pos_embedding.shape == (32, 16)
    assert torch.isclose(module.query_scale.detach(), torch.tensor([5.0])).all()
    assert sum(p.numel() for p in module.query_generator.parameters()) > 1000


def test_fix_g_different_speakers_generate_different_queries():
    module = _make()
    reference = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8, dtype=torch.bool)
    speakers = torch.randn(2, 8)
    with torch.no_grad():
        q = module.query_generator(speakers).view(2, 32, 16)
        q = (q + module.query_pos_embedding).mul(module.query_scale)
        q = torch.nn.functional.normalize(q, dim=-1)
    cross = torch.nn.functional.cosine_similarity(q[0], q[1], dim=-1).mean()
    assert float(cross) < 0.7
    output = module(reference, ref_mask=mask, speaker_embedding=speakers)
    output.timbre_tokens.square().mean().backward()
    assert any(p.grad is not None for p in module.query_generator.parameters())


def test_fix_g_unconditional_speaker_is_supported():
    module = _make()
    output = module(
        torch.randn(1, 8, 64),
        ref_mask=torch.ones(1, 8, dtype=torch.bool),
        speaker_embedding=torch.zeros(1, 8),
    )
    assert torch.isfinite(output.timbre_tokens).all()
