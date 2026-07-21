import torch

from moss_codecvc.models.timbre_memory import ReferenceCodecTimbreMemory


def test_fix_f_query_initialization_position_embedding_and_scale():
    module = ReferenceCodecTimbreMemory(
        hidden_size=64,
        num_memory_tokens=32,
        adapter_dim=16,
        num_heads=4,
        encoder_type="conformer",
        encoder_layers=1,
    )
    assert module.query.shape == (32, 16)
    assert module.query_pos_embedding.shape == (32, 16)
    assert not hasattr(module, "query_norm")
    assert torch.isclose(module.query_scale.detach(), torch.tensor([5.0])).all()
    assert 0.35 < float(module.query.detach().std()) < 0.65
    query = (module.query.detach() + module.query_pos_embedding) * module.query_scale.detach()
    assert float(query.norm(dim=-1).mean()) > 1.0


def test_fix_f_query_position_produces_distinct_queries_and_backward():
    module = ReferenceCodecTimbreMemory(
        hidden_size=64,
        num_memory_tokens=32,
        adapter_dim=16,
        num_heads=4,
        encoder_type="conformer",
        encoder_layers=1,
    )
    query = (module.query.detach() + module.query_pos_embedding) * module.query_scale.detach()
    query = torch.nn.functional.normalize(query, dim=-1)
    cosine = query @ query.t()
    off_diag = cosine[~torch.eye(cosine.shape[0], dtype=torch.bool)]
    assert float(off_diag.mean()) < 0.7
    output = module(
        torch.randn(2, 8, 64),
        ref_mask=torch.ones(2, 8, dtype=torch.bool),
    )
    output.timbre_tokens.square().mean().backward()
    assert module.query.grad is not None
    assert module.query_scale.grad is not None
