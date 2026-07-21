import torch

from moss_codecvc.models.timbre_memory import ReferenceCodecTimbreMemory


def test_fix_e_query_initialization_and_position_embedding():
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
    assert 0.35 < float(module.query.detach().std()) < 0.65
    query = module.query.detach() + module.query_pos_embedding
    assert float(query.norm(dim=-1).mean()) > 1.0


def test_fix_e_query_position_produces_distinct_queries_and_backward():
    module = ReferenceCodecTimbreMemory(
        hidden_size=64,
        num_memory_tokens=32,
        adapter_dim=16,
        num_heads=4,
        encoder_type="conformer",
        encoder_layers=1,
    )
    query = module.query.detach() + module.query_pos_embedding
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

