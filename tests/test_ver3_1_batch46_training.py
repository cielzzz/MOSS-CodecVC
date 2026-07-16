from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from scripts.ver3_1.train_ddlfm_cfm import (
    DDLFMDataset,
    DDLFMTrainModule,
    ExponentialMovingAverage,
    apply_speaker_dropout,
    save_checkpoint,
    cfm_loss_weights,
    masked_mse,
    sample_cfm_time,
)
from scripts.ver3_1.infer_ddlfm_cfm import combine_cfg_velocity, load_checkpoint


def test_logit_normal_time_is_centered_and_more_midrange_than_uniform() -> None:
    torch.manual_seed(1234)
    logit = sample_cfm_time(
        50_000,
        device=torch.device("cpu"),
        schedule="logit_normal",
    )
    torch.manual_seed(1234)
    uniform = sample_cfm_time(
        50_000,
        device=torch.device("cpu"),
        schedule="uniform",
    )
    assert 0.49 < float(logit.mean()) < 0.51
    logit_mid = ((logit > 0.2) & (logit < 0.8)).float().mean()
    uniform_mid = ((uniform > 0.2) & (uniform < 0.8)).float().mean()
    assert float(logit_mid) > float(uniform_mid) + 0.15


def test_low_t_weighting_really_emphasizes_noise_endpoint() -> None:
    t = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95])
    low = cfm_loss_weights(t, mode="low_t", eps=0.05, cap=5.0)
    high = cfm_loss_weights(t, mode="high_t", eps=0.05, cap=5.0)
    assert low[0] > low[-1]
    assert high[0] < high[-1]
    torch.testing.assert_close(low.mean(), torch.tensor(1.0))
    torch.testing.assert_close(high.mean(), torch.tensor(1.0))


def test_weighted_masked_mse_respects_per_example_weights() -> None:
    prediction = torch.tensor([[[2.0]], [[1.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.ones(2, 1, dtype=torch.bool)
    plain = masked_mse(prediction, target, mask)
    weighted = masked_mse(
        prediction,
        target,
        mask,
        sample_weight=torch.tensor([2.0, 0.5]),
    )
    torch.testing.assert_close(plain, torch.tensor(2.5))
    torch.testing.assert_close(weighted, torch.tensor(3.4))


def test_speaker_dropout_zeros_whole_condition_rows() -> None:
    speaker = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    kept, keep_mask = apply_speaker_dropout(speaker, 0.0)
    torch.testing.assert_close(kept, speaker)
    assert not bool(keep_mask.any())
    dropped, drop_mask = apply_speaker_dropout(speaker, 1.0)
    assert bool(drop_mask.all())
    torch.testing.assert_close(dropped, torch.zeros_like(speaker))


def test_no_text_dataset_filter_excludes_text_rows(tmp_path: Path) -> None:
    rows = [
        {
            "moss_codecvc_mode": "no_text",
            "zq_path": "a.npy",
            "semantic_path": "a-sem.npy",
            "speaker_embedding_path": "a.pt",
        },
        {
            "moss_codecvc_mode": "text",
            "zq_path": "b.npy",
            "content_token_ids": [1, 2, 3],
            "speaker_embedding_path": "b.pt",
        },
    ]
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert len(DDLFMDataset(index, mode="no_text")) == 1
    assert len(DDLFMDataset(index, mode="all")) == 2


def test_ema_tracks_updated_module_and_serializes_full_state() -> None:
    module = nn.Linear(3, 2)
    initial = {name: value.detach().clone() for name, value in module.state_dict().items()}
    ema = ExponentialMovingAverage(module, decay=0.5, warmup=False)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(2.0)
    ema.update(module)
    state = ema.state_dict_cpu()
    assert state.keys() == module.state_dict().keys()
    for name, value in module.state_dict().items():
        expected = initial[name] * 0.5 + value.detach() * 0.5
        torch.testing.assert_close(state[name], expected.cpu())


def test_ema_warmup_avoids_initialization_dominated_short_probe() -> None:
    module = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.zero_()
    ema = ExponentialMovingAverage(module, decay=0.9999, warmup=True)
    with torch.no_grad():
        module.weight.fill_(1.0)
    ema.update(module)
    assert ema.num_updates == 1
    assert abs(ema.last_decay - (2.0 / 11.0)) < 1.0e-12
    assert float(ema.state_dict_cpu()["weight"].item()) > 0.80
    assert ema.effective_decay(500) < 0.99
    assert ema.effective_decay(3000) < 0.998


def test_batch46_module_applies_cross_gate_small_init() -> None:
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
    )
    module = DDLFMTrainModule(args)
    hidden = module.decoder.layers[0].hidden_size
    bias = module.decoder.layers[0].condition[-1].bias
    torch.testing.assert_close(
        bias[5 * hidden : 6 * hidden],
        torch.full((hidden,), 0.05),
    )


def test_cfg_combination_matches_conditional_formula_and_rejects_nonfinite() -> None:
    cond = torch.tensor([1.0, 3.0])
    uncond = torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(combine_cfg_velocity(cond, uncond, 0.0), uncond)
    torch.testing.assert_close(combine_cfg_velocity(cond, uncond, 1.0), cond)
    torch.testing.assert_close(
        combine_cfg_velocity(cond, uncond, 1.5),
        uncond + 1.5 * (cond - uncond),
    )
    for invalid in (float("nan"), float("inf"), -0.1):
        try:
            combine_cfg_velocity(cond, uncond, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid cfg scale was accepted: {invalid}")


def test_checkpoint_loader_reports_actual_ema_fallback(tmp_path: Path) -> None:
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
    )
    module = DDLFMTrainModule(args)
    checkpoint = tmp_path / "legacy-no-ema.pt"
    torch.save(
        {
            "model": module.state_dict(),
            "config": vars(args),
        },
        checkpoint,
    )
    loaded, cfg = load_checkpoint(checkpoint, torch.device("cpu"), use_ema=True)
    assert isinstance(loaded, DDLFMTrainModule)
    assert cfg["_checkpoint_has_ema"] is False
    assert cfg["_checkpoint_using_ema"] is False


def test_checkpoint_publish_is_atomic_and_writes_ready_marker(tmp_path: Path) -> None:
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
    )
    module = DDLFMTrainModule(args)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    ema = ExponentialMovingAverage(module, decay=0.9999, warmup=True)
    ema.update(module)
    save_checkpoint(module, optimizer, ema, 500, args, tmp_path, rank=0)
    checkpoint = tmp_path / "step-000500.pt"
    inference_checkpoint = tmp_path / "step-000500.infer.pt"
    last = tmp_path / "last.pt"
    ready = tmp_path / "step-000500.ready.json"
    assert checkpoint.is_file() and inference_checkpoint.is_file() and last.is_file() and ready.is_file()
    marker = json.loads(ready.read_text(encoding="utf-8"))
    assert marker["status"] == "ready" and marker["step"] == 500
    assert marker["checkpoint_size_bytes"] == checkpoint.stat().st_size
    assert marker["inference_checkpoint_size_bytes"] == inference_checkpoint.stat().st_size
    assert inference_checkpoint.stat().st_ino == last.stat().st_ino
    assert not list(tmp_path.glob(".*.tmp-*"))
