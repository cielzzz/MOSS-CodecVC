#!/usr/bin/env python
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.models import MossCodecVCTimbreMemoryWrapper, TimbreMemoryConfig


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return (torch.tanh(self.proj(hidden_states)),)


class FakeLanguageModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoderLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


class FakeMossDelayModel(nn.Module):
    def __init__(self, *, n_vq: int = 4, hidden_size: int = 16, text_vocab_size: int = 32, audio_vocab_size: int = 20) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            n_vq=n_vq,
            audio_vocab_size=audio_vocab_size,
            audio_pad_code=audio_vocab_size,
            language_config=SimpleNamespace(hidden_size=hidden_size, vocab_size=text_vocab_size),
        )
        self.text_embed = nn.Embedding(text_vocab_size, hidden_size)
        self.emb_ext = nn.ModuleList([nn.Embedding(audio_vocab_size + 1, hidden_size) for _ in range(n_vq)])
        self.language_model = FakeLanguageModel(hidden_size, num_layers=4)
        self.lm_heads = nn.ModuleList([nn.Linear(hidden_size, text_vocab_size)])
        self.lm_heads.extend(nn.Linear(hidden_size, audio_vocab_size + 1) for _ in range(n_vq))

    def get_base_model(self):
        return self

    def get_input_embeddings(self):
        return self.text_embed

    def _compute_input_embeddings(self, input_ids):
        embeds = self.text_embed(input_ids[..., 0])
        for idx, emb in enumerate(self.emb_ext):
            embeds = embeds + emb(input_ids[..., idx + 1])
        return embeds

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        channelwise_loss_weight=None,
        inputs_embeds=None,
        **kwargs,
    ):
        _ = (attention_mask, channelwise_loss_weight, kwargs)
        hidden_states = self.language_model(inputs_embeds if inputs_embeds is not None else self._compute_input_embeddings(input_ids))
        logits = [head(hidden_states) for head in self.lm_heads]
        loss = None
        if labels is not None:
            losses = []
            for channel, channel_logits in enumerate(logits):
                if not bool((labels[..., channel] != -100).any().item()):
                    continue
                vocab = channel_logits.shape[-1]
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                losses.append(loss_fct(channel_logits.reshape(-1, vocab), labels[..., channel].reshape(-1)))
            loss = torch.stack(losses).mean()
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            hidden_states=(hidden_states,),
            past_key_values=None,
            attentions=None,
            all_sum_losses=None,
            all_token_nums=None,
            sample_losses=None,
            channel_losses=None,
        )


def main() -> int:
    torch.manual_seed(7)
    batch_size = 2
    seq_len = 12
    ref_len = 7
    n_vq = 4
    hidden_size = 16
    audio_vocab_size = 20
    model = FakeMossDelayModel(n_vq=n_vq, hidden_size=hidden_size, audio_vocab_size=audio_vocab_size)
    for param in model.parameters():
        param.requires_grad = False

    with tempfile.TemporaryDirectory() as tmpdir:
        source_embedding_paths = []
        timbre_embedding_paths = []
        target_embedding_paths = []
        for idx in range(batch_size):
            source_path = Path(tmpdir) / f"source_{idx}.pt"
            timbre_path = Path(tmpdir) / f"timbre_{idx}.pt"
            target_path = Path(tmpdir) / f"target_{idx}.pt"
            torch.save({"embedding": torch.randn(hidden_size)}, source_path)
            torch.save({"embedding": torch.randn(hidden_size)}, timbre_path)
            torch.save({"embedding": torch.randn(hidden_size)}, target_path)
            source_embedding_paths.append(str(source_path))
            timbre_embedding_paths.append(str(timbre_path))
            target_embedding_paths.append(str(target_path))

        wrapper = MossCodecVCTimbreMemoryWrapper(
            model,
            TimbreMemoryConfig(
                enabled=True,
                num_memory_tokens=4,
                adapter_layers="last_2",
                num_heads=4,
                dropout=0.0,
                target_speaker_similarity_weight=0.1,
                source_speaker_suppression_weight=0.05,
                speaker_embedding_dim=hidden_size,
                use_role_routing=True,
                route_loss_weight=0.01,
                prosody_memory_tokens=3,
                target_head_routing=True,
            ),
        )
        input_ids = torch.zeros(batch_size, seq_len, n_vq + 1, dtype=torch.long)
        input_ids[..., 0] = torch.randint(0, model.config.language_config.vocab_size, (batch_size, seq_len))
        input_ids[..., 1:] = torch.randint(0, audio_vocab_size, (batch_size, seq_len, n_vq))
        labels = torch.full_like(input_ids, -100)
        labels[:, -3:, 1:] = torch.randint(0, audio_vocab_size, (batch_size, 3, n_vq))
        target_positions = (labels != -100).any(dim=-1)
        source_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        timbre_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        source_positions[:, :4] = True
        timbre_positions[:, 4:7] = True
        timbre_ref_codes = torch.randint(0, audio_vocab_size, (batch_size, ref_len, n_vq))
        timbre_ref_mask = torch.ones(batch_size, ref_len, dtype=torch.bool)

        outputs = wrapper(
            input_ids=input_ids,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            labels=labels,
            timbre_ref_codes=timbre_ref_codes,
            timbre_ref_mask=timbre_ref_mask,
            target_position_mask=target_positions,
            source_prompt_positions=source_positions,
            timbre_ref_prompt_positions=timbre_positions,
            source_speaker_embedding_path=source_embedding_paths,
            timbre_ref_speaker_embedding_path=timbre_embedding_paths,
            target_speaker_embedding_path=target_embedding_paths,
        )
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            raise AssertionError("wrapper forward did not produce a finite loss")
        outputs.loss.backward()

        expected_shape = (batch_size, 4, hidden_size)
        if wrapper.last_timbre_memory_shape != expected_shape:
            raise AssertionError(f"T_ref shape mismatch: {wrapper.last_timbre_memory_shape} != {expected_shape}")
        expected_prosody_shape = (batch_size, 3, hidden_size)
        if wrapper.last_prosody_memory_shape != expected_prosody_shape:
            raise AssertionError(f"P_src shape mismatch: {wrapper.last_prosody_memory_shape} != {expected_prosody_shape}")
        if wrapper.last_route_loss is None:
            raise AssertionError("route loss was not computed")

        adapter_params = [
            (name, param)
            for name, param in wrapper.named_parameters()
            if (
                "timbre_memory" in name
                or "layer_adapters" in name
                or "role_router" in name
                or "source_prosody_encoder" in name
                or "target_head_router" in name
            )
        ]
        speaker_params = [(name, param) for name, param in wrapper.named_parameters() if "speaker_projection" in name]
        if not adapter_params:
            raise AssertionError("no timbre adapter parameters found")
        if not speaker_params:
            raise AssertionError("no speaker projection parameters found")
        if not all(param.requires_grad for _, param in adapter_params):
            raise AssertionError("some timbre adapter parameters are not trainable")
        if not all(param.requires_grad for _, param in speaker_params):
            raise AssertionError("some speaker projection parameters are not trainable")
        if not any(param.grad is not None and torch.count_nonzero(param.grad).item() > 0 for _, param in adapter_params):
            raise AssertionError("timbre/role adapter gradients are all zero")
        if wrapper.last_speaker_aux_loss is None:
            raise AssertionError("speaker auxiliary loss was not computed")
        if not any(param.grad is not None and torch.count_nonzero(param.grad).item() > 0 for _, param in speaker_params):
            raise AssertionError("speaker projection gradients are all zero")
        if any(param.requires_grad for name, param in wrapper.named_parameters() if name.startswith("model.")):
            raise AssertionError("base model trainability was changed by the wrapper")

        print(f"T_ref={wrapper.last_timbre_memory_shape}")
        print(f"P_src={wrapper.last_prosody_memory_shape}")
        print(f"adapter_trainable_params={sum(param.numel() for _, param in adapter_params)}")
        print(f"route_loss={wrapper.last_route_loss:.6f} route_stats={wrapper.last_route_stats}")
        print(f"speaker_aux_loss={wrapper.last_speaker_aux_loss:.4f}")
        print("Timbre memory smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
