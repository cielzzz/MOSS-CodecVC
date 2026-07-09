from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from moss_codecvc.io_utils import load_torch_file
from moss_codecvc.modes import VC_MODE_NO_TEXT, VC_MODE_TEXT, VC_NO_TEXT_PLACEHOLDER, normalize_vc_mode
from moss_codecvc.ref_prompt_permutation import permute_ref_prompt_codes
from moss_codecvc.roles import (
    REF_CODEC,
    SOURCE_CODEC,
    TARGET_CODEC,
    TEXT_OR_OTHER,
    build_role_ids,
    infer_prompt_role_ids_from_audio_spans,
    ref_speaker_prompt_slot_positions as find_ref_speaker_prompt_slot_positions,
)

from .auxiliary_losses import (
    AuxiliaryLossState,
    ContentCTCHead,
    ContentEmbeddingHead,
    ContentTokenHead,
    ProgressStopHead,
    ProsodyHead,
    SemanticFeatureHead,
    SourceCodecContentHead,
    compute_content_ctc_loss,
    compute_content_embedding_loss,
    compute_content_token_loss,
    compute_progress_stop_loss,
    compute_prosody_proxy_loss,
    compute_semantic_feature_loss,
    compute_source_codec_content_loss,
)
from .content_cross_attn import (
    ContentConformerEncoder,
    ContentCrossAttentionLayer,
    ContentPhonemeClassifierHead,
    compute_guided_attention_loss,
    compute_phoneme_classifier_loss,
)
from .role_routing import PerCodebookTargetHeadRouter, RoleCodecRouter, SourceProsodyEncoder
from .semantic_coverage import compute_semantic_attention_progress
from .speaker_encoder import build_frozen_speaker_encoder
from .speaker_cross_attn import SpeakerCrossAttentionLayer, SpeakerSequenceProjector, SpeakerTokenProjector
from .source_semantic_memory import (
    SourceCodecBottleneckMemoryEncoder,
    SourceSemanticAdapter,
    SourceSemanticMemoryEncoder,
    SourceTokenMemoryEncoder,
)
from .timbre_memory import ReferenceCodecTimbreMemory, TargetOnlyTimbreAdapter


@dataclass
class TimbreMemoryConfig:
    enabled: bool = False
    timbre_side_only: bool = False
    num_memory_tokens: int = 8
    adapter_layers: str | list[int] = "last_4"
    num_heads: int = 8
    adapter_dim: int = 256
    dropout: float = 0.0
    init_gate: float = -4.0
    encoder_type: str = "conformer"
    encoder_layers: int = 2
    conv_kernel_size: int = 7
    speaker_conditioning: bool = True
    target_speaker_similarity_weight: float = 0.0
    source_speaker_suppression_weight: float = 0.0
    speaker_embedding_dim: int = 0
    speaker_loss_margin: float = 0.0
    speaker_encoder_type: str = "embedding_loader"
    speaker_encoder_path: str | None = None
    freeze_speaker_encoder: bool = True
    ref_speaker_prompt_tokens: int = 0
    ref_speaker_prompt_dropout: float = 0.0
    ref_speaker_prompt_mode: str = "memory"
    ref_speaker_prompt_token_source: str = "speaker_mlp"
    ref_speaker_prompt_slot: bool = False
    ref_speaker_prompt_slot_code: int = -1
    ref_speaker_prompt_slot_pack_mode: str = "pad"
    ref_speaker_prompt_output_norm: bool = False
    ref_speaker_prompt_output_scale: float = 1.0
    ref_prompt_codec_permutation_enabled: bool = False
    ref_prompt_codec_permutation_min_seconds: float = 2.0
    ref_prompt_codec_permutation_max_seconds: float = 4.0
    ref_prompt_codec_permutation_frame_rate: float = 12.5
    ref_prompt_codec_permutation_seed: int = 1234
    ref_prompt_codec_permutation_mode: str = "shuffle"
    ref_prompt_codec_permutation_block_seconds: float = 0.4
    target_front_ce_weight: float = 1.0
    target_front_ce_seconds: float = 0.0
    target_front_ce_frame_rate: float = 12.5
    ref_speaker_adaln_weight: float = 0.0
    speaker_infonce_weight: float = 0.0
    speaker_infonce_temperature: float = 0.07
    speaker_infonce_negative_pool_size: int = 0
    speaker_infonce_negative_pool_seed: int = 1234
    speaker_condition_dropout: float = 0.0
    speaker_side_pathway_enabled: bool = False
    speaker_side_pathway_layers: str | list[int] = "all"
    speaker_side_pathway_kv_bias: bool = True
    speaker_side_pathway_gate_init: float = 0.0
    speaker_side_pathway_dropout: float = 0.15
    speaker_cross_attn_enabled: bool = False
    speaker_cross_attn_layers: str | list[int] = "all"
    speaker_cross_attn_tokens: int = 0
    speaker_cross_attn_gate_init: float = 0.0
    speaker_cross_attn_dropout: float = 0.0
    speaker_cross_attn_output_scale: float = 1.0
    speaker_cross_attn_token_init_std: float | None = None
    speaker_cross_attn_alpha_warmup_steps: int = 0
    speaker_cross_attn_runtime_scale_multiplier: float = 1.0
    speaker_cross_attn_source: str = "vector"
    speaker_cross_attn_seq_dim: int = 0
    use_perturbed_source_prompt: bool = False
    use_role_routing: bool = False
    route_loss_weight: float = 0.0
    prosody_memory_tokens: int = 8
    source_prosody_encoder_type: str = "conformer"
    source_prosody_encoder_layers: int = 2
    source_prosody_conv_kernel_size: int = 7
    source_prosody_no_text_gate: float = 1.0
    source_prosody_text_gate: float = 1.0
    target_head_routing: bool = False
    prosody_loss_weight: float = 0.0
    prosody_f0_weight: float = 1.0
    prosody_voiced_weight: float = 0.5
    prosody_energy_weight: float = 0.5
    prosody_pause_weight: float = 1.0
    prosody_duration_weight: float = 0.5
    prosody_normalize_f0: bool = True
    prosody_normalize_energy: bool = True
    content_loss_weight: float = 0.0
    content_embedding_dim: int = 0
    content_positive: str = "source"
    content_embedding_weight: float = 1.0
    content_ctc_weight: float = 0.0
    content_ctc_vocab_size: int = 0
    content_ctc_blank_id: int = 0
    content_ctc_token_offset: int = 1
    content_token_vocab_size: int = 0
    content_token_weight: float = 0.0
    content_source_codec_weight: float = 0.0
    content_source_codec_codebooks: str = "0,1,2,3"
    semantic_loss_weight: float = 0.0
    semantic_mode: str = "discrete"
    semantic_source: str = "source"
    semantic_vocab_size: int = 0
    semantic_feature_dim: int = 0
    semantic_feature_loss_type: str = "cosine"
    progress_loss_weight: float = 0.0
    stop_loss_weight: float = 0.0
    progress_num_bins: int = 32
    source_semantic_memory_enabled: bool = False
    source_semantic_feature_dim: int = 768
    source_semantic_adapter_layers: str | list[int] = "28,30,32,34,35"
    source_semantic_no_text_gate: float = 1.0
    source_semantic_text_gate: float = 0.0
    source_semantic_allow_learned_text_gate: bool = False
    source_semantic_progress_weight: float = 0.0
    source_semantic_dropout: float = 0.1
    source_semantic_init_gate: float = -2.0
    source_semantic_position_scale: float = 0.0
    source_semantic_monotonic_bias_strength: float = 0.0
    source_semantic_monotonic_bias_width: float = 0.25
    source_content_memory_type: str = "hubert_continuous"
    source_content_vocab_size: int = 0
    source_content_padding_id: int = 0
    source_content_codec_bottleneck_dim: int = 256
    source_content_codec_codebooks: str = "first_4"
    source_content_dedup_units: bool = False
    source_codec_residual_memory_weight: float = 0.0
    source_codec_residual_memory_detach: bool = False
    content_cross_attn_enabled: bool = False
    content_cross_attn_layers: str | list[int] = "all"
    content_cross_attn_feature_dim: int = 768
    content_cross_attn_gate_init: float = -0.5
    content_cross_attn_dropout: float = 0.0
    content_cross_attn_output_scale: float = 0.3
    content_encoder_layers: int = 2
    content_encoder_conv_kernel_size: int = 7
    guided_attn_loss_weight: float = 0.0
    guided_attn_warmup_steps: int = 1000
    guided_attn_band_frames: int = 3
    phoneme_classifier_loss_weight: float = 0.0
    ref_content_suppression_weight: float = 0.0
    ref_content_suppression_margin: float = 0.0
    ref_content_suppression_source: str = "auto"
    ref_content_suppression_detach_ref: bool = True


SOURCE_CONTINUOUS_MEMORY_TYPES = {
    "hubert_continuous",
    "asr_bnf_continuous",
    "wavlm_bnf_continuous",
    "wavlm_continuous",
}

SOURCE_MEMORY_TYPE_IDS = {
    "hubert_continuous": 1.0,
    "text_tokens": 2.0,
    "semantic_units": 3.0,
    "codec_bottleneck": 4.0,
    "asr_bnf_continuous": 5.0,
    "wavlm_bnf_continuous": 6.0,
    "wavlm_continuous": 7.0,
}

SPEAKER_SIDE_DECODER_BLOCK_COUNT = 32


def normalize_source_content_memory_type(value: str | None) -> str:
    memory_type = str(value or "hubert_continuous").strip().lower()
    aliases = {
        "hubert": "hubert_continuous",
        "continuous": "hubert_continuous",
        "ssl_continuous": "hubert_continuous",
        "asr_bnf": "asr_bnf_continuous",
        "bnf": "asr_bnf_continuous",
        "asr_continuous": "asr_bnf_continuous",
        "wavlm": "wavlm_bnf_continuous",
        "wavlm_bnf": "wavlm_bnf_continuous",
        "wavlm_ssl": "wavlm_continuous",
    }
    return aliases.get(memory_type, memory_type)


def is_continuous_source_content_memory_type(value: str | None) -> bool:
    return normalize_source_content_memory_type(value) in SOURCE_CONTINUOUS_MEMORY_TYPES


def speaker_side_decoder_layer_count(layers: Any) -> int:
    total = len(layers)
    if total > SPEAKER_SIDE_DECODER_BLOCK_COUNT:
        return SPEAKER_SIDE_DECODER_BLOCK_COUNT
    return total


def parse_adapter_layers(spec: str | list[int] | tuple[int, ...], num_layers: int) -> list[int]:
    if isinstance(spec, (list, tuple)):
        values = [int(item) for item in spec]
    else:
        value = str(spec).strip().lower()
        if value == "all":
            values = list(range(num_layers))
        elif value.startswith("last_"):
            count = int(value.split("_", 1)[1])
            if count <= 0:
                raise ValueError("last_N adapter layer count must be positive")
            values = list(range(max(0, num_layers - count), num_layers))
        elif value:
            values = [int(item.strip()) for item in value.split(",") if item.strip()]
        else:
            values = []
    normalized = []
    for idx in values:
        if idx < 0:
            idx = num_layers + idx
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"adapter layer index {idx} is outside [0, {num_layers})")
        if idx not in normalized:
            normalized.append(idx)
    if not normalized:
        raise ValueError("timbre adapter layers cannot be empty when timbre memory is enabled")
    return normalized


def parse_codebook_indices(spec: str | list[int] | tuple[int, ...], n_vq: int) -> list[int]:
    if isinstance(spec, (list, tuple)):
        values = [int(item) for item in spec]
    else:
        text = str(spec or "").strip().lower()
        if not text:
            return []
        if text.startswith("first_"):
            count = int(text.split("_", 1)[1])
            values = list(range(max(0, count)))
        else:
            values = [int(item.strip()) for item in text.split(",") if item.strip()]
    normalized = []
    for idx in values:
        if idx < 0:
            idx = int(n_vq) + idx
        if idx < 0 or idx >= int(n_vq):
            raise ValueError(f"codebook index {idx} is outside [0, {int(n_vq)})")
        if idx not in normalized:
            normalized.append(idx)
    return normalized


def _to_code_tensor(codes: Any, field_name: str) -> torch.Tensor:
    if torch.is_tensor(codes):
        tensor = codes.detach().cpu().long()
    else:
        tensor = torch.tensor(codes, dtype=torch.long)
    if tensor.dim() != 2:
        raise ValueError(f"{field_name} must be [T, n_vq], got {tuple(tensor.shape)}")
    return tensor


def _audio_activity_mask(input_ids: torch.Tensor, audio_pad_code: int) -> torch.Tensor:
    return (input_ids[:, 1:] != int(audio_pad_code)).any(dim=-1)


def _contiguous_spans(mask: torch.Tensor) -> list[tuple[int, int]]:
    values = mask.detach().cpu().bool().tolist()
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, active in enumerate(values):
        if active and start is None:
            start = idx
        if start is not None and (not active or idx == len(values) - 1):
            end = idx if not active else idx + 1
            spans.append((start, end))
            start = None
    return spans


def _as_path_list(paths, batch_size: int) -> list[str | None] | None:
    if paths is None:
        return None
    if isinstance(paths, (list, tuple)):
        values = list(paths)
    else:
        values = [paths]
    if len(values) == 1 and batch_size > 1:
        values = values * batch_size
    values = values[:batch_size]
    if len(values) < batch_size:
        values.extend([None] * (batch_size - len(values)))
    return values


def _record_audio_path(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if value:
        return str(value)
    meta = record.get("moss_codecvc_meta")
    if isinstance(meta, dict) and meta.get(key):
        return str(meta[key])
    return None


def _record_path(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if value:
        return str(value)
    meta = record.get("moss_codecvc_meta")
    if isinstance(meta, dict) and meta.get(key):
        return str(meta[key])
    return None


def _record_value(record: dict[str, Any], key: str) -> Any | None:
    if key in record and record[key] is not None:
        return record[key]
    meta = record.get("moss_codecvc_meta")
    if isinstance(meta, dict) and key in meta and meta[key] is not None:
        return meta[key]
    return None


def _record_bool(record: dict[str, Any], key: str, *, default: bool = True) -> bool:
    value = _record_value(record, key)
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "keep"}:
        return True
    if text in {"0", "false", "no", "n", "drop", "filtered"}:
        return False
    return bool(default)


def _load_feature_payload(path: str | None) -> Any | None:
    if not path:
        return None
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise FileNotFoundError(f"feature path does not exist: {path_obj}")
    suffix = path_obj.suffix.lower()
    if suffix == ".npy":
        import numpy as np

        return np.load(path_obj)
    if suffix == ".npz":
        import numpy as np

        return dict(np.load(path_obj))
    return load_torch_file(path_obj)


def _feature_get(payload: Any, keys: tuple[str, ...]) -> Any | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return value
        return None
    return payload


def _optional_1d_tensor(payload: Any, *keys: str) -> torch.Tensor | None:
    value = _feature_get(payload, tuple(keys))
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 0:
        return None
    return tensor


def _optional_scalar_tensor(payload: Any, *keys: str) -> torch.Tensor | None:
    value = _feature_get(payload, tuple(keys))
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 0:
        return None
    return tensor[:1]


def _load_prosody_bundle(path: str | None, prefix: str) -> dict[str, torch.Tensor | None]:
    payload = _load_feature_payload(path)
    if payload is None:
        return {}
    return {
        f"{prefix}_logf0": _optional_1d_tensor(payload, f"{prefix}_logf0", "logf0", "f0_log"),
        f"{prefix}_voiced_mask": _optional_1d_tensor(payload, f"{prefix}_voiced_mask", "voiced_mask", "voiced"),
        f"{prefix}_energy": _optional_1d_tensor(payload, f"{prefix}_energy", "energy", "log_energy", "rms"),
        f"{prefix}_pause_mask": _optional_1d_tensor(payload, f"{prefix}_pause_mask", "pause_mask", "silence_mask"),
        f"{prefix}_duration": _optional_scalar_tensor(
            payload,
            f"{prefix}_duration",
            f"{prefix}_duration_sec",
            "duration",
            "duration_sec",
        ),
    }


def _load_content_embedding(path: str | None) -> torch.Tensor | None:
    payload = _load_feature_payload(path)
    if payload is None:
        return None
    value = _feature_get(payload, ("content_embedding", "embedding", "semantic_embedding", "source_content_embedding"))
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() > 1:
        tensor = tensor.reshape(-1, tensor.shape[-1]).mean(dim=0)
    tensor = tensor.flatten()
    if tensor.numel() == 0:
        return None
    return tensor


def _content_ids_from_value(value: Any | None) -> torch.Tensor | None:
    if value is None or isinstance(value, str):
        return None
    tensor = torch.as_tensor(value, dtype=torch.long).flatten()
    if tensor.numel() == 0:
        return None
    return tensor


def _load_content_ids(path: str | None) -> torch.Tensor | None:
    payload = _load_feature_payload(path)
    if payload is None:
        return None
    value = _feature_get(
        payload,
        (
            "content_ids",
            "semantic_tokens",
            "semantic_ids",
            "unit_ids",
            "units",
            "source_content_ids",
            "target_content_ids",
        ),
    )
    return _content_ids_from_value(value)


def _load_semantic_features(path: str | None) -> torch.Tensor | None:
    payload = _load_feature_payload(path)
    if payload is None:
        return None
    value = _feature_get(
        payload,
        (
            "semantic_features",
            "features",
            "hidden_states",
            "source_semantic_features",
            "target_semantic_features",
            "source_asr_bnf_features",
            "target_asr_bnf_features",
            "asr_bnf_features",
            "bnf_features",
            "source_wavlm_bnf_features",
            "target_wavlm_bnf_features",
            "wavlm_bnf_features",
            "hubert_features",
            "wavlm_features",
        ),
    )
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.dim() != 2 or tensor.numel() == 0:
        return None
    return tensor


def _load_speaker_sequence_features(path: str | None) -> torch.Tensor | None:
    payload = _load_feature_payload(path)
    if payload is None:
        return None
    value = _feature_get(
        payload,
        (
            "speaker_seq_features",
            "speaker_sequence_features",
            "speaker_features",
            "features",
            "hidden_states",
            "wavlm_features",
            "wavlm_hidden_states",
        ),
    )
    if value is None:
        return None
    return _feature_matrix_from_value(value)


def _feature_matrix_from_value(value: Any | None) -> torch.Tensor | None:
    if value is None or isinstance(value, str):
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.dim() != 2 or tensor.numel() == 0:
        return None
    return tensor


def _pad_optional_1d(items: list[dict[str, Any]], key: str, *, padding_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor] | None:
    values = [item.get(key) for item in items]
    if not any(value is not None for value in values):
        return None
    max_len = max(int(value.numel()) for value in values if value is not None)
    padded = []
    masks = []
    for value in values:
        if value is None:
            padded.append(torch.full((max_len,), float(padding_value), dtype=torch.float32))
            masks.append(torch.zeros((max_len,), dtype=torch.bool))
            continue
        cur = value.float().flatten()
        pad_len = max_len - int(cur.numel())
        padded.append(F.pad(cur, (0, pad_len), value=float(padding_value)))
        masks.append(F.pad(torch.ones_like(cur, dtype=torch.bool), (0, pad_len), value=False))
    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)


def _pad_optional_long_1d(items: list[dict[str, Any]], key: str, *, padding_value: int = 0) -> tuple[torch.Tensor, torch.Tensor] | None:
    values = [item.get(key) for item in items]
    if not any(value is not None for value in values):
        return None
    max_len = max(int(value.numel()) for value in values if value is not None)
    padded = []
    masks = []
    for value in values:
        if value is None:
            padded.append(torch.full((max_len,), int(padding_value), dtype=torch.long))
            masks.append(torch.zeros((max_len,), dtype=torch.bool))
            continue
        cur = value.long().flatten()
        pad_len = max_len - int(cur.numel())
        padded.append(F.pad(cur, (0, pad_len), value=int(padding_value)))
        masks.append(F.pad(torch.ones_like(cur, dtype=torch.bool), (0, pad_len), value=False))
    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)


def _pad_optional_2d(items: list[dict[str, Any]], key: str, *, padding_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor] | None:
    values = [item.get(key) for item in items]
    if not any(value is not None for value in values):
        return None
    first = next(value for value in values if value is not None)
    feature_dim = int(first.shape[-1])
    max_len = max(int(value.shape[0]) for value in values if value is not None)
    padded = []
    masks = []
    for value in values:
        if value is None:
            padded.append(torch.full((max_len, feature_dim), float(padding_value), dtype=torch.float32))
            masks.append(torch.zeros((max_len,), dtype=torch.bool))
            continue
        cur = value.float()
        if cur.dim() != 2 or int(cur.shape[-1]) != feature_dim:
            raise ValueError(f"{key} feature dimension mismatch in batch: got {tuple(cur.shape)}, expected D={feature_dim}")
        pad_len = max_len - int(cur.shape[0])
        padded.append(F.pad(cur, (0, 0, 0, pad_len), value=float(padding_value)))
        masks.append(F.pad(torch.ones((cur.shape[0],), dtype=torch.bool), (0, pad_len), value=False))
    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)


def _stack_optional_vectors(items: list[dict[str, Any]], key: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    values = [item.get(key) for item in items]
    if not any(value is not None for value in values):
        return None
    dim = next(int(value.numel()) for value in values if value is not None)
    stacked = []
    valid = []
    for value in values:
        if value is None:
            stacked.append(torch.zeros(dim, dtype=torch.float32))
            valid.append(False)
            continue
        cur = value.float().flatten()
        if int(cur.numel()) != dim:
            raise ValueError(f"{key} dimension mismatch in batch: got {int(cur.numel())}, expected {dim}")
        stacked.append(cur)
        valid.append(True)
    return torch.stack(stacked, dim=0), torch.tensor(valid, dtype=torch.bool)


class MossCodecVCTimbreSFTDataset:
    """Adds role metadata and S2 codec memory inputs around upstream MossTTSSFTDataset."""

    MODE_TO_ID = {
        VC_MODE_TEXT: 1,
        VC_MODE_NO_TEXT: 2,
    }

    def __init__(
        self,
        records,
        base_dataset,
        *,
        n_vq: int,
        audio_pad_code: int,
        content_tokenizer: Any | None = None,
        content_ctc_token_offset: int = 1,
        timbre_side_only: bool = False,
        use_perturbed_source_prompt: bool = False,
        ref_speaker_prompt_slot: bool = False,
        ref_speaker_prompt_tokens: int = 0,
        ref_speaker_prompt_slot_code: int = -1,
        ref_speaker_prompt_slot_pack_mode: str = "pad",
        ref_prompt_codec_permutation_enabled: bool = False,
        ref_prompt_codec_permutation_min_seconds: float = 2.0,
        ref_prompt_codec_permutation_max_seconds: float = 4.0,
        ref_prompt_codec_permutation_frame_rate: float = 12.5,
        ref_prompt_codec_permutation_mode: str = "shuffle",
        ref_prompt_codec_permutation_block_seconds: float = 0.4,
        speaker_side_pathway_enabled: bool = False,
    ) -> None:
        self.records = records
        self.base_dataset = base_dataset
        self.n_vq = int(n_vq)
        self.audio_pad_code = int(audio_pad_code)
        self.content_tokenizer = content_tokenizer
        self.content_ctc_token_offset = int(content_ctc_token_offset)
        self.timbre_side_only = bool(timbre_side_only)
        self.use_perturbed_source_prompt = bool(use_perturbed_source_prompt)
        self.ref_speaker_prompt_slot = bool(ref_speaker_prompt_slot)
        self.ref_speaker_prompt_tokens = max(0, int(ref_speaker_prompt_tokens))
        self.ref_speaker_prompt_slot_code = int(ref_speaker_prompt_slot_code)
        self.ref_speaker_prompt_slot_pack_mode = str(ref_speaker_prompt_slot_pack_mode or "pad").strip().lower()
        self.ref_prompt_codec_permutation_enabled = bool(ref_prompt_codec_permutation_enabled)
        self.ref_prompt_codec_permutation_min_seconds = max(0.0, float(ref_prompt_codec_permutation_min_seconds))
        self.ref_prompt_codec_permutation_max_seconds = max(
            self.ref_prompt_codec_permutation_min_seconds,
            float(ref_prompt_codec_permutation_max_seconds),
        )
        self.ref_prompt_codec_permutation_frame_rate = max(1.0e-6, float(ref_prompt_codec_permutation_frame_rate))
        self.ref_prompt_codec_permutation_mode = str(ref_prompt_codec_permutation_mode or "shuffle")
        self.ref_prompt_codec_permutation_block_seconds = max(0.0, float(ref_prompt_codec_permutation_block_seconds))
        self.speaker_side_pathway_enabled = bool(speaker_side_pathway_enabled)
        model_config = getattr(getattr(base_dataset, "processor", None), "model_config", None)
        self.audio_start_token_id = int(getattr(model_config, "audio_start_token_id", 0) or 0)
        self.audio_end_token_id = int(getattr(model_config, "audio_end_token_id", 0) or 0)
        self.audio_user_slot_token_id = int(getattr(model_config, "audio_user_slot_token_id", 0) or 0)
        self.audio_gen_slot_token_id = int(getattr(model_config, "audio_assistant_gen_slot_token_id", 0) or 0)

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize_content_text(self, text: str | None) -> torch.Tensor | None:
        if not text or self.content_tokenizer is None:
            return None
        normalized = str(text).strip()
        if not normalized or normalized in {VC_NO_TEXT_PLACEHOLDER, "None", "none", "null"}:
            return None
        if hasattr(self.content_tokenizer, "encode"):
            try:
                ids = self.content_tokenizer.encode(normalized, add_special_tokens=False)
            except TypeError:
                ids = self.content_tokenizer.encode(normalized, out_type=int)
        else:
            encoded = self.content_tokenizer(normalized, add_special_tokens=False)
            ids = encoded.get("input_ids", [])
        if not ids:
            return None
        tensor = torch.as_tensor(ids, dtype=torch.long).flatten()
        if self.content_ctc_token_offset:
            tensor = tensor + int(self.content_ctc_token_offset)
        return tensor

    def _permuted_timbre_prompt_codes(self, timbre_ref_codes: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
        prompt_codes, stats = permute_ref_prompt_codes(
            timbre_ref_codes,
            enabled=bool(self.ref_prompt_codec_permutation_enabled),
            min_seconds=float(self.ref_prompt_codec_permutation_min_seconds),
            max_seconds=float(self.ref_prompt_codec_permutation_max_seconds),
            frame_rate=float(self.ref_prompt_codec_permutation_frame_rate),
            seed=None,
            mode=str(self.ref_prompt_codec_permutation_mode),
            block_seconds=float(self.ref_prompt_codec_permutation_block_seconds),
        )
        return prompt_codes, stats.as_dict()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        references = record.get("reference_audio_codes") or record.get("ref_audio_codes")
        if not references or len(references) < 2 or references[1] is None:
            raise ValueError("Timbre memory requires reference_audio_codes[1] for S2 target timbre.")
        source_ref_codes_value = references[0]
        if self.use_perturbed_source_prompt:
            perturbed_value = _record_value(record, "source_prompt_perturbed_codes")
            if perturbed_value is None:
                perturbed_value = _record_value(record, "source_prompt_perturbed_audio_codes")
            if perturbed_value is not None:
                source_ref_codes_value = perturbed_value
            perturbed_codes_path = _record_path(record, "source_prompt_perturbed_codes_path")
            if perturbed_codes_path:
                payload = load_torch_file(perturbed_codes_path)
                source_ref_codes_value = payload.get("codes", payload)
        source_ref_codes = _to_code_tensor(source_ref_codes_value, "reference_audio_codes[0]")
        if source_ref_codes.shape[1] != self.n_vq:
            raise ValueError(f"S1 source codes n_vq={source_ref_codes.shape[1]} does not match expected {self.n_vq}")
        timbre_ref_codes = _to_code_tensor(references[1], "reference_audio_codes[1]")
        if timbre_ref_codes.shape[1] != self.n_vq:
            raise ValueError(f"S2 timbre codes n_vq={timbre_ref_codes.shape[1]} does not match expected {self.n_vq}")
        timbre_prompt_codes, timbre_prompt_permutation_stats = self._permuted_timbre_prompt_codes(timbre_ref_codes)

        slot_codes = None
        if self.ref_speaker_prompt_slot and self.ref_speaker_prompt_tokens > 0:
            slot_fill_value = self.audio_pad_code if self.ref_speaker_prompt_slot_code < 0 else self.ref_speaker_prompt_slot_code
            slot_codes = torch.full(
                (self.ref_speaker_prompt_tokens, self.n_vq),
                slot_fill_value,
                dtype=torch.long,
            )
            if self.ref_speaker_prompt_slot_pack_mode in {"audio_like", "audio-like", "active"}:
                slot_codes.fill_(self.audio_pad_code)
                slot_codes[:, 0] = int(0 if self.ref_speaker_prompt_slot_code < 0 else self.ref_speaker_prompt_slot_code)

        pack_record = record
        if (
            self.timbre_side_only
            or self.speaker_side_pathway_enabled
            or self.use_perturbed_source_prompt
            or slot_codes is not None
            or self.ref_prompt_codec_permutation_enabled
        ):
            pack_record = dict(record)
            if slot_codes is not None:
                pack_record["reference_audio_codes"] = [source_ref_codes, slot_codes]
            else:
                pack_record["reference_audio_codes"] = (
                    [source_ref_codes]
                    if (self.timbre_side_only or self.speaker_side_pathway_enabled)
                    else [source_ref_codes, timbre_prompt_codes]
                )
            pack_record.pop("ref_audio_codes", None)
        if (self.timbre_side_only or self.speaker_side_pathway_enabled) and slot_codes is None:
            for path_key in ("reference", "reference_audio", "ref_audio"):
                if path_key in pack_record:
                    pack_record.pop(path_key, None)

        if hasattr(self.base_dataset, "_pack_record"):
            item = dict(self.base_dataset._pack_record(pack_record))
        else:
            item = dict(self.base_dataset[index])

        full_input_ids = item["input_ids"]
        input_len = full_input_ids.shape[0] - 1
        source_positions = torch.zeros(input_len, dtype=torch.bool)
        timbre_positions = torch.zeros(input_len, dtype=torch.bool)
        active_spans = _contiguous_spans(_audio_activity_mask(full_input_ids, self.audio_pad_code))
        if len(active_spans) >= 1:
            start, end = active_spans[0]
            source_positions[start : min(end, input_len)] = True
        if len(active_spans) >= 2:
            start, end = active_spans[1]
            timbre_positions[start : min(end, input_len)] = True
        slot_positions = torch.zeros(input_len, dtype=torch.bool)
        if slot_codes is not None:
            slot_mask = find_ref_speaker_prompt_slot_positions(
                full_input_ids[:input_len].unsqueeze(0),
                audio_start_token_id=self.audio_start_token_id,
                audio_end_token_id=self.audio_end_token_id,
                audio_gen_slot_token_id=(self.audio_user_slot_token_id, self.audio_gen_slot_token_id),
                token_count=self.ref_speaker_prompt_tokens,
                occurrence=2,
            )[0].cpu()
            slot_positions[: min(input_len, int(slot_mask.shape[0]))] = slot_mask[:input_len]
            timbre_positions |= slot_positions

        mode = record.get("moss_codecvc_mode") or record.get("mode")
        if mode is None:
            text_value = str(record.get("text") or "").strip()
            sample_id = str(record.get("sample_id") or "")
            if not text_value or text_value == VC_NO_TEXT_PLACEHOLDER or sample_id.endswith(":vc_no_text"):
                mode = VC_MODE_NO_TEXT
            else:
                mode = VC_MODE_TEXT
        try:
            mode_id = self.MODE_TO_ID[normalize_vc_mode(mode)]
        except ValueError:
            mode_id = 0

        source_prosody = _load_prosody_bundle(_record_path(record, "source_prosody_path"), "source")
        target_prosody = _load_prosody_bundle(_record_path(record, "target_prosody_path"), "target")
        source_content_embedding = _load_content_embedding(
            _record_path(record, "source_content_path") or _record_path(record, "source_content_embedding_path")
        )
        target_content_embedding = _load_content_embedding(
            _record_path(record, "target_content_path") or _record_path(record, "target_content_embedding_path")
        )
        source_content_ids = _load_content_ids(
            _record_path(record, "source_content_ids_path")
            or _record_path(record, "source_semantic_tokens_path")
            or _record_path(record, "source_content_tokens_path")
        )
        if source_content_ids is None:
            source_content_ids = _content_ids_from_value(_record_value(record, "source_content_ids"))
        if source_content_ids is None:
            source_content_ids = _content_ids_from_value(_record_value(record, "source_semantic_tokens"))
        target_content_ids = _load_content_ids(
            _record_path(record, "target_content_ids_path")
            or _record_path(record, "target_semantic_tokens_path")
            or _record_path(record, "target_content_tokens_path")
        )
        if target_content_ids is None:
            target_content_ids = _content_ids_from_value(_record_value(record, "target_content_ids"))
        if target_content_ids is None:
            target_content_ids = _content_ids_from_value(_record_value(record, "target_semantic_tokens"))
        content_ref_text = _record_value(record, "content_ref_text")
        if not isinstance(content_ref_text, str):
            content_ref_text = None
        content_ctc_allowed = _record_bool(record, "content_keep", default=True) and _record_bool(
            record,
            "content_token_keep",
            default=True,
        )
        content_token_ids = _load_content_ids(
            _record_path(record, "content_token_ids_path")
            or _record_path(record, "content_tokens_path")
            or _record_path(record, "content_ref_token_ids_path")
        ) if content_ctc_allowed else None
        if content_ctc_allowed:
            if content_token_ids is None:
                content_token_ids = _content_ids_from_value(_record_value(record, "content_token_ids"))
            if content_token_ids is None:
                content_token_ids = _content_ids_from_value(_record_value(record, "content_ref_token_ids"))
            if content_token_ids is None:
                content_token_ids = self._tokenize_content_text(content_ref_text)

        source_semantic_units = _load_content_ids(
            _record_path(record, "source_semantic_units_path")
            or _record_path(record, "source_semantic_tokens_path")
            or _record_path(record, "source_semantic_ids_path")
        )
        if source_semantic_units is None:
            source_semantic_units = _content_ids_from_value(_record_value(record, "source_semantic_units"))
        if source_semantic_units is None:
            source_semantic_units = _content_ids_from_value(_record_value(record, "source_semantic_tokens"))
        target_semantic_units = _load_content_ids(
            _record_path(record, "target_semantic_units_path")
            or _record_path(record, "target_semantic_tokens_path")
            or _record_path(record, "target_semantic_ids_path")
        )
        if target_semantic_units is None:
            target_semantic_units = _content_ids_from_value(_record_value(record, "target_semantic_units"))
        if target_semantic_units is None:
            target_semantic_units = _content_ids_from_value(_record_value(record, "target_semantic_tokens"))

        source_semantic_features = _load_semantic_features(
            _record_path(record, "source_asr_bnf_feature_path")
            or _record_path(record, "source_asr_bnf_features_path")
            or _record_path(record, "source_bnf_feature_path")
            or _record_path(record, "source_bnf_features_path")
            or _record_path(record, "source_wavlm_bnf_feature_path")
            or _record_path(record, "source_wavlm_bnf_features_path")
            or _record_path(record, "source_wavlm_features_path")
            or _record_path(record, "source_wavlm_feature_path")
            or _record_path(record, "source_semantic_feature_path")
            or _record_path(record, "source_semantic_features_path")
            or _record_path(record, "source_hubert_features_path")
            or _record_path(record, "source_hubert_feature_path")
        )
        for value_key in (
            "source_asr_bnf_features",
            "source_bnf_features",
            "source_wavlm_bnf_features",
            "source_wavlm_features",
            "source_semantic_features",
            "source_hubert_features",
        ):
            if source_semantic_features is None:
                source_semantic_features = _feature_matrix_from_value(_record_value(record, value_key))
        target_semantic_features = _load_semantic_features(
            _record_path(record, "target_asr_bnf_feature_path")
            or _record_path(record, "target_asr_bnf_features_path")
            or _record_path(record, "target_bnf_feature_path")
            or _record_path(record, "target_bnf_features_path")
            or _record_path(record, "target_wavlm_bnf_feature_path")
            or _record_path(record, "target_wavlm_bnf_features_path")
            or _record_path(record, "target_wavlm_features_path")
            or _record_path(record, "target_wavlm_feature_path")
            or _record_path(record, "target_semantic_feature_path")
            or _record_path(record, "target_semantic_features_path")
            or _record_path(record, "teacher_target_semantic_feature_path")
            or _record_path(record, "teacher_target_semantic_features_path")
            or _record_path(record, "target_hubert_features_path")
            or _record_path(record, "target_hubert_feature_path")
        )
        for value_key in (
            "target_asr_bnf_features",
            "target_bnf_features",
            "target_wavlm_bnf_features",
            "target_wavlm_features",
            "target_semantic_features",
            "target_hubert_features",
        ):
            if target_semantic_features is None:
                target_semantic_features = _feature_matrix_from_value(_record_value(record, value_key))
        speaker_seq_path = (
            _record_path(record, "speaker_seq_path")
            or _record_path(record, "timbre_ref_speaker_seq_path")
            or _record_path(record, "speaker_seq_features_path")
            or _record_path(record, "timbre_ref_speaker_seq_features_path")
        )
        speaker_seq_features = _load_speaker_sequence_features(speaker_seq_path)
        for value_key in (
            "speaker_seq_features",
            "timbre_ref_speaker_seq_features",
            "speaker_sequence_features",
            "speaker_features",
        ):
            if speaker_seq_features is None:
                speaker_seq_features = _feature_matrix_from_value(_record_value(record, value_key))

        item.update(
            {
                "source_ref_codes": source_ref_codes,
                "timbre_ref_codes": timbre_ref_codes,
                "timbre_ref_prompt_permutation": torch.tensor(
                    [
                        int(timbre_prompt_permutation_stats["enabled"]),
                        int(timbre_prompt_permutation_stats["source_frames"]),
                        int(timbre_prompt_permutation_stats["prompt_frames"]),
                        int(timbre_prompt_permutation_stats["start"]),
                        int(timbre_prompt_permutation_stats["shuffled"]),
                    ],
                    dtype=torch.long,
                ),
                "source_prompt_positions": source_positions,
                "timbre_ref_prompt_positions": timbre_positions,
                "ref_speaker_prompt_slot_positions": slot_positions,
                "vc_mode_id": torch.tensor(mode_id, dtype=torch.long),
                "source_speaker_embedding_path": record.get("source_speaker_embedding_path"),
                "timbre_ref_speaker_embedding_path": record.get("timbre_ref_speaker_embedding_path"),
                "target_speaker_embedding_path": record.get("target_speaker_embedding_path"),
                "speaker_vec_path": _record_path(record, "speaker_vec_path")
                or _record_path(record, "timbre_ref_speaker_vec_path"),
                "source_speaker_audio_path": _record_audio_path(record, "source_audio"),
                "timbre_ref_speaker_audio_path": _record_audio_path(record, "timbre_ref_audio"),
                "target_speaker_audio_path": _record_audio_path(record, "target_audio"),
                "speaker_seq_path": speaker_seq_path,
                "speaker_seq_features": speaker_seq_features,
                "source_content_embedding": source_content_embedding,
                "target_content_embedding": target_content_embedding,
                "source_content_ids": source_content_ids,
                "target_content_ids": target_content_ids,
                "content_ref_text": content_ref_text,
                "content_token_ids": content_token_ids,
                "source_semantic_units": source_semantic_units,
                "target_semantic_units": target_semantic_units,
                "source_semantic_features": source_semantic_features,
                "target_semantic_features": target_semantic_features,
            }
        )
        item.update(source_prosody)
        item.update(target_prosody)
        return item

    @staticmethod
    def _left_pad_bool(values: list[torch.Tensor], *, padding_value: bool = False) -> torch.Tensor:
        max_len = max(int(value.shape[0]) for value in values)
        padded = []
        for value in values:
            pad_len = max_len - int(value.shape[0])
            padded.append(torch.nn.functional.pad(value.bool(), (pad_len, 0), value=padding_value))
        return torch.stack(padded, dim=0)

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        base_batch = self.base_dataset.collate_fn(batch)
        source_codes = [item["source_ref_codes"] for item in batch]
        source_ref_codes = pad_sequence(
            source_codes,
            batch_first=True,
            padding_value=self.audio_pad_code,
            padding_side="right",
        ).long()
        source_ref_mask = pad_sequence(
            [torch.ones(code.shape[0], dtype=torch.bool) for code in source_codes],
            batch_first=True,
            padding_value=False,
            padding_side="right",
        ).bool()
        timbre_codes = [item["timbre_ref_codes"] for item in batch]
        timbre_ref_codes = pad_sequence(
            timbre_codes,
            batch_first=True,
            padding_value=self.audio_pad_code,
            padding_side="right",
        ).long()
        timbre_ref_mask = pad_sequence(
            [torch.ones(code.shape[0], dtype=torch.bool) for code in timbre_codes],
            batch_first=True,
            padding_value=False,
            padding_side="right",
        ).bool()
        source_positions = self._left_pad_bool([item["source_prompt_positions"] for item in batch])
        timbre_positions = self._left_pad_bool([item["timbre_ref_prompt_positions"] for item in batch])
        slot_positions = self._left_pad_bool([item["ref_speaker_prompt_slot_positions"] for item in batch])
        target_positions = (base_batch["labels"] != -100).any(dim=-1)
        source_positions = source_positions[:, -base_batch["input_ids"].shape[1] :].contiguous()
        timbre_positions = timbre_positions[:, -base_batch["input_ids"].shape[1] :].contiguous()
        slot_positions = slot_positions[:, -base_batch["input_ids"].shape[1] :].contiguous()
        role_ids = torch.full(
            base_batch["input_ids"].shape[:2],
            TEXT_OR_OTHER,
            dtype=torch.long,
        )
        role_ids[source_positions] = SOURCE_CODEC
        role_ids[timbre_positions] = REF_CODEC
        role_ids[target_positions] = TARGET_CODEC
        base_batch.update(
            {
                "timbre_ref_codes": timbre_ref_codes,
                "timbre_ref_mask": timbre_ref_mask,
                "timbre_ref_prompt_permutation": torch.stack(
                    [item["timbre_ref_prompt_permutation"] for item in batch],
                    dim=0,
                ),
                "source_ref_codes": source_ref_codes,
                "source_ref_mask": source_ref_mask,
                "source_prompt_positions": source_positions,
                "timbre_ref_prompt_positions": timbre_positions,
                "ref_speaker_prompt_slot_positions": slot_positions,
                "target_assistant_positions": target_positions.contiguous(),
                "role_ids": role_ids.contiguous(),
                "vc_mode_id": torch.stack([item["vc_mode_id"] for item in batch], dim=0),
                "source_speaker_embedding_path": [item.get("source_speaker_embedding_path") for item in batch],
                "timbre_ref_speaker_embedding_path": [item.get("timbre_ref_speaker_embedding_path") for item in batch],
                "target_speaker_embedding_path": [item.get("target_speaker_embedding_path") for item in batch],
                "speaker_vec_path": [item.get("speaker_vec_path") for item in batch],
                "speaker_seq_path": [item.get("speaker_seq_path") for item in batch],
                "source_speaker_audio_path": [item.get("source_speaker_audio_path") for item in batch],
                "timbre_ref_speaker_audio_path": [item.get("timbre_ref_speaker_audio_path") for item in batch],
                "target_speaker_audio_path": [item.get("target_speaker_audio_path") for item in batch],
            }
        )
        for key in (
            "source_logf0",
            "source_voiced_mask",
            "source_energy",
            "source_pause_mask",
            "target_logf0",
            "target_voiced_mask",
            "target_energy",
            "target_pause_mask",
        ):
            padded = _pad_optional_1d(batch, key)
            if padded is not None:
                base_batch[key], base_batch[f"{key}_mask"] = padded
        for key in ("source_duration", "target_duration"):
            stacked = _stack_optional_vectors(batch, key)
            if stacked is not None:
                values, mask = stacked
                base_batch[key] = values.squeeze(-1)
                base_batch[f"{key}_mask"] = mask
        for key in ("source_content_embedding", "target_content_embedding"):
            stacked = _stack_optional_vectors(batch, key)
            if stacked is not None:
                base_batch[key], base_batch[f"{key}_mask"] = stacked
        for key in ("source_content_ids", "target_content_ids"):
            padded_ids = _pad_optional_long_1d(batch, key)
            if padded_ids is not None:
                base_batch[key], base_batch[f"{key}_mask"] = padded_ids
        padded_content_tokens = _pad_optional_long_1d(batch, "content_token_ids")
        if padded_content_tokens is not None:
            base_batch["content_token_ids"], base_batch["content_token_ids_mask"] = padded_content_tokens
        for key in ("source_semantic_units", "target_semantic_units"):
            padded_ids = _pad_optional_long_1d(batch, key)
            if padded_ids is not None:
                base_batch[key], base_batch[f"{key}_mask"] = padded_ids
        for key in ("source_semantic_features", "target_semantic_features"):
            padded_features = _pad_optional_2d(batch, key)
            if padded_features is not None:
                base_batch[key], base_batch[f"{key}_mask"] = padded_features
        speaker_seq = _pad_optional_2d(batch, "speaker_seq_features")
        if speaker_seq is not None:
            base_batch["speaker_seq_features"], base_batch["speaker_seq_features_mask"] = speaker_seq
        return base_batch


class MossCodecVCTimbreMemoryWrapper(nn.Module):
    """Training wrapper that injects S2 codec timbre memory into selected target positions."""

    MODE_TO_ID = {
        VC_MODE_TEXT: 1,
        VC_MODE_NO_TEXT: 2,
    }

    def __init__(self, model: nn.Module, config: TimbreMemoryConfig) -> None:
        super().__init__()
        self.model = model
        self.timbre_memory_config = config
        self.peft_adapter_fallback_directory: Path | None = None
        base_model = self.get_base_model()
        self.config = getattr(base_model, "config", getattr(model, "config", None))
        if self.config is None:
            raise ValueError("Wrapped model must expose a config.")
        hidden_size = int(self.config.language_config.hidden_size)
        layers = getattr(base_model.language_model, "layers", None)
        if layers is None:
            raise ValueError("Wrapped MossTTSDelayModel must expose language_model.layers.")
        self.legacy_timbre_memory_enabled = int(config.num_memory_tokens) > 0 and bool(str(config.adapter_layers).strip())
        self.timbre_memory: ReferenceCodecTimbreMemory | None = None
        self.adapter_layer_indices: list[int] = []
        self.layer_adapters = nn.ModuleDict()
        if self.legacy_timbre_memory_enabled:
            self.timbre_memory = ReferenceCodecTimbreMemory(
                hidden_size=hidden_size,
                num_memory_tokens=config.num_memory_tokens,
                num_heads=config.num_heads,
                adapter_dim=config.adapter_dim,
                dropout=config.dropout,
                encoder_type=config.encoder_type,
                encoder_layers=config.encoder_layers,
                conv_kernel_size=config.conv_kernel_size,
                speaker_embedding_dim=config.speaker_embedding_dim,
                speaker_conditioning=config.speaker_conditioning,
            )
            self.adapter_layer_indices = parse_adapter_layers(config.adapter_layers, len(layers))
            self.layer_adapters = nn.ModuleDict(
                {
                    str(idx): TargetOnlyTimbreAdapter(
                        hidden_size=hidden_size,
                        num_heads=config.num_heads,
                        adapter_dim=config.adapter_dim,
                        dropout=config.dropout,
                        init_gate=config.init_gate,
                    )
                    for idx in self.adapter_layer_indices
                }
            )
        self.speaker_encoder = build_frozen_speaker_encoder(
            config.speaker_encoder_type,
            encoder_path=config.speaker_encoder_path,
            embedding_dim=config.speaker_embedding_dim if config.speaker_embedding_dim > 0 else None,
        )
        self.speaker_side_layer_indices: list[int] = []
        self.speaker_side_adaln: nn.ModuleDict | None = None
        self.speaker_side_kv_bias: nn.ModuleDict | None = None
        self.speaker_side_gate_logits: nn.ParameterDict | None = None
        self.speaker_side_kv_dims: dict[str, tuple[int, int]] = {}
        self.speaker_cross_attn_layer_indices: list[int] = []
        self.speaker_cross_attn_tokens: SpeakerTokenProjector | None = None
        self.speaker_cross_attn_seq_projector: SpeakerSequenceProjector | None = None
        self.speaker_cross_attn_layers: nn.ModuleDict | None = None
        self.null_speaker_embedding: nn.Parameter | None = None
        speaker_condition_pathway_enabled = bool(config.speaker_side_pathway_enabled) or bool(
            config.speaker_cross_attn_enabled
        )
        if speaker_condition_pathway_enabled:
            if int(config.speaker_embedding_dim) <= 0:
                raise ValueError("speaker side conditioning requires speaker_embedding_dim > 0")
            self.null_speaker_embedding = nn.Parameter(torch.zeros(int(config.speaker_embedding_dim)))

            def make_side_mlp(output_dim: int) -> nn.Sequential:
                mlp = nn.Sequential(
                    nn.LayerNorm(int(config.speaker_embedding_dim)),
                    nn.Linear(int(config.speaker_embedding_dim), int(config.adapter_dim)),
                    nn.SiLU(),
                    nn.Linear(int(config.adapter_dim), int(output_dim)),
                )
                nn.init.normal_(mlp[-1].weight, mean=0.0, std=1.0e-3)
                nn.init.zeros_(mlp[-1].bias)
                return mlp

            speaker_side_num_layers = speaker_side_decoder_layer_count(layers)
        if bool(config.speaker_side_pathway_enabled):
            self.speaker_side_layer_indices = parse_adapter_layers(
                config.speaker_side_pathway_layers,
                speaker_side_num_layers,
            )
            self.speaker_side_adaln = nn.ModuleDict(
                {str(idx): make_side_mlp(hidden_size * 2) for idx in self.speaker_side_layer_indices}
            )
            self.speaker_side_gate_logits = nn.ParameterDict(
                {
                    str(idx): nn.Parameter(torch.tensor(float(config.speaker_side_pathway_gate_init)))
                    for idx in self.speaker_side_layer_indices
                }
            )
            if bool(config.speaker_side_pathway_kv_bias):
                kv_modules: dict[str, nn.Module] = {}
                for idx in self.speaker_side_layer_indices:
                    layer = layers[idx]
                    attn = getattr(layer, "self_attn", getattr(layer, "attention", None))
                    k_proj = getattr(attn, "k_proj", None) if attn is not None else None
                    v_proj = getattr(attn, "v_proj", None) if attn is not None else None
                    k_dim = self._linear_out_features(k_proj)
                    v_dim = self._linear_out_features(v_proj)
                    if k_dim is None or v_dim is None:
                        continue
                    self.speaker_side_kv_dims[str(idx)] = (int(k_dim), int(v_dim))
                    kv_modules[str(idx)] = make_side_mlp(int(k_dim) + int(v_dim))
                self.speaker_side_kv_bias = nn.ModuleDict(kv_modules)
        if bool(config.speaker_cross_attn_enabled):
            cross_source = str(config.speaker_cross_attn_source or "vector").strip().lower()
            if cross_source not in {"vector", "sequence"}:
                raise ValueError("speaker_cross_attn_source must be 'vector' or 'sequence'")
            speaker_side_num_layers = speaker_side_decoder_layer_count(layers)
            self.speaker_cross_attn_layer_indices = parse_adapter_layers(
                config.speaker_cross_attn_layers,
                speaker_side_num_layers,
            )
            if cross_source == "vector":
                cross_tokens = int(config.speaker_cross_attn_tokens)
                if cross_tokens <= 0:
                    raise ValueError("speaker_cross_attn_source=vector requires speaker_cross_attn_tokens > 0")
                self.speaker_cross_attn_tokens = SpeakerTokenProjector(
                    int(config.speaker_embedding_dim),
                    hidden_size,
                    num_tokens=cross_tokens,
                    adapter_dim=int(config.adapter_dim),
                    dropout=float(config.speaker_cross_attn_dropout),
                    output_init_std=config.speaker_cross_attn_token_init_std,
                )
            else:
                if int(config.speaker_cross_attn_seq_dim) <= 0:
                    raise ValueError("speaker_cross_attn_source=sequence requires speaker_cross_attn_seq_dim > 0")
                self.speaker_cross_attn_seq_projector = SpeakerSequenceProjector(
                    int(config.speaker_cross_attn_seq_dim),
                    hidden_size,
                    dropout=float(config.speaker_cross_attn_dropout),
                )
            self.speaker_cross_attn_layers = nn.ModuleDict(
                {
                    str(idx): SpeakerCrossAttentionLayer(
                        hidden_size,
                        num_heads=int(config.num_heads),
                        adapter_dim=int(config.adapter_dim),
                        dropout=float(config.speaker_cross_attn_dropout),
                        gate_init=float(config.speaker_cross_attn_gate_init),
                        output_scale=float(config.speaker_cross_attn_output_scale),
                        normalize_tokens=config.speaker_cross_attn_token_init_std is None,
                    )
                    for idx in self.speaker_cross_attn_layer_indices
                }
            )
            self.set_speaker_cross_attn_runtime_scale_multiplier(
                float(config.speaker_cross_attn_runtime_scale_multiplier)
            )
        self.ref_speaker_prompt: nn.Module | None = None
        if (
            self._normalize_ref_speaker_prompt_token_source(config.ref_speaker_prompt_token_source) == "speaker_mlp"
            and config.speaker_embedding_dim > 0
            and int(config.ref_speaker_prompt_tokens) > 0
        ):
            self.ref_speaker_prompt = nn.Sequential(
                nn.LayerNorm(int(config.speaker_embedding_dim)),
                nn.Linear(int(config.speaker_embedding_dim), int(config.adapter_dim)),
                nn.SiLU(),
                nn.Linear(int(config.adapter_dim), int(config.ref_speaker_prompt_tokens) * hidden_size),
            )
        self.ref_speaker_adaln: nn.Module | None = None
        if config.speaker_embedding_dim > 0 and float(config.ref_speaker_adaln_weight) > 0.0:
            self.ref_speaker_adaln = nn.Sequential(
                nn.LayerNorm(int(config.speaker_embedding_dim)),
                nn.Linear(int(config.speaker_embedding_dim), int(config.adapter_dim)),
                nn.SiLU(),
                nn.Linear(int(config.adapter_dim), hidden_size * 2),
            )
        self.speaker_projection: nn.Module | None = None
        if (
            config.speaker_embedding_dim > 0
            and (
                config.target_speaker_similarity_weight > 0
                or config.source_speaker_suppression_weight > 0
                or config.speaker_infonce_weight > 0
            )
        ):
            self.speaker_projection = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, int(config.speaker_embedding_dim)),
            )
        negative_pool_dim = max(1, int(config.speaker_embedding_dim))
        self.register_buffer(
            "speaker_infonce_negative_pool",
            torch.empty(0, negative_pool_dim, dtype=torch.float32),
            persistent=True,
        )
        self.speaker_infonce_negative_pool_paths: list[str] = []
        self.prosody_head: ProsodyHead | None = None
        if config.prosody_loss_weight > 0:
            self.prosody_head = ProsodyHead(
                hidden_size=hidden_size,
                adapter_dim=config.adapter_dim,
                dropout=config.dropout,
            )
        self.content_head: ContentEmbeddingHead | None = None
        if (
            config.content_loss_weight > 0
            and config.content_embedding_dim > 0
            and config.content_embedding_weight > 0
        ):
            self.content_head = ContentEmbeddingHead(
                hidden_size=hidden_size,
                embedding_dim=config.content_embedding_dim,
                adapter_dim=config.adapter_dim,
                dropout=config.dropout,
            )
        self.content_ctc_head: ContentCTCHead | None = None
        if config.content_ctc_weight > 0 and config.content_ctc_vocab_size > 1:
            self.content_ctc_head = ContentCTCHead(
                hidden_size=hidden_size,
                vocab_size=config.content_ctc_vocab_size,
                adapter_dim=config.adapter_dim,
                dropout=config.dropout,
            )
        self.content_token_head: ContentTokenHead | None = None
        if (
            config.content_loss_weight > 0
            and config.content_token_vocab_size > 0
            and config.content_token_weight > 0
        ):
            self.content_token_head = ContentTokenHead(
                hidden_size=hidden_size,
                vocab_size=config.content_token_vocab_size,
                adapter_dim=config.adapter_dim,
                dropout=config.dropout,
            )
        self.semantic_token_head: ContentTokenHead | None = None
        self.semantic_feature_head: SemanticFeatureHead | None = None
        semantic_mode = str(config.semantic_mode or "discrete").strip().lower()
        if config.semantic_loss_weight > 0:
            if semantic_mode == "continuous":
                if config.semantic_feature_dim <= 0:
                    raise ValueError("semantic_mode=continuous requires semantic_feature_dim > 0")
                self.semantic_feature_head = SemanticFeatureHead(
                    hidden_size=hidden_size,
                    feature_dim=config.semantic_feature_dim,
                    adapter_dim=config.adapter_dim,
                    dropout=config.dropout,
                )
            else:
                if config.semantic_vocab_size <= 0:
                    raise ValueError("semantic_mode=discrete requires semantic_vocab_size > 0")
                self.semantic_token_head = ContentTokenHead(
                    hidden_size=hidden_size,
                    vocab_size=config.semantic_vocab_size,
                    adapter_dim=config.adapter_dim,
                    dropout=config.dropout,
                )
        self.progress_stop_head: ProgressStopHead | None = None
        if config.progress_loss_weight > 0 or config.stop_loss_weight > 0:
            self.progress_stop_head = ProgressStopHead(
                hidden_size=hidden_size,
                num_bins=config.progress_num_bins,
                adapter_dim=config.adapter_dim,
                dropout=config.dropout,
            )
        n_vq = int(base_model.config.n_vq)
        self.source_content_memory_type = normalize_source_content_memory_type(config.source_content_memory_type)
        self.source_content_codec_codebooks = parse_codebook_indices(config.source_content_codec_codebooks, n_vq)
        self.source_semantic_memory_encoder: nn.Module | None = None
        self.source_semantic_codec_residual_encoder: nn.Module | None = None
        self.source_semantic_layer_adapters: nn.ModuleDict | None = None
        self.source_semantic_layer_indices: list[int] = []
        if config.source_semantic_memory_enabled and self.source_content_memory_type != "none":
            if self.source_content_memory_type in SOURCE_CONTINUOUS_MEMORY_TYPES:
                self.source_semantic_memory_encoder = SourceSemanticMemoryEncoder(
                    input_dim=int(config.source_semantic_feature_dim),
                    hidden_size=hidden_size,
                    dropout=float(config.source_semantic_dropout),
                    position_scale=float(config.source_semantic_position_scale),
                )
            elif self.source_content_memory_type in {"text_tokens", "semantic_units"}:
                self.source_semantic_memory_encoder = SourceTokenMemoryEncoder(
                    vocab_size=int(config.source_content_vocab_size),
                    hidden_size=hidden_size,
                    padding_id=int(config.source_content_padding_id),
                    dropout=float(config.source_semantic_dropout),
                    position_scale=float(config.source_semantic_position_scale),
                    dedup_units=bool(config.source_content_dedup_units)
                    and self.source_content_memory_type == "semantic_units",
                )
            elif self.source_content_memory_type == "codec_bottleneck":
                if not self.source_content_codec_codebooks:
                    raise ValueError("codec_bottleneck source content memory requires non-empty source_content_codec_codebooks")
                self.source_semantic_memory_encoder = SourceCodecBottleneckMemoryEncoder(
                    hidden_size=hidden_size,
                    bottleneck_dim=int(config.source_content_codec_bottleneck_dim),
                    dropout=float(config.source_semantic_dropout),
                    position_scale=float(config.source_semantic_position_scale),
                )
            else:
                raise ValueError(
                    "Unsupported source_content_memory_type="
                    f"{config.source_content_memory_type!r}; expected none, hubert_continuous, "
                    "wavlm_bnf_continuous, asr_bnf_continuous, wavlm_continuous, "
                    "text_tokens, semantic_units, or codec_bottleneck"
                )
            if (
                self.source_content_memory_type in SOURCE_CONTINUOUS_MEMORY_TYPES
                and float(config.source_codec_residual_memory_weight) > 0.0
            ):
                if not self.source_content_codec_codebooks:
                    raise ValueError(
                        "source_codec_residual_memory_weight > 0 requires non-empty source_content_codec_codebooks"
                    )
                self.source_semantic_codec_residual_encoder = SourceCodecBottleneckMemoryEncoder(
                    hidden_size=hidden_size,
                    bottleneck_dim=int(config.source_content_codec_bottleneck_dim),
                    dropout=float(config.source_semantic_dropout),
                    position_scale=float(config.source_semantic_position_scale),
                )
            self.source_semantic_layer_indices = parse_adapter_layers(
                config.source_semantic_adapter_layers,
                len(layers),
            )
            self.source_semantic_layer_adapters = nn.ModuleDict(
                {
                    str(idx): SourceSemanticAdapter(
                        hidden_size=hidden_size,
                        num_heads=config.num_heads,
                        adapter_dim=config.adapter_dim,
                        dropout=config.dropout,
                        init_gate=config.source_semantic_init_gate,
                        no_text_gate=config.source_semantic_no_text_gate,
                        text_gate=config.source_semantic_text_gate,
                        allow_learned_text_gate=config.source_semantic_allow_learned_text_gate,
                        monotonic_bias_strength=config.source_semantic_monotonic_bias_strength,
                        monotonic_bias_width=config.source_semantic_monotonic_bias_width,
                    )
                    for idx in self.source_semantic_layer_indices
                }
            )
        self.content_cross_attn_encoder: ContentConformerEncoder | None = None
        self.content_cross_attn_layers: nn.ModuleDict | None = None
        self.content_cross_attn_layer_indices: list[int] = []
        self.content_phoneme_classifier: ContentPhonemeClassifierHead | None = None
        if bool(config.content_cross_attn_enabled):
            self.content_cross_attn_encoder = ContentConformerEncoder(
                input_dim=int(config.content_cross_attn_feature_dim),
                hidden_size=hidden_size,
                num_layers=int(config.content_encoder_layers),
                num_heads=int(config.num_heads),
                dropout=float(config.content_cross_attn_dropout),
                conv_kernel_size=int(config.content_encoder_conv_kernel_size),
            )
            self.content_cross_attn_layer_indices = parse_adapter_layers(
                config.content_cross_attn_layers,
                len(layers),
            )
            self.content_cross_attn_layers = nn.ModuleDict(
                {
                    str(idx): ContentCrossAttentionLayer(
                        hidden_size=hidden_size,
                        num_heads=int(config.num_heads),
                        adapter_dim=int(config.adapter_dim),
                        dropout=float(config.content_cross_attn_dropout),
                        gate_init=float(config.content_cross_attn_gate_init),
                        output_scale=float(config.content_cross_attn_output_scale),
                    )
                    for idx in self.content_cross_attn_layer_indices
                }
            )
            if float(config.phoneme_classifier_loss_weight) > 0.0:
                if int(config.content_token_vocab_size) <= 1:
                    raise ValueError(
                        "phoneme_classifier_loss_weight > 0 requires content_token_vocab_size > 1"
                    )
                self.content_phoneme_classifier = ContentPhonemeClassifierHead(
                    hidden_size=hidden_size,
                    vocab_size=int(config.content_token_vocab_size),
                    adapter_dim=int(config.adapter_dim),
                    dropout=float(config.content_cross_attn_dropout),
                )
        self.content_source_codec_codebooks = parse_codebook_indices(config.content_source_codec_codebooks, n_vq)
        self.content_codec_head: SourceCodecContentHead | None = None
        if config.content_loss_weight > 0 and config.content_source_codec_weight > 0:
            if not self.content_source_codec_codebooks:
                raise ValueError("content_source_codec_weight > 0 requires non-empty content_source_codec_codebooks")
            audio_vocab_size = int(getattr(base_model.config, "audio_vocab_size", 1024))
            self.content_codec_head = SourceCodecContentHead(
                hidden_size=hidden_size,
                codebooks=self.content_source_codec_codebooks,
                audio_vocab_size=audio_vocab_size,
                adapter_dim=config.adapter_dim,
                dropout=config.dropout,
            )
        self.role_router: RoleCodecRouter | None = None
        self.source_prosody_encoder: SourceProsodyEncoder | None = None
        self.target_head_router: PerCodebookTargetHeadRouter | None = None
        if config.use_role_routing:
            self.role_router = RoleCodecRouter(n_vq=n_vq)
            self.source_prosody_encoder = SourceProsodyEncoder(
                hidden_size=hidden_size,
                num_memory_tokens=config.prosody_memory_tokens,
                num_heads=config.num_heads,
                adapter_dim=config.adapter_dim,
                dropout=config.dropout,
                encoder_type=config.source_prosody_encoder_type,
                encoder_layers=config.source_prosody_encoder_layers,
                conv_kernel_size=config.source_prosody_conv_kernel_size,
            )
            if config.target_head_routing:
                self.target_head_router = PerCodebookTargetHeadRouter(
                    hidden_size=hidden_size,
                    n_vq=n_vq,
                    num_heads=config.num_heads,
                    adapter_dim=config.adapter_dim,
                    dropout=config.dropout,
                )
        self._active_timbre_tokens: torch.Tensor | None = None
        self._active_ref_speaker_embedding: torch.Tensor | None = None
        self._active_ref_speaker_mask: torch.Tensor | None = None
        self._active_ref_speaker_adaln: tuple[torch.Tensor, torch.Tensor] | None = None
        self._active_speaker_side_embedding: torch.Tensor | None = None
        self._active_speaker_side_mask: torch.Tensor | None = None
        self._active_speaker_side_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._active_speaker_cross_attn_tokens: torch.Tensor | None = None
        self._active_speaker_cross_attn_mask: torch.Tensor | None = None
        self._active_speaker_cross_attn_stats: list[dict[str, float]] = []
        self._active_target_position_mask: torch.Tensor | None = None
        self._active_target_position_progress: torch.Tensor | None = None
        self._active_source_semantic_memory: torch.Tensor | None = None
        self._active_source_semantic_mask: torch.Tensor | None = None
        self._active_content_cross_attn_memory: torch.Tensor | None = None
        self._active_content_cross_attn_mask: torch.Tensor | None = None
        self._active_vc_mode_id: torch.Tensor | None = None
        self._active_source_semantic_attentions: list[torch.Tensor] = []
        self._active_source_semantic_adapter_stats: list[dict[str, float]] = []
        self._active_content_cross_attn_attentions: list[torch.Tensor] = []
        self._active_content_cross_attn_stats: list[dict[str, float]] = []
        self._content_guided_attn_runtime_step: int = 0
        self.capture_source_semantic_attention: bool = False
        self.source_semantic_attention_capture_max_tokens: int = 2048
        self.last_source_semantic_attention_maps: list[dict[str, Any]] = []
        self._source_semantic_attention_captured_tokens: int = 0
        self.last_timbre_memory_shape: tuple[int, ...] | None = None
        self.last_prosody_memory_shape: tuple[int, ...] | None = None
        self.last_source_semantic_memory_shape: tuple[int, ...] | None = None
        self.last_content_cross_attn_memory_shape: tuple[int, ...] | None = None
        self.last_content_cross_attn_aux_loss: float | None = None
        self.last_content_cross_attn_aux_stats: dict[str, float] = {}
        self.last_source_semantic_aux_loss: float | None = None
        self.last_source_semantic_aux_stats: dict[str, float] = {}
        self.last_source_prosody_gate_stats: dict[str, float] = {}
        self.last_route_loss: float | None = None
        self.last_route_stats: dict[str, float] = {}
        self.last_speaker_aux_loss: float | None = None
        self.last_speaker_aux_stats: dict[str, float] = {}
        self.last_ref_speaker_prompt_slot_stats: dict[str, float] = {}
        self.last_ref_speaker_prompt_attention_stats: dict[str, Any] = {}
        self.last_target_front_ce_stats: dict[str, float] = {}
        self.last_prosody_aux_loss: float | None = None
        self.last_prosody_aux_stats: dict[str, float] = {}
        self.last_content_aux_loss: float | None = None
        self.last_content_aux_stats: dict[str, float] = {}
        self.last_content_ctc_aux_loss: float | None = None
        self.last_content_ctc_aux_stats: dict[str, float] = {}
        self.last_semantic_aux_loss: float | None = None
        self.last_semantic_aux_stats: dict[str, float] = {}
        self.last_ref_content_suppression_loss: float | None = None
        self.last_ref_content_suppression_stats: dict[str, float] = {}
        self.last_progress_stop_aux_loss: float | None = None
        self.last_progress_stop_aux_stats: dict[str, float] = {}
        self.last_progress_stop_infer_stats: dict[str, float] = {}
        self.last_speaker_side_stats: dict[str, float] = {}
        self.last_forward_debug: dict[str, Any] = {}
        self._hook_handles = []
        self._kv_hook_handles = []
        self._install_hooks()

    def build_speaker_infonce_negative_pool(
        self,
        paths: list[str | Path | None] | tuple[str | Path | None, ...],
        *,
        pool_size: int | None = None,
        seed: int | None = None,
    ) -> dict[str, float]:
        requested = int(
            self.timbre_memory_config.speaker_infonce_negative_pool_size
            if pool_size is None
            else pool_size
        )
        if requested <= 0:
            self.speaker_infonce_negative_pool = torch.empty(
                0,
                max(1, int(self.timbre_memory_config.speaker_embedding_dim)),
                dtype=torch.float32,
                device=self.speaker_infonce_negative_pool.device,
            )
            self.speaker_infonce_negative_pool_paths = []
            return {"speaker_infonce_negative_pool_size": 0.0}
        unique_paths: list[str] = []
        seen: set[str] = set()
        for path in paths:
            if path in (None, ""):
                continue
            text = str(path)
            if text in seen:
                continue
            seen.add(text)
            unique_paths.append(text)
        if len(unique_paths) < requested:
            raise ValueError(
                f"speaker InfoNCE negative pool requested {requested} unique paths, got {len(unique_paths)}"
            )
        rng = random.Random(
            int(self.timbre_memory_config.speaker_infonce_negative_pool_seed if seed is None else seed)
        )
        selected = rng.sample(unique_paths, requested) if len(unique_paths) > requested else unique_paths
        emb, mask = self.speaker_encoder(selected, device=torch.device("cpu"), dtype=torch.float32)
        if emb is None or mask is None:
            raise ValueError("speaker InfoNCE negative pool could not load speaker embeddings")
        valid = mask.detach().cpu().bool()
        if int(valid.sum().item()) < requested:
            raise ValueError(
                f"speaker InfoNCE negative pool requested {requested} valid embeddings, "
                f"got {int(valid.sum().item())}"
            )
        pool = F.normalize(emb.detach().float().cpu()[valid][:requested], dim=-1)
        device = self.speaker_infonce_negative_pool.device
        self.speaker_infonce_negative_pool = pool.to(device=device)
        self.speaker_infonce_negative_pool_paths = selected[:requested]
        return {
            "speaker_infonce_negative_pool_size": float(pool.shape[0]),
            "speaker_infonce_negative_pool_dim": float(pool.shape[1]),
        }

    def _speaker_infonce_loss_from_embeddings(
        self,
        pred_valid: torch.Tensor,
        ref_valid_emb: torch.Tensor,
        *,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        requested = int(self.timbre_memory_config.speaker_infonce_negative_pool_size)
        if requested > 0:
            pool = self.speaker_infonce_negative_pool
            if pool is None or int(pool.shape[0]) < requested:
                raise RuntimeError(
                    "speaker InfoNCE negative pool is enabled but not populated: "
                    f"requested={requested} available={0 if pool is None else int(pool.shape[0])}"
                )
            negatives = F.normalize(pool[:requested].to(device=pred_valid.device, dtype=torch.float32), dim=-1)
            positive_logits = (pred_valid.float() * ref_valid_emb.float()).sum(dim=-1, keepdim=True)
            negative_logits = pred_valid.float() @ negatives.t()
            logits = torch.cat([positive_logits, negative_logits], dim=-1) / float(temperature)
            expected_denominator = requested + 1
            if int(logits.shape[1]) != expected_denominator:
                raise RuntimeError(
                    "speaker InfoNCE denominator mismatch: "
                    f"got={int(logits.shape[1])} expected={expected_denominator}"
                )
            labels = torch.zeros(int(logits.shape[0]), device=logits.device, dtype=torch.long)
            return F.cross_entropy(logits, labels), logits, labels
        logits = pred_valid.float() @ ref_valid_emb.float().t() / float(temperature)
        if int(logits.shape[1]) < 2:
            raise RuntimeError(
                "speaker InfoNCE has no negatives: effective batch produced denominator "
                f"{int(logits.shape[1])}. Set speaker_infonce_negative_pool_size > 0."
            )
        labels = torch.arange(int(logits.shape[0]), device=logits.device)
        return F.cross_entropy(logits, labels), logits, labels

    def clear_source_semantic_attention_capture(self) -> None:
        self.last_source_semantic_attention_maps = []
        self._source_semantic_attention_captured_tokens = 0

    def _capture_source_semantic_attention(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
        target_mask: torch.Tensor,
        source_semantic_mask: torch.Tensor | None,
    ) -> None:
        if not self.capture_source_semantic_attention:
            return
        max_tokens = int(self.source_semantic_attention_capture_max_tokens)
        if max_tokens <= 0 or self._source_semantic_attention_captured_tokens >= max_tokens:
            return
        if attention_weights.dim() != 4:
            return
        with torch.no_grad():
            # attention_weights: [B, heads, T, S]. Store only target positions
            # and average heads so visualization stays compact.
            probs = attention_weights.detach().float()
            token_mask = target_mask.to(device=probs.device).bool()
            if token_mask.shape != probs.shape[:1] + probs.shape[2:3]:
                token_mask = token_mask[:, -probs.shape[2] :]
            if source_semantic_mask is not None:
                source_mask = source_semantic_mask.to(device=probs.device).bool()
                probs = probs.masked_fill(~source_mask[:, None, None, :], 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
            else:
                source_mask = torch.ones(probs.shape[0], probs.shape[-1], dtype=torch.bool, device=probs.device)
            head_mean = probs.mean(dim=1)
            remaining = max_tokens - self._source_semantic_attention_captured_tokens
            for batch_idx in range(head_mean.shape[0]):
                selected = head_mean[batch_idx].masked_select(token_mask[batch_idx].unsqueeze(-1)).reshape(
                    -1,
                    head_mean.shape[-1],
                )
                if selected.numel() == 0:
                    continue
                if selected.shape[0] > remaining:
                    selected = selected[:remaining]
                valid_len = int(source_mask[batch_idx].long().sum().item())
                self.last_source_semantic_attention_maps.append(
                    {
                        "layer_idx": int(layer_idx),
                        "batch_idx": int(batch_idx),
                        "attention": selected.cpu().to(dtype=torch.float16),
                        "target_tokens": int(selected.shape[0]),
                        "source_tokens": int(selected.shape[1]),
                        "source_valid_tokens": valid_len,
                    }
                )
                self._source_semantic_attention_captured_tokens += int(selected.shape[0])
                remaining = max_tokens - self._source_semantic_attention_captured_tokens
                if remaining <= 0:
                    break

    def get_fsdp_ignored_modules(self) -> list[nn.Module]:
        """Return lightweight modules that should stay outside FSDP sharding."""
        modules: list[nn.Module] = []
        for module in (
            self.timbre_memory,
            self.layer_adapters,
            self.speaker_projection,
            self.prosody_head,
            self.content_head,
            self.content_ctc_head,
            self.content_token_head,
            self.content_codec_head,
            self.semantic_token_head,
            self.semantic_feature_head,
            self.progress_stop_head,
            self.source_semantic_memory_encoder,
            self.source_semantic_codec_residual_encoder,
            self.source_semantic_layer_adapters,
            self.content_cross_attn_encoder,
            self.content_cross_attn_layers,
            self.content_phoneme_classifier,
            self.role_router,
            self.source_prosody_encoder,
            self.target_head_router,
            self.ref_speaker_prompt,
            self.ref_speaker_adaln,
            self.speaker_side_adaln,
            self.speaker_side_kv_bias,
            self.speaker_side_gate_logits,
            self.speaker_cross_attn_tokens,
            self.speaker_cross_attn_seq_projector,
            self.speaker_cross_attn_layers,
            self.speaker_encoder,
        ):
            if module is not None:
                modules.append(module)
        return modules

    def _install_hooks(self) -> None:
        base_model = self.get_base_model()
        layers = base_model.language_model.layers
        hook_layer_indices = sorted(
            set(self.adapter_layer_indices)
            | set(self.source_semantic_layer_indices)
            | set(self.content_cross_attn_layer_indices)
            | set(self.speaker_side_layer_indices)
            | set(self.speaker_cross_attn_layer_indices)
        )
        for layer_idx in hook_layer_indices:
            handle = layers[layer_idx].register_forward_hook(self._make_layer_hook(layer_idx))
            self._hook_handles.append(handle)
        if self.speaker_side_kv_bias is not None:
            for layer_idx in self.speaker_side_layer_indices:
                layer = layers[layer_idx]
                attn = getattr(layer, "self_attn", getattr(layer, "attention", None))
                if attn is None or str(layer_idx) not in self.speaker_side_kv_bias:
                    continue
                for which in ("k", "v"):
                    proj = getattr(attn, f"{which}_proj", None)
                    if proj is not None:
                        handle = proj.register_forward_hook(self._make_speaker_side_kv_hook(layer_idx, which))
                        self._kv_hook_handles.append(handle)

    def _make_layer_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            timbre_tokens = self._active_timbre_tokens
            semantic_memory = self._active_source_semantic_memory
            semantic_mask = self._active_source_semantic_mask
            content_memory = self._active_content_cross_attn_memory
            content_mask = self._active_content_cross_attn_mask
            vc_mode_id = self._active_vc_mode_id
            target_mask = self._active_target_position_mask
            target_progress = self._active_target_position_progress
            has_timbre = timbre_tokens is not None and str(layer_idx) in self.layer_adapters
            has_semantic = (
                semantic_memory is not None
                and self.source_semantic_layer_adapters is not None
                and str(layer_idx) in self.source_semantic_layer_adapters
            )
            has_speaker_side = (
                self._active_speaker_side_embedding is not None
                and self.speaker_side_adaln is not None
                and self.speaker_side_gate_logits is not None
                and str(layer_idx) in self.speaker_side_adaln
            )
            has_speaker_cross_attn = (
                self._active_speaker_cross_attn_tokens is not None
                and self._active_speaker_cross_attn_mask is not None
                and self.speaker_cross_attn_layers is not None
                and str(layer_idx) in self.speaker_cross_attn_layers
            )
            has_content_cross_attn = (
                content_memory is not None
                and self.content_cross_attn_layers is not None
                and str(layer_idx) in self.content_cross_attn_layers
            )
            if (
                not has_timbre
                and not has_semantic
                and not has_speaker_side
                and not has_speaker_cross_attn
                and not has_content_cross_attn
            ) or target_mask is None:
                return output
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states.dim() != 3:
                return output
            if target_mask.shape[1] != hidden_states.shape[1]:
                mask = target_mask[:, -hidden_states.shape[1] :]
            else:
                mask = target_mask
            if target_progress is not None:
                if target_progress.shape[1] != hidden_states.shape[1]:
                    progress = target_progress[:, -hidden_states.shape[1] :]
                else:
                    progress = target_progress
            else:
                progress = None
            updated = hidden_states
            if has_speaker_side:
                side_delta = self._speaker_side_adaln_delta(layer_idx, updated)
                if side_delta is not None:
                    updated = updated + side_delta
            if has_speaker_cross_attn:
                cross_delta = self._speaker_cross_attn_delta(layer_idx, updated, mask)
                if cross_delta is not None:
                    updated = updated + cross_delta
            if has_content_cross_attn:
                content_delta = self._content_cross_attn_delta(layer_idx, updated, mask)
                if content_delta is not None:
                    updated = updated + content_delta
            if self._active_ref_speaker_adaln is not None:
                scale, shift = self._active_ref_speaker_adaln
                scale = scale.to(device=updated.device, dtype=updated.dtype)
                shift = shift.to(device=updated.device, dtype=updated.dtype)
                updated = updated + mask.to(device=updated.device, dtype=updated.dtype).unsqueeze(-1) * (
                    updated * scale + shift
                )
            if has_semantic:
                semantic_adapter = self.source_semantic_layer_adapters[str(layer_idx)]
                semantic_out = semantic_adapter(
                    updated,
                    semantic_memory.to(updated.device),
                    mask.to(updated.device),
                    source_semantic_mask=None if semantic_mask is None else semantic_mask.to(updated.device),
                    vc_mode_id=None if vc_mode_id is None else vc_mode_id.to(updated.device),
                    target_progress=None if progress is None else progress.to(updated.device),
                )
                updated = semantic_out.hidden_states
                if semantic_out.attention_weights is not None:
                    self._active_source_semantic_attentions.append(semantic_out.attention_weights)
                    self._capture_source_semantic_attention(
                        layer_idx,
                        semantic_out.attention_weights,
                        mask.to(updated.device),
                        None if semantic_mask is None else semantic_mask.to(updated.device),
                    )
                if semantic_out.stats:
                    semantic_stats = dict(semantic_out.stats)
                    semantic_stats["source_semantic_layer_counted"] = 1.0
                    for stat_key in (
                        "source_semantic_delta_ratio",
                        "source_semantic_delta_norm",
                        "source_semantic_hidden_norm",
                        "source_semantic_raw_delta_norm",
                        "source_semantic_prompt_delta_norm",
                        "source_semantic_attn_entropy",
                        "source_semantic_attn_peak_mean",
                        "source_semantic_attn_coverage",
                        "source_semantic_attn_expected_pos_mean",
                        "source_semantic_attn_expected_pos_begin",
                        "source_semantic_attn_expected_pos_mid",
                        "source_semantic_attn_expected_pos_end",
                        "source_semantic_attn_expected_pos_slope",
                    ):
                        if stat_key in semantic_out.stats:
                            short_key = stat_key.replace("source_semantic_", "")
                            semantic_stats[f"source_semantic_layer_{layer_idx}_{short_key}"] = float(semantic_out.stats[stat_key])
                    self._active_source_semantic_adapter_stats.append(semantic_stats)
            if has_timbre:
                adapter = self.layer_adapters[str(layer_idx)]
                updated = adapter(updated, timbre_tokens.to(updated.device), mask.to(updated.device))
            if isinstance(output, tuple):
                return (updated,) + output[1:]
            return updated

        return hook

    def get_base_model(self):
        if hasattr(self.model, "get_base_model"):
            return self.model.get_base_model()
        return self.model

    def _forward_model(self):
        if hasattr(self.model, "peft_config"):
            return self.get_base_model()
        return self.model

    def _embed_timbre_ref_codes(self, timbre_ref_codes: torch.Tensor) -> torch.Tensor:
        base_model = self.get_base_model()
        if timbre_ref_codes.dim() != 3 or timbre_ref_codes.shape[-1] != int(base_model.config.n_vq):
            raise ValueError(
                f"timbre_ref_codes must be [B, T, n_vq={base_model.config.n_vq}], got {tuple(timbre_ref_codes.shape)}"
            )
        timbre_ref_codes = timbre_ref_codes.to(next(self.parameters()).device)
        embeds = None
        for idx, embed_layer in enumerate(base_model.emb_ext):
            cur = embed_layer(timbre_ref_codes[..., idx])
            embeds = cur if embeds is None else embeds + cur
        if embeds is None:
            raise RuntimeError("No audio embedding layers found on base model.")
        return embeds

    def _apply_speaker_condition_dropout(
        self,
        speaker_embedding: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, float]:
        if speaker_embedding is None or speaker_mask is None:
            return speaker_embedding, speaker_mask, 0.0
        drop_prob = float(self.timbre_memory_config.speaker_condition_dropout)
        if not self.training or drop_prob <= 0.0:
            return speaker_embedding, speaker_mask, 0.0
        keep = torch.rand(speaker_mask.shape, device=speaker_mask.device) >= drop_prob
        keep = keep & speaker_mask.bool()
        dropped_rate = float((speaker_mask.bool() & ~keep).float().mean().detach().item())
        speaker_embedding = speaker_embedding.clone()
        speaker_embedding[~keep] = 0.0
        return speaker_embedding, keep, dropped_rate

    @staticmethod
    def _normalize_ref_speaker_prompt_token_source(source: str | None) -> str:
        normalized = str(source or "speaker_mlp").strip().lower().replace("-", "_")
        if normalized in {"speaker", "speaker_embedding", "speaker_mlp", "eref", "e_ref"}:
            return "speaker_mlp"
        if normalized in {"tref", "t_ref", "timbre_memory", "timbre_tokens", "tte"}:
            return "timbre_memory"
        return "speaker_mlp"

    def _postprocess_ref_prompt_tokens(
        self,
        tokens: torch.Tensor,
        *,
        dtype: torch.dtype,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(tokens.shape[0])
        hidden_size = int(self.config.language_config.hidden_size)
        if bool(self.timbre_memory_config.ref_speaker_prompt_output_norm):
            scale = float(self.timbre_memory_config.ref_speaker_prompt_output_scale)
            tokens = F.layer_norm(tokens.float(), (hidden_size,)).to(dtype=tokens.dtype) * scale
        if mask is not None:
            tokens = tokens * mask.to(device=tokens.device, dtype=tokens.dtype).view(batch_size, 1, 1)
        drop_prob = float(self.timbre_memory_config.ref_speaker_prompt_dropout)
        if self.training and drop_prob > 0.0:
            keep = (torch.rand((batch_size, 1, 1), device=tokens.device) >= drop_prob).to(dtype=tokens.dtype)
            tokens = tokens * keep
        return tokens.to(dtype=dtype)

    def _ref_speaker_prompt_tokens(
        self,
        dtype: torch.dtype,
        *,
        source_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        token_count = int(self.timbre_memory_config.ref_speaker_prompt_tokens)
        if token_count <= 0:
            return None
        hidden_size = int(self.config.language_config.hidden_size)
        source = self._normalize_ref_speaker_prompt_token_source(self.timbre_memory_config.ref_speaker_prompt_token_source)
        if source == "timbre_memory":
            if source_tokens is None:
                return None
            tokens = source_tokens[:, : min(token_count, int(source_tokens.shape[1]))]
            if int(tokens.shape[1]) < token_count:
                pad = tokens.new_zeros((int(tokens.shape[0]), token_count - int(tokens.shape[1]), hidden_size))
                tokens = torch.cat([tokens, pad], dim=1)
            return self._postprocess_ref_prompt_tokens(tokens, dtype=dtype)
        if self.ref_speaker_prompt is None or self._active_ref_speaker_embedding is None:
            return None
        embedding = self._active_ref_speaker_embedding
        tokens = self.ref_speaker_prompt(embedding.to(dtype=self.ref_speaker_prompt[0].weight.dtype))
        tokens = tokens.view(int(tokens.shape[0]), token_count, hidden_size)
        return self._postprocess_ref_prompt_tokens(tokens, dtype=dtype, mask=self._active_ref_speaker_mask)

    def _ref_speaker_prompt_mode(self) -> str:
        mode = str(self.timbre_memory_config.ref_speaker_prompt_mode or "memory").strip().lower()
        if mode in {"append", "side", "side_memory", "memory"}:
            return "memory"
        if mode in {"slot", "prompt_slot", "prompt-slot"}:
            return "slot"
        if mode in {"both", "memory_and_slot", "slot_and_memory"}:
            return "both"
        return "memory"

    def _prepare_ref_speaker_adaln(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.ref_speaker_adaln is None or self._active_ref_speaker_embedding is None:
            return None
        embedding = self._active_ref_speaker_embedding
        params = self.ref_speaker_adaln(embedding.to(dtype=self.ref_speaker_adaln[0].weight.dtype))
        scale, shift = params.chunk(2, dim=-1)
        weight = float(self.timbre_memory_config.ref_speaker_adaln_weight)
        scale = torch.tanh(scale).unsqueeze(1) * weight
        shift = shift.unsqueeze(1) * weight
        mask = self._active_ref_speaker_mask
        if mask is not None:
            valid = mask.to(device=scale.device, dtype=scale.dtype).view(-1, 1, 1)
            scale = scale * valid
            shift = shift * valid
        return scale.to(dtype=dtype), shift.to(dtype=dtype)

    def _inject_ref_speaker_prompt_embeds(
        self,
        kwargs: dict[str, Any],
        input_ids: torch.Tensor | None,
        timbre_ref_prompt_positions: torch.Tensor | None,
        *,
        ref_speaker_prompt_slot_positions: torch.Tensor | None = None,
        role_ids: torch.Tensor | None,
        ref_speaker_prompt_source_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if input_ids is None:
            return role_ids
        mode = self._ref_speaker_prompt_mode()
        if mode not in {"slot", "both"} or not bool(self.timbre_memory_config.ref_speaker_prompt_slot):
            return role_ids
        speaker_tokens = self._ref_speaker_prompt_tokens(
            dtype=next(self.parameters()).dtype,
            source_tokens=ref_speaker_prompt_source_tokens,
        )
        if speaker_tokens is None:
            return role_ids
        positions = self._align_time_mask(ref_speaker_prompt_slot_positions, input_ids.shape[:2])
        if positions is None or not bool(positions.any().item()):
            self.last_ref_speaker_prompt_slot_stats = {
                "ref_speaker_prompt_slot_wrote": 0.0,
                "ref_speaker_prompt_slot_pre_norm": 0.0,
                "ref_speaker_prompt_slot_post_norm": 0.0,
                "ref_speaker_prompt_slot_post_pre_ratio": 0.0,
                "ref_speaker_prompt_slot_missing": 1.0,
                "ref_speaker_prompt_slot_token_source": 1.0
                if self._normalize_ref_speaker_prompt_token_source(
                    self.timbre_memory_config.ref_speaker_prompt_token_source
                )
                == "timbre_memory"
                else 0.0,
            }
            return role_ids
        base_model = self.get_base_model()
        if self.role_router is not None and role_ids is not None:
            inputs_embeds = self.role_router.compute_input_embeddings(base_model, input_ids, role_ids)
        else:
            text_embeds = base_model.get_input_embeddings()(input_ids[..., 0])
            audio_embeds = None
            for idx, embed_layer in enumerate(base_model.emb_ext):
                cur = embed_layer(input_ids[..., idx + 1])
                audio_embeds = cur if audio_embeds is None else audio_embeds + cur
            inputs_embeds = text_embeds + audio_embeds
        inputs_embeds = inputs_embeds.clone()
        token_count = int(speaker_tokens.shape[1])
        wrote = 0
        pre_norm_total = 0.0
        post_norm_total = 0.0
        for batch_idx in range(int(input_ids.shape[0])):
            idxs = torch.nonzero(positions[batch_idx].to(device=input_ids.device).bool(), as_tuple=False).flatten()
            if idxs.numel() == 0:
                continue
            take = min(token_count, int(idxs.numel()))
            before = inputs_embeds[batch_idx, idxs[:take]].detach().float()
            inputs_embeds[batch_idx, idxs[:take]] = speaker_tokens[batch_idx, :take].to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            after = inputs_embeds[batch_idx, idxs[:take]].detach().float()
            pre_norm_total += float(before.norm(dim=-1).sum().item())
            post_norm_total += float(after.norm(dim=-1).sum().item())
            wrote += take
        if wrote > 0:
            kwargs["inputs_embeds"] = inputs_embeds
        self.last_ref_speaker_prompt_slot_stats = {
            "ref_speaker_prompt_slot_wrote": float(wrote),
            "ref_speaker_prompt_slot_pre_norm": pre_norm_total / max(1, wrote),
            "ref_speaker_prompt_slot_post_norm": post_norm_total / max(1, wrote),
            "ref_speaker_prompt_slot_post_pre_ratio": (
                (post_norm_total / max(1, wrote)) / max(1e-6, pre_norm_total / max(1, wrote))
            ),
            "ref_speaker_prompt_slot_missing": 0.0,
            "ref_speaker_prompt_slot_token_source": 1.0
            if self._normalize_ref_speaker_prompt_token_source(self.timbre_memory_config.ref_speaker_prompt_token_source)
            == "timbre_memory"
            else 0.0,
        }
        return role_ids

    def _compute_timbre_tokens(
        self,
        timbre_ref_codes: torch.Tensor,
        timbre_ref_mask: torch.Tensor | None = None,
        timbre_ref_speaker_embedding_path=None,
        timbre_ref_speaker_audio_path=None,
        force_drop_speaker_condition: bool = False,
    ) -> torch.Tensor:
        if self.timbre_memory is None:
            raise RuntimeError("Legacy timbre memory is disabled for this wrapper.")
        ref_embeddings = self._embed_timbre_ref_codes(timbre_ref_codes)
        if timbre_ref_mask is not None:
            timbre_ref_mask = timbre_ref_mask.to(ref_embeddings.device).bool()
        speaker_embedding = None
        speaker_mask = None
        needs_ref_speaker = (
            self.timbre_memory_config.speaker_embedding_dim > 0
            and (
                self.timbre_memory_config.speaker_conditioning
                or int(self.timbre_memory_config.ref_speaker_prompt_tokens) > 0
                or float(self.timbre_memory_config.ref_speaker_adaln_weight) > 0.0
                or float(self.timbre_memory_config.speaker_infonce_weight) > 0.0
            )
        )
        if needs_ref_speaker:
            timbre_paths = self._resolve_speaker_paths(
                timbre_ref_speaker_embedding_path,
                timbre_ref_speaker_audio_path,
                ref_embeddings.shape[0],
            )
            speaker_embedding, speaker_mask = self.speaker_encoder(
                timbre_paths,
                device=ref_embeddings.device,
                dtype=self.timbre_memory.ref_norm.weight.dtype,
            )
            if speaker_mask is not None and not bool(speaker_mask.any().item()):
                speaker_embedding = None
                speaker_mask = None
            speaker_embedding, speaker_mask, dropped = self._apply_speaker_condition_dropout(
                speaker_embedding,
                speaker_mask,
            )
            if force_drop_speaker_condition and speaker_embedding is not None:
                speaker_embedding = torch.zeros_like(speaker_embedding)
                if speaker_mask is not None:
                    speaker_mask = torch.zeros_like(speaker_mask, dtype=torch.bool)
            if dropped > 0.0:
                self.last_speaker_aux_stats["speaker_condition_dropout_rate"] = dropped
        self._active_ref_speaker_embedding = speaker_embedding
        self._active_ref_speaker_mask = speaker_mask
        memory_state = self.timbre_memory(
            ref_embeddings,
            timbre_ref_mask,
            speaker_embedding=speaker_embedding
            if bool(self.timbre_memory_config.speaker_conditioning)
            else None,
            speaker_mask=speaker_mask if bool(self.timbre_memory_config.speaker_conditioning) else None,
        )
        timbre_tokens = memory_state.timbre_tokens
        prompt_tokens = self._ref_speaker_prompt_tokens(dtype=timbre_tokens.dtype)
        if prompt_tokens is not None and self._ref_speaker_prompt_mode() in {"memory", "both"}:
            timbre_tokens = torch.cat([prompt_tokens.to(device=timbre_tokens.device), timbre_tokens], dim=1)
        self.last_timbre_memory_shape = tuple(timbre_tokens.shape)
        return timbre_tokens

    def _embed_source_content_codes(self, source_ref_codes: torch.Tensor) -> torch.Tensor:
        base_model = self.get_base_model()
        if source_ref_codes.dim() != 3 or source_ref_codes.shape[-1] != int(base_model.config.n_vq):
            raise ValueError(
                f"source_ref_codes must be [B, T, n_vq={base_model.config.n_vq}], got {tuple(source_ref_codes.shape)}"
            )
        if not self.source_content_codec_codebooks:
            raise ValueError("source content codec codebook list is empty")
        source_ref_codes = source_ref_codes.to(next(self.parameters()).device)
        embeds = None
        for idx in self.source_content_codec_codebooks:
            cur = base_model.emb_ext[int(idx)](source_ref_codes[..., int(idx)])
            embeds = cur if embeds is None else embeds + cur
        if embeds is None:
            raise RuntimeError("No source codec embeddings were computed.")
        return embeds

    def _source_ref_codes_from_prompt(
        self,
        input_ids: torch.Tensor | None,
        source_prompt_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if input_ids is None or source_prompt_positions is None:
            return None, None
        source_prompt_positions = self._align_time_mask(source_prompt_positions, input_ids.shape[:2])
        if source_prompt_positions is None or not bool(source_prompt_positions.any().item()):
            return None, None
        base_model = self.get_base_model()
        audio_pad_code = int(base_model.config.audio_pad_code)
        audio_codes = input_ids[..., 1:].long()
        batch_size = int(audio_codes.shape[0])
        n_vq = int(base_model.config.n_vq)
        lengths = source_prompt_positions.to(device=audio_codes.device).bool().sum(dim=1)
        max_len = max(1, int(lengths.max().item()))
        codes = audio_codes.new_full((batch_size, max_len, n_vq), audio_pad_code)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=audio_codes.device)
        for batch_idx in range(batch_size):
            cur = audio_codes[batch_idx, source_prompt_positions[batch_idx].to(device=audio_codes.device)]
            if cur.numel() == 0:
                continue
            codes[batch_idx, : cur.shape[0]] = cur
            mask[batch_idx, : cur.shape[0]] = True
        return codes, mask

    def _compute_source_codec_residual_memory(
        self,
        *,
        source_ref_codes: torch.Tensor | None = None,
        source_ref_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        source_prompt_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
        encoder = self.source_semantic_codec_residual_encoder
        weight = float(self.timbre_memory_config.source_codec_residual_memory_weight)
        if encoder is None or weight <= 0.0:
            return None, None, {"source_codec_residual_memory_enabled": 0.0}
        if source_ref_codes is None:
            source_ref_codes, source_ref_mask = self._source_ref_codes_from_prompt(input_ids, source_prompt_positions)
        if source_ref_codes is None:
            return None, None, {
                "source_codec_residual_memory_enabled": 1.0,
                "source_codec_residual_memory_missing": 1.0,
            }
        source_embeddings = self._embed_source_content_codes(source_ref_codes)
        if bool(self.timbre_memory_config.source_codec_residual_memory_detach):
            source_embeddings = source_embeddings.detach()
        if source_ref_mask is not None:
            source_ref_mask = source_ref_mask.to(device=source_embeddings.device).bool()
        state = encoder(source_embeddings, attention_mask=source_ref_mask)
        memory = state.memory * weight
        stats = {
            "source_codec_residual_memory_enabled": 1.0,
            "source_codec_residual_memory_weight": float(weight),
            "source_codec_residual_memory_detach": 1.0
            if bool(self.timbre_memory_config.source_codec_residual_memory_detach)
            else 0.0,
            "source_codec_residual_memory_tokens": float(memory.shape[1]),
        }
        for key, value in state.stats.items():
            stats[f"source_codec_residual_{key}"] = float(value)
        return memory, state.mask, stats

    def _compute_source_semantic_memory(
        self,
        source_semantic_features: torch.Tensor | None,
        source_semantic_features_mask: torch.Tensor | None = None,
        *,
        content_token_ids: torch.Tensor | None = None,
        content_token_ids_mask: torch.Tensor | None = None,
        source_semantic_units: torch.Tensor | None = None,
        source_semantic_units_mask: torch.Tensor | None = None,
        source_ref_codes: torch.Tensor | None = None,
        source_ref_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        source_prompt_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        self.last_source_semantic_memory_shape = None
        self.last_source_semantic_aux_stats = {}
        if self.source_semantic_memory_encoder is None:
            return None, None
        memory_type = self.source_content_memory_type
        self.last_source_semantic_aux_stats["source_content_memory_type_id"] = float(
            SOURCE_MEMORY_TYPE_IDS.get(memory_type, 0.0)
        )
        device = next(self.source_semantic_memory_encoder.parameters()).device
        if memory_type in SOURCE_CONTINUOUS_MEMORY_TYPES:
            if source_semantic_features is None:
                self.last_source_semantic_aux_stats["source_content_memory_missing"] = 1.0
                return None, None
            source_semantic_features = source_semantic_features.to(device=device)
            if source_semantic_features_mask is not None:
                source_semantic_features_mask = source_semantic_features_mask.to(device=device).bool()
            state = self.source_semantic_memory_encoder(
                source_semantic_features,
                attention_mask=source_semantic_features_mask,
            )
        elif memory_type == "text_tokens":
            if content_token_ids is None:
                self.last_source_semantic_aux_stats["source_content_memory_missing"] = 1.0
                return None, None
            state = self.source_semantic_memory_encoder(
                content_token_ids.to(device=device),
                attention_mask=None if content_token_ids_mask is None else content_token_ids_mask.to(device=device).bool(),
            )
        elif memory_type == "semantic_units":
            if source_semantic_units is None:
                self.last_source_semantic_aux_stats["source_content_memory_missing"] = 1.0
                return None, None
            state = self.source_semantic_memory_encoder(
                source_semantic_units.to(device=device),
                attention_mask=None if source_semantic_units_mask is None else source_semantic_units_mask.to(device=device).bool(),
            )
        elif memory_type == "codec_bottleneck":
            if source_ref_codes is None:
                source_ref_codes, source_ref_mask = self._source_ref_codes_from_prompt(input_ids, source_prompt_positions)
            if source_ref_codes is None:
                self.last_source_semantic_aux_stats["source_content_memory_missing"] = 1.0
                return None, None
            source_embeddings = self._embed_source_content_codes(source_ref_codes)
            if source_ref_mask is not None:
                source_ref_mask = source_ref_mask.to(device=source_embeddings.device).bool()
            state = self.source_semantic_memory_encoder(
                source_embeddings,
                attention_mask=source_ref_mask,
            )
        else:
            self.last_source_semantic_aux_stats["source_content_memory_missing"] = 1.0
            return None, None
        memory = state.memory
        mask = state.mask
        stats = dict(state.stats)
        if memory_type in SOURCE_CONTINUOUS_MEMORY_TYPES:
            residual_memory, residual_mask, residual_stats = self._compute_source_codec_residual_memory(
                source_ref_codes=source_ref_codes,
                source_ref_mask=source_ref_mask,
                input_ids=input_ids,
                source_prompt_positions=source_prompt_positions,
            )
            stats.update(residual_stats)
            if residual_memory is not None:
                residual_memory = residual_memory.to(device=memory.device, dtype=memory.dtype)
                if mask is None and residual_mask is None:
                    combined_mask = None
                else:
                    if mask is None:
                        mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
                    else:
                        mask = mask.to(device=memory.device).bool()
                    if residual_mask is None:
                        residual_mask = torch.ones(
                            residual_memory.shape[:2],
                            dtype=torch.bool,
                            device=memory.device,
                        )
                    else:
                        residual_mask = residual_mask.to(device=memory.device).bool()
                    combined_mask = torch.cat([mask, residual_mask], dim=1)
                memory = torch.cat([memory, residual_memory], dim=1)
                mask = combined_mask
                stats["source_semantic_continuous_tokens"] = float(state.memory.shape[1])
                stats["source_semantic_codec_residual_tokens"] = float(residual_memory.shape[1])
                stats["source_semantic_total_memory_tokens"] = float(memory.shape[1])
        self.last_source_semantic_memory_shape = tuple(memory.shape)
        self.last_source_semantic_aux_stats.update(stats)
        return memory, mask

    def _source_semantic_progress_aux_loss(
        self,
        target_position_mask: torch.Tensor | None,
        source_semantic_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        self.last_source_semantic_aux_loss = None
        weight = float(self.timbre_memory_config.source_semantic_progress_weight)
        stats = dict(self.last_source_semantic_aux_stats)
        adapter_stats = self._active_source_semantic_adapter_stats
        if adapter_stats:
            keys = sorted({key for item in adapter_stats for key in item})
            for key in keys:
                values = [float(item[key]) for item in adapter_stats if key in item]
                if values:
                    stats[key] = sum(values) / len(values)
        stats["source_semantic_attention_layers"] = float(len(self._active_source_semantic_attentions))
        if target_position_mask is None:
            self.last_source_semantic_aux_stats = stats
            return None
        state = compute_semantic_attention_progress(
            self._active_source_semantic_attentions,
            target_position_mask,
            source_semantic_mask,
        )
        if state.stats:
            stats.update(state.stats)
        self.last_source_semantic_aux_stats = stats
        if state.loss is None or weight <= 0:
            return None
        weighted = state.loss * weight
        stats["source_semantic_progress_loss_weighted"] = float(weighted.detach().item())
        self.last_source_semantic_aux_loss = float(weighted.detach().item())
        return weighted

    def set_content_guided_attn_runtime_step(self, step: int) -> None:
        self._content_guided_attn_runtime_step = max(0, int(step))

    def _compute_content_cross_attn_memory(
        self,
        source_semantic_features: torch.Tensor | None,
        source_semantic_features_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        self.last_content_cross_attn_memory_shape = None
        self.last_content_cross_attn_aux_stats = {}
        if self.content_cross_attn_encoder is None:
            return None, None
        if source_semantic_features is None:
            self.last_content_cross_attn_aux_stats = {"content_cross_attn_memory_missing": 1.0}
            return None, None
        device = next(self.content_cross_attn_encoder.parameters()).device
        features = source_semantic_features.to(device=device)
        mask = None if source_semantic_features_mask is None else source_semantic_features_mask.to(device=device).bool()
        state = self.content_cross_attn_encoder(features, mask)
        self.last_content_cross_attn_memory_shape = tuple(state.memory.shape)
        self.last_content_cross_attn_aux_stats.update(state.stats)
        return state.memory, state.mask

    def _refresh_content_cross_attn_stats(self) -> None:
        stats_rows = self._active_content_cross_attn_stats
        if not stats_rows:
            return
        aggregate_keys = (
            "content_cross_attn_gate_mean",
            "content_cross_attn_delta_norm",
            "content_cross_attn_raw_delta_norm",
            "content_cross_attn_hidden_norm",
            "content_cross_attn_delta_ratio",
            "content_cross_attn_output_scale",
            "content_cross_attn_attn_entropy",
            "content_cross_attn_attn_peak_mean",
        )
        for key in aggregate_keys:
            values = [float(row[key]) for row in stats_rows if key in row]
            if values:
                self.last_content_cross_attn_aux_stats[key] = float(sum(values) / len(values))
        self.last_content_cross_attn_aux_stats["content_cross_attn_layer_counted"] = float(len(stats_rows))

    def _content_cross_attn_delta(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            self._active_content_cross_attn_memory is None
            or self.content_cross_attn_layers is None
            or str(layer_idx) not in self.content_cross_attn_layers
        ):
            return None
        layer = self.content_cross_attn_layers[str(layer_idx)]
        out = layer(
            hidden_states,
            self._active_content_cross_attn_memory.to(device=hidden_states.device),
            target_mask.to(device=hidden_states.device),
            None
            if self._active_content_cross_attn_mask is None
            else self._active_content_cross_attn_mask.to(device=hidden_states.device),
        )
        if out.attention_weights is not None:
            self._active_content_cross_attn_attentions.append(out.attention_weights)
        layer_stats = dict(out.stats)
        for key, value in out.stats.items():
            short_key = key.replace("content_cross_attn_", "")
            layer_stats[f"content_cross_attn_layer_{layer_idx}_{short_key}"] = float(value)
        self._active_content_cross_attn_stats.append(layer_stats)
        self.last_content_cross_attn_aux_stats.update(
            {
                key: float(value)
                for key, value in layer_stats.items()
                if key.startswith(f"content_cross_attn_layer_{layer_idx}_")
            }
        )
        self._refresh_content_cross_attn_stats()
        return out.delta.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def _content_cross_attn_aux_loss(
        self,
        target_position_mask: torch.Tensor | None,
        content_cross_attn_mask: torch.Tensor | None = None,
        *,
        content_cross_attn_memory: torch.Tensor | None = None,
        content_token_ids: torch.Tensor | None = None,
        content_token_ids_mask: torch.Tensor | None = None,
        source_content_ids: torch.Tensor | None = None,
        source_content_ids_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        self.last_content_cross_attn_aux_loss = None
        stats = dict(self.last_content_cross_attn_aux_stats)
        stats["content_cross_attn_enabled"] = 1.0 if self.content_cross_attn_encoder is not None else 0.0
        stats["content_cross_attn_attention_layers"] = float(len(self._active_content_cross_attn_attentions))
        if target_position_mask is None:
            self.last_content_cross_attn_aux_stats = stats
            return None
        terms: list[torch.Tensor] = []
        guided_weight = float(self.timbre_memory_config.guided_attn_loss_weight)
        warmup_steps = int(self.timbre_memory_config.guided_attn_warmup_steps)
        if guided_weight > 0.0:
            if warmup_steps > 0:
                step = max(0, int(self._content_guided_attn_runtime_step))
                warmup = min(1.0, float(step) / float(max(1, warmup_steps)))
            else:
                warmup = 1.0
            guided_loss, guided_stats = compute_guided_attention_loss(
                self._active_content_cross_attn_attentions,
                target_position_mask,
                content_cross_attn_mask,
                band_frames=int(self.timbre_memory_config.guided_attn_band_frames),
            )
            stats.update(guided_stats)
            stats["content_guided_attn_weight"] = guided_weight
            stats["content_guided_attn_warmup"] = float(warmup)
            if guided_loss is not None and warmup > 0.0:
                terms.append(guided_loss * guided_loss.new_tensor(guided_weight * warmup))
        phoneme_weight = float(self.timbre_memory_config.phoneme_classifier_loss_weight)
        if phoneme_weight > 0.0 and self.content_phoneme_classifier is not None and content_cross_attn_memory is not None:
            token_ids = content_token_ids if content_token_ids is not None else source_content_ids
            token_mask = content_token_ids_mask if content_token_ids is not None else source_content_ids_mask
            logits = self.content_phoneme_classifier(
                content_cross_attn_memory.to(device=next(self.content_phoneme_classifier.parameters()).device)
            )
            phoneme_loss, phoneme_stats = compute_phoneme_classifier_loss(logits, token_ids, token_mask)
            stats.update(phoneme_stats)
            stats["content_phoneme_classifier_weight"] = phoneme_weight
            if phoneme_loss is not None:
                terms.append(phoneme_loss * phoneme_loss.new_tensor(phoneme_weight))
        self.last_content_cross_attn_aux_stats = stats
        if not terms:
            return None
        loss = torch.stack(terms).sum()
        stats["content_cross_attn_aux_loss_weighted"] = float(loss.detach().item())
        self.last_content_cross_attn_aux_loss = float(loss.detach().item())
        return loss

    @staticmethod
    def _align_time_mask(mask: torch.Tensor | None, shape: tuple[int, int]) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.shape == shape:
            return mask.bool()
        if mask.dim() != 2 or mask.shape[0] != shape[0]:
            raise ValueError(f"mask shape {tuple(mask.shape)} cannot align to {shape}")
        if mask.shape[1] < shape[1]:
            pad_len = shape[1] - mask.shape[1]
            mask = F.pad(mask.bool(), (pad_len, 0), value=False)
        elif mask.shape[1] > shape[1]:
            mask = mask[:, -shape[1] :]
        return mask.bool()

    @staticmethod
    def _align_time_values(
        values: torch.Tensor | None,
        shape: tuple[int, int],
        *,
        padding_value: int = TEXT_OR_OTHER,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        if values.shape == shape:
            return values.long()
        if values.dim() != 2 or values.shape[0] != shape[0]:
            raise ValueError(f"value shape {tuple(values.shape)} cannot align to {shape}")
        if values.shape[1] < shape[1]:
            pad_len = shape[1] - values.shape[1]
            values = F.pad(values.long(), (pad_len, 0), value=int(padding_value))
        elif values.shape[1] > shape[1]:
            values = values[:, -shape[1] :]
        return values.long()

    def _build_role_ids(
        self,
        input_ids: torch.Tensor,
        role_ids: torch.Tensor | None,
        source_prompt_positions: torch.Tensor | None,
        timbre_ref_prompt_positions: torch.Tensor | None,
        target_position_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.role_router is None:
            return None
        shape = input_ids.shape[:2]
        if role_ids is not None:
            if role_ids.shape != shape:
                role_ids = self._align_time_values(role_ids, shape)
            return role_ids.to(device=input_ids.device, dtype=torch.long)
        source_prompt_positions = self._align_time_mask(source_prompt_positions, shape)
        timbre_ref_prompt_positions = self._align_time_mask(timbre_ref_prompt_positions, shape)
        target_position_mask = self._align_time_mask(target_position_mask, shape)
        return build_role_ids(
            shape,
            source_positions=source_prompt_positions,
            ref_positions=timbre_ref_prompt_positions,
            target_positions=target_position_mask,
            device=input_ids.device,
        )

    @staticmethod
    def _select_padded_positions(
        embeddings: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(device=embeddings.device).bool()
        batch_size, _, hidden_size = embeddings.shape
        lengths = positions.sum(dim=1)
        max_len = max(1, int(lengths.max().item()))
        selected = embeddings.new_zeros((batch_size, max_len, hidden_size))
        selected_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=embeddings.device)
        for batch_idx in range(batch_size):
            cur = embeddings[batch_idx, positions[batch_idx]]
            if cur.numel() == 0:
                continue
            selected[batch_idx, : cur.shape[0]] = cur
            selected_mask[batch_idx, : cur.shape[0]] = True
        return selected, selected_mask

    def _compute_source_prosody_tokens(
        self,
        input_ids: torch.Tensor,
        role_ids: torch.Tensor | None,
        source_prompt_positions: torch.Tensor | None,
    ) -> torch.Tensor | None:
        self.last_prosody_memory_shape = None
        if self.source_prosody_encoder is None or source_prompt_positions is None:
            return None
        source_prompt_positions = self._align_time_mask(source_prompt_positions, input_ids.shape[:2])
        if source_prompt_positions is None:
            return None
        base_model = self.get_base_model()
        if self.role_router is not None:
            audio_embeddings = self.role_router.compute_audio_embeddings(base_model, input_ids, role_ids)
        else:
            audio_embeddings = None
            for idx, embed_layer in enumerate(base_model.emb_ext):
                cur = embed_layer(input_ids[..., idx + 1])
                audio_embeddings = cur if audio_embeddings is None else audio_embeddings + cur
            if audio_embeddings is None:
                return None
        source_embeddings, source_mask = self._select_padded_positions(audio_embeddings, source_prompt_positions)
        prosody_tokens = self.source_prosody_encoder(source_embeddings, source_mask)
        self.last_prosody_memory_shape = tuple(prosody_tokens.shape)
        return prosody_tokens

    def _source_prosody_batch_gate(
        self,
        vc_mode_id: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        self.last_source_prosody_gate_stats = {}
        no_text_gate = float(self.timbre_memory_config.source_prosody_no_text_gate)
        text_gate = float(self.timbre_memory_config.source_prosody_text_gate)
        if vc_mode_id is None:
            return None
        mode = vc_mode_id.to(device=device).long().view(-1)
        if mode.numel() == 1 and batch_size > 1:
            mode = mode.expand(batch_size)
        if mode.numel() != batch_size:
            return None
        gate = torch.full((batch_size,), no_text_gate, dtype=dtype, device=device)
        text_id = int(self.MODE_TO_ID.get(VC_MODE_TEXT, 1))
        no_text_id = int(self.MODE_TO_ID.get(VC_MODE_NO_TEXT, 2))
        gate = torch.where(
            mode == text_id,
            torch.full_like(gate, text_gate),
            gate,
        )
        self.last_source_prosody_gate_stats = {
            "source_prosody_no_text_gate": no_text_gate,
            "source_prosody_text_gate": text_gate,
            "source_prosody_batch_gate_mean": float(gate.detach().float().mean().item()),
            "source_prosody_text_rows": float(
                (mode == text_id).detach().float().sum().item()
            ),
            "source_prosody_no_text_rows": float(
                (mode == no_text_id).detach().float().sum().item()
            ),
        }
        return gate

    @staticmethod
    def _hidden_states_for_heads(outputs, hidden_out_layers, n_heads: int) -> list[torch.Tensor]:
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Ver2 target head routing requires hidden_states from the base model.")
        if hidden_out_layers is None:
            return [hidden_states[-1]] * n_heads
        return [hidden_states[int(idx)] for idx in hidden_out_layers]

    @staticmethod
    def _compute_delay_loss(
        logits: list[torch.Tensor],
        labels: torch.Tensor | None,
        channelwise_loss_weight=None,
        per_position_loss_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if labels is None:
            return None, None, None, None, None
        if labels.dim() != 3:
            raise ValueError(f"Labels must have rank 3 [B, T, C], got {tuple(labels.shape)}")
        batch_size = int(labels.shape[0])
        all_token_nums = torch.sum(labels != -100, dim=1)
        all_sum_losses_list = []
        for idx, cur_logits in enumerate(logits):
            vocab_size = cur_logits.size(-1)
            cur_labels = labels[..., idx].to(cur_logits.device)
            position_weight = None
            if per_position_loss_weight is not None:
                position_weight = per_position_loss_weight.to(device=cur_logits.device, dtype=cur_logits.dtype)
                if tuple(position_weight.shape) != tuple(cur_labels.shape):
                    raise ValueError(
                        "per_position_loss_weight shape "
                        f"{tuple(position_weight.shape)} != labels channel shape {tuple(cur_labels.shape)}"
                    )
            per_token_loss = F.cross_entropy(
                cur_logits.reshape(-1, vocab_size),
                cur_labels.contiguous().view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(batch_size, -1)
            if position_weight is not None:
                per_token_loss = per_token_loss * position_weight
            all_sum_losses_list.append(per_token_loss.sum(dim=-1))
        all_sum_losses = torch.stack(all_sum_losses_list, dim=1)
        if channelwise_loss_weight is not None:
            if len(channelwise_loss_weight) != len(logits):
                raise ValueError(f"channelwise_loss_weight length {len(channelwise_loss_weight)} != {len(logits)}")
            weights = torch.tensor(channelwise_loss_weight, device=all_sum_losses.device, dtype=all_sum_losses.dtype)
            token_counts_safe = all_token_nums.to(device=all_sum_losses.device, dtype=all_sum_losses.dtype).clamp(min=1.0)
            normalized_losses = all_sum_losses / token_counts_safe
            sample_losses = (normalized_losses * weights).sum(dim=1) / weights.sum().clamp(min=1.0e-8)
            total_loss_per_channel = all_sum_losses.sum(dim=0)
            total_tokens_per_channel = token_counts_safe.sum(dim=0).clamp(min=1.0)
            channel_losses = total_loss_per_channel / total_tokens_per_channel
            loss = (channel_losses * weights).sum() / weights.sum().clamp(min=1.0e-8)
        else:
            total_tokens = all_token_nums.to(device=all_sum_losses.device, dtype=all_sum_losses.dtype).sum().clamp(min=1.0)
            loss = all_sum_losses.sum() / total_tokens
            sample_losses = None
            channel_losses = all_sum_losses.sum(dim=0) / all_token_nums.to(
                device=all_sum_losses.device,
                dtype=all_sum_losses.dtype,
            ).sum(dim=0).clamp(min=1.0)
        return loss, all_sum_losses, all_token_nums, sample_losses, channel_losses

    def _target_front_ce_weight_map(
        self,
        labels: torch.Tensor | None,
        target_position_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        self.last_target_front_ce_stats = {}
        weight = float(getattr(self.timbre_memory_config, "target_front_ce_weight", 1.0))
        seconds = float(getattr(self.timbre_memory_config, "target_front_ce_seconds", 0.0))
        frame_rate = float(getattr(self.timbre_memory_config, "target_front_ce_frame_rate", 12.5))
        if labels is None or target_position_mask is None or weight <= 1.0 or seconds <= 0.0:
            stats = {
                "target_front_ce_weight": weight,
                "target_front_ce_seconds": seconds,
                "target_front_ce_frames": 0.0,
                "target_front_ce_weighted_tokens": 0.0,
            }
            self.last_target_front_ce_stats = stats
            return None, stats
        if labels.dim() != 3:
            raise ValueError(f"Labels must have rank 3 [B, T, C], got {tuple(labels.shape)}")
        target_mask = self._align_time_mask(target_position_mask, labels.shape[:2])
        if target_mask is None:
            return None, {}
        target_mask = target_mask.to(device=labels.device).bool()
        valid_label_rows = (labels != -100).any(dim=-1)
        active_rows = target_mask & valid_label_rows
        front_frames = max(1, int(round(seconds * max(frame_rate, 1.0e-6))))
        position_weight = torch.ones(labels.shape[:2], device=labels.device, dtype=torch.float32)
        selected_rows = torch.zeros(labels.shape[:2], device=labels.device, dtype=torch.bool)
        for batch_idx in range(int(labels.shape[0])):
            idxs = torch.nonzero(active_rows[batch_idx], as_tuple=False).flatten()
            if idxs.numel() == 0:
                continue
            take = min(front_frames, int(idxs.numel()))
            selected_rows[batch_idx, idxs[:take]] = True
        if bool(selected_rows.any().item()):
            position_weight[selected_rows] = weight
        valid_weighted_tokens = (selected_rows.unsqueeze(-1) & (labels != -100)).sum().detach().float()
        stats = {
            "target_front_ce_weight": weight,
            "target_front_ce_seconds": seconds,
            "target_front_ce_frames": float(front_frames),
            "target_front_ce_weighted_rows": float(selected_rows.detach().float().sum().item()),
            "target_front_ce_weighted_tokens": float(valid_weighted_tokens.item()),
        }
        self.last_target_front_ce_stats = stats
        return position_weight, stats

    def _routing_aux_loss(self) -> torch.Tensor | None:
        self.last_route_loss = None
        self.last_route_stats = {}
        if not self.timbre_memory_config.use_role_routing:
            return None
        states = []
        if self.role_router is not None:
            states.append(self.role_router.regularization_loss())
        if self.target_head_router is not None:
            states.append(self.target_head_router.regularization_loss())
        if not states:
            return None
        raw_loss = torch.stack([state.loss for state in states]).sum()
        stats: dict[str, float] = {}
        for state in states:
            stats.update(state.stats)
        stats["route_loss_raw"] = float(raw_loss.detach().item())
        weight = float(self.timbre_memory_config.route_loss_weight)
        weighted = raw_loss * weight
        self.last_route_loss = float(weighted.detach().item())
        stats["route_loss_weighted"] = self.last_route_loss
        self.last_route_stats = stats
        if weight <= 0:
            return None
        return weighted

    def _apply_target_head_routing(
        self,
        outputs,
        *,
        labels: torch.Tensor | None,
        channelwise_loss_weight=None,
        hidden_out_layers=None,
        target_position_mask: torch.Tensor | None,
        prosody_tokens: torch.Tensor | None,
        timbre_tokens: torch.Tensor | None,
        prosody_batch_gate: torch.Tensor | None = None,
    ):
        if self.target_head_router is None:
            return outputs
        base_model = self.get_base_model()
        n_heads = int(base_model.config.n_vq) + 1
        hidden_states_for_heads = self._hidden_states_for_heads(outputs, hidden_out_layers, n_heads)
        logits = self.target_head_router.routed_logits(
            hidden_states_for_heads=hidden_states_for_heads,
            lm_heads=base_model.lm_heads,
            target_position_mask=target_position_mask,
            prosody_tokens=prosody_tokens,
            timbre_tokens=timbre_tokens,
            prosody_batch_gate=prosody_batch_gate,
        )
        target_front_ce_weight_map, _ = self._target_front_ce_weight_map(labels, target_position_mask)
        loss, all_sum_losses, all_token_nums, sample_losses, channel_losses = self._compute_delay_loss(
            logits,
            labels,
            channelwise_loss_weight=channelwise_loss_weight,
            per_position_loss_weight=target_front_ce_weight_map,
        )
        try:
            return outputs.__class__(
                loss=loss,
                all_sum_losses=all_sum_losses,
                all_token_nums=all_token_nums,
                sample_losses=sample_losses,
                channel_losses=channel_losses,
                logits=logits,
                past_key_values=getattr(outputs, "past_key_values", None),
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )
        except TypeError:
            outputs.logits = logits
            outputs.loss = loss
            outputs.all_sum_losses = all_sum_losses
            outputs.all_token_nums = all_token_nums
            outputs.sample_losses = sample_losses
            outputs.channel_losses = channel_losses
            return outputs

    def _resolve_speaker_paths(self, embedding_paths, audio_paths, batch_size: int) -> list[str | None] | None:
        if bool(getattr(self.speaker_encoder, "expects_audio_paths", False)):
            resolved = _as_path_list(audio_paths, batch_size)
            if resolved is not None and any(path for path in resolved):
                return resolved
        return _as_path_list(embedding_paths, batch_size)

    @staticmethod
    def _linear_out_features(module: nn.Module | None) -> int | None:
        if module is None:
            return None
        value = getattr(module, "out_features", None)
        if value is not None:
            return int(value)
        base_layer = getattr(module, "base_layer", None)
        value = getattr(base_layer, "out_features", None)
        if value is not None:
            return int(value)
        weight = getattr(module, "weight", None)
        if torch.is_tensor(weight) and weight.dim() >= 2:
            return int(weight.shape[0])
        weight = getattr(base_layer, "weight", None)
        if torch.is_tensor(weight) and weight.dim() >= 2:
            return int(weight.shape[0])
        return None

    @staticmethod
    def _load_speaker_vec_payload(path: str | Path) -> torch.Tensor:
        path_obj = Path(path).expanduser()
        suffix = path_obj.suffix.lower()
        if suffix == ".npy":
            import numpy as np

            payload = np.load(path_obj)
        elif suffix == ".npz":
            import numpy as np

            npz = np.load(path_obj)
            payload = npz["embedding"] if "embedding" in npz.files else npz[npz.files[0]]
        else:
            payload = load_torch_file(path_obj)
        if isinstance(payload, dict):
            value = None
            for key in ("speaker_vec", "embedding", "speaker_embedding", "emb", "vector"):
                if payload.get(key) is not None:
                    value = payload[key]
                    break
            if value is None:
                raise ValueError(f"No speaker vector found in {path_obj}")
            payload = value
        vec = torch.as_tensor(payload, dtype=torch.float32)
        if vec.dim() > 1:
            vec = vec.reshape(-1, vec.shape[-1]).mean(dim=0)
        if vec.dim() != 1:
            raise ValueError(f"speaker vector must flatten to [D], got {tuple(vec.shape)} from {path_obj}")
        return vec

    def _speaker_vecs_from_paths(
        self,
        paths,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        path_list = _as_path_list(paths, batch_size)
        if path_list is None or not any(path for path in path_list):
            return None, None
        expected_dim = int(self.timbre_memory_config.speaker_embedding_dim)
        rows: list[torch.Tensor] = []
        valid: list[bool] = []
        for path in path_list:
            if not path:
                rows.append(torch.zeros(expected_dim, dtype=torch.float32))
                valid.append(False)
                continue
            vec = self._load_speaker_vec_payload(path)
            if int(vec.numel()) != expected_dim:
                raise ValueError(
                    f"speaker_vec_path dimension mismatch for {path}: "
                    f"got {int(vec.numel())}, expected {expected_dim}"
                )
            rows.append(F.normalize(vec.float(), dim=0))
            valid.append(True)
        return (
            torch.stack(rows, dim=0).to(device=device, dtype=dtype),
            torch.tensor(valid, device=device, dtype=torch.bool),
        )

    def _prepare_speaker_side_condition(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        speaker_vec: torch.Tensor | None = None,
        speaker_vec_mask: torch.Tensor | None = None,
        speaker_vec_path=None,
        speaker_seq_features: torch.Tensor | None = None,
        speaker_seq_features_mask: torch.Tensor | None = None,
        timbre_ref_speaker_embedding_path=None,
        timbre_ref_speaker_audio_path=None,
        force_drop_speaker_condition: bool = False,
    ) -> None:
        self._active_speaker_side_embedding = None
        self._active_speaker_side_mask = None
        self._active_speaker_side_kv = {}
        self._active_speaker_cross_attn_tokens = None
        self._active_speaker_cross_attn_mask = None
        self._active_speaker_cross_attn_stats = []
        self.last_speaker_side_stats = {}
        speaker_side_enabled = bool(self.timbre_memory_config.speaker_side_pathway_enabled)
        speaker_cross_attn_enabled = bool(self.timbre_memory_config.speaker_cross_attn_enabled)
        speaker_cross_attn_source = str(self.timbre_memory_config.speaker_cross_attn_source or "vector").strip().lower()
        if not speaker_side_enabled and not speaker_cross_attn_enabled:
            return
        if self.null_speaker_embedding is None:
            return
        if speaker_side_enabled and self.speaker_side_adaln is None:
            return
        if speaker_cross_attn_enabled and (
            (
                speaker_cross_attn_source == "vector"
                and self.speaker_cross_attn_tokens is None
            )
            or (
                speaker_cross_attn_source == "sequence"
                and self.speaker_cross_attn_seq_projector is None
            )
            or self.speaker_cross_attn_layers is None
        ):
            return

        speaker_embedding = None
        speaker_mask = None
        speaker_encoder_type = str(getattr(self.timbre_memory_config, "speaker_encoder_type", "") or "")
        prefer_online_speaker_encoder = bool(
            getattr(self.speaker_encoder, "expects_audio_paths", False)
        ) and speaker_encoder_type in {"seed_tts_eval_ecapa", "speechbrain_ecapa"}

        def try_online_speaker_encoder() -> tuple[torch.Tensor | None, torch.Tensor | None]:
            paths = self._resolve_speaker_paths(
                timbre_ref_speaker_embedding_path,
                timbre_ref_speaker_audio_path,
                batch_size,
            )
            embedding, mask = self.speaker_encoder(paths, device=device, dtype=dtype)
            if embedding is not None:
                embedding = F.normalize(embedding.float(), dim=-1).to(dtype=dtype)
            return embedding, mask

        if force_drop_speaker_condition:
            speaker_embedding = self.null_speaker_embedding.to(device=device, dtype=dtype).view(1, -1).expand(
                batch_size,
                -1,
            )
            speaker_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        elif speaker_vec is not None:
            speaker_embedding = speaker_vec.to(device=device, dtype=torch.float32)
            if speaker_embedding.dim() == 1:
                speaker_embedding = speaker_embedding.view(1, -1)
            if speaker_embedding.shape[0] == 1 and batch_size > 1:
                speaker_embedding = speaker_embedding.expand(batch_size, -1)
            if speaker_embedding.shape[0] != batch_size:
                raise ValueError(
                    f"speaker_vec batch mismatch: got {speaker_embedding.shape[0]}, expected {batch_size}"
                )
            expected_dim = int(self.timbre_memory_config.speaker_embedding_dim)
            if speaker_embedding.shape[-1] != expected_dim:
                raise ValueError(
                    f"speaker_vec dim mismatch: got {speaker_embedding.shape[-1]}, expected {expected_dim}"
                )
            speaker_embedding = F.normalize(speaker_embedding.float(), dim=-1).to(dtype=dtype)
            if speaker_vec_mask is None:
                speaker_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
            else:
                speaker_mask = speaker_vec_mask.to(device=device).bool().view(-1)
                if speaker_mask.numel() == 1 and batch_size > 1:
                    speaker_mask = speaker_mask.expand(batch_size)
        if speaker_embedding is None and prefer_online_speaker_encoder:
            speaker_embedding, speaker_mask = try_online_speaker_encoder()
        if speaker_embedding is None:
            speaker_embedding, speaker_mask = self._speaker_vecs_from_paths(
                speaker_vec_path,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        if speaker_embedding is None:
            speaker_embedding, speaker_mask = try_online_speaker_encoder()
        if speaker_embedding is None or speaker_mask is None:
            self.last_speaker_side_stats = {
                "speaker_side_enabled": 1.0,
                "speaker_side_missing": 1.0,
            }
            return

        speaker_mask = speaker_mask.to(device=device).bool().view(-1)
        if speaker_mask.numel() == 1 and batch_size > 1:
            speaker_mask = speaker_mask.expand(batch_size)
        if speaker_mask.numel() != batch_size:
            raise ValueError(f"speaker_vec_mask batch mismatch: got {speaker_mask.numel()}, expected {batch_size}")
        speaker_embedding = speaker_embedding.to(device=device, dtype=dtype)
        null_embedding = self.null_speaker_embedding.to(device=device, dtype=dtype)
        dropped_rate = 0.0
        drop = torch.zeros(batch_size, device=device, dtype=torch.bool)
        drop_prob = max(
            float(self.timbre_memory_config.speaker_side_pathway_dropout),
            float(self.timbre_memory_config.speaker_condition_dropout)
            if speaker_cross_attn_enabled
            else 0.0,
        )
        if force_drop_speaker_condition:
            drop = speaker_mask.clone()
        elif self.training and drop_prob > 0.0:
            drop = (torch.rand((batch_size,), device=device) < drop_prob) & speaker_mask
            if bool(drop.any().item()):
                speaker_embedding = speaker_embedding.clone()
                speaker_embedding[drop] = null_embedding
                dropped_rate = float(drop.float().mean().detach().item())
        speaker_embedding = torch.where(
            speaker_mask.view(batch_size, 1),
            speaker_embedding,
            null_embedding.view(1, -1).expand(batch_size, -1),
        )
        self._active_speaker_side_embedding = speaker_embedding
        self._active_speaker_side_mask = speaker_mask

        if speaker_cross_attn_enabled and speaker_cross_attn_source == "vector" and self.speaker_cross_attn_tokens is not None:
            cross_tokens = self.speaker_cross_attn_tokens(
                speaker_embedding.to(dtype=self.speaker_cross_attn_tokens.net[0].weight.dtype)
            ).to(device=device, dtype=dtype)
            valid = speaker_mask.to(device=device, dtype=dtype).view(batch_size, 1, 1)
            if bool(drop.any().item()) or force_drop_speaker_condition:
                valid = valid * (~drop).to(device=device, dtype=dtype).view(batch_size, 1, 1)
            self._active_speaker_cross_attn_tokens = cross_tokens * valid
            self._active_speaker_cross_attn_mask = (speaker_mask & ~drop).to(device=device).bool()
        elif (
            speaker_cross_attn_enabled
            and speaker_cross_attn_source == "sequence"
            and self.speaker_cross_attn_seq_projector is not None
        ):
            if speaker_seq_features is None:
                raise ValueError("speaker_cross_attn_source=sequence requires speaker_seq_features in the batch")
            sequence = speaker_seq_features.to(device=device, dtype=torch.float32)
            if sequence.dim() == 2:
                sequence = sequence.unsqueeze(0)
            if sequence.shape[0] == 1 and batch_size > 1:
                sequence = sequence.expand(batch_size, -1, -1)
            if sequence.shape[0] != batch_size:
                raise ValueError(
                    f"speaker_seq_features batch mismatch: got {sequence.shape[0]}, expected {batch_size}"
                )
            expected_seq_dim = int(self.timbre_memory_config.speaker_cross_attn_seq_dim)
            if int(sequence.shape[-1]) != expected_seq_dim:
                raise ValueError(
                    f"speaker_seq_features dim mismatch: got {sequence.shape[-1]}, expected {expected_seq_dim}"
                )
            if speaker_seq_features_mask is None:
                sequence_mask = torch.ones(sequence.shape[:2], device=device, dtype=torch.bool)
            else:
                sequence_mask = speaker_seq_features_mask.to(device=device).bool()
                if sequence_mask.dim() == 1:
                    sequence_mask = sequence_mask.unsqueeze(0)
                if sequence_mask.shape[0] == 1 and batch_size > 1:
                    sequence_mask = sequence_mask.expand(batch_size, -1)
                if sequence_mask.shape != sequence.shape[:2]:
                    raise ValueError(
                        f"speaker_seq_features_mask shape {tuple(sequence_mask.shape)} "
                        f"does not match {tuple(sequence.shape[:2])}"
                    )
            sequence_mask = sequence_mask & speaker_mask.view(batch_size, 1)
            if bool(drop.any().item()) or force_drop_speaker_condition:
                sequence_mask = sequence_mask & (~drop).view(batch_size, 1)
            projected = self.speaker_cross_attn_seq_projector(sequence.to(dtype=self.speaker_cross_attn_seq_projector.net[0].weight.dtype))
            valid = sequence_mask.to(device=projected.device, dtype=projected.dtype).unsqueeze(-1)
            self._active_speaker_cross_attn_tokens = projected.to(device=device, dtype=dtype) * valid.to(
                device=device,
                dtype=dtype,
            )
            self._active_speaker_cross_attn_mask = sequence_mask.to(device=device)

        if self.speaker_side_kv_bias is not None and self.speaker_side_gate_logits is not None:
            for key, module in self.speaker_side_kv_bias.items():
                params = module(speaker_embedding.to(dtype=module[0].weight.dtype))
                k_dim, v_dim = self.speaker_side_kv_dims[key]
                k_bias, v_bias = params.split([k_dim, v_dim], dim=-1)
                gate = torch.sigmoid(self.speaker_side_gate_logits[key]).to(device=params.device, dtype=params.dtype)
                valid = speaker_mask.to(device=params.device, dtype=params.dtype).view(batch_size, 1)
                self._active_speaker_side_kv[key] = (
                    (k_bias * gate * valid).to(dtype=dtype),
                    (v_bias * gate * valid).to(dtype=dtype),
                )

        gates = (
            torch.stack([torch.sigmoid(param.detach().float()) for param in self.speaker_side_gate_logits.values()])
            if self.speaker_side_gate_logits is not None and len(self.speaker_side_gate_logits) > 0
            else torch.empty(0)
        )
        self.last_speaker_side_stats = {
            "speaker_side_enabled": 1.0,
            "speaker_side_layers": float(len(self.speaker_side_layer_indices)),
            "speaker_side_valid_rows": float(speaker_mask.detach().float().sum().item()),
            "speaker_side_dropout_rate": dropped_rate,
            "speaker_side_gate_mean": float(gates.mean().item()) if gates.numel() > 0 else 0.0,
            "speaker_side_gate_std": float(gates.std(unbiased=False).item()) if gates.numel() > 1 else 0.0,
            "speaker_side_gate_min": float(gates.min().item()) if gates.numel() > 0 else 0.0,
            "speaker_side_gate_max": float(gates.max().item()) if gates.numel() > 0 else 0.0,
            "speaker_cross_attn_enabled": 1.0 if speaker_cross_attn_enabled else 0.0,
            "speaker_cross_attn_layers": float(len(self.speaker_cross_attn_layer_indices)),
            "speaker_cross_attn_tokens": float(int(self.timbre_memory_config.speaker_cross_attn_tokens)),
            "speaker_cross_attn_source_id": 1.0 if speaker_cross_attn_source == "sequence" else 0.0,
            "speaker_cross_attn_seq_dim": float(int(self.timbre_memory_config.speaker_cross_attn_seq_dim)),
            "speaker_cross_attn_condition_dropout": float(self.timbre_memory_config.speaker_condition_dropout),
            "speaker_cross_attn_output_scale": float(self.timbre_memory_config.speaker_cross_attn_output_scale),
            "speaker_cross_attn_runtime_scale_multiplier": float(
                self.timbre_memory_config.speaker_cross_attn_runtime_scale_multiplier
            ),
            "speaker_cross_attn_token_init_std": (
                float(self.timbre_memory_config.speaker_cross_attn_token_init_std)
                if self.timbre_memory_config.speaker_cross_attn_token_init_std is not None
                else 0.0
            ),
        }
        if self._active_speaker_cross_attn_tokens is not None:
            token_norm = self._active_speaker_cross_attn_tokens.detach().float().norm(dim=-1)
            if self._active_speaker_cross_attn_mask is not None and self._active_speaker_cross_attn_mask.dim() == 2:
                token_mask = self._active_speaker_cross_attn_mask.to(device=token_norm.device).bool()
            else:
                token_mask = (speaker_mask & ~drop).view(batch_size, 1).expand_as(token_norm)
            self.last_speaker_side_stats["speaker_cross_attn_token_norm_mean"] = (
                float(token_norm.masked_select(token_mask).mean().item()) if bool(token_mask.any().item()) else 0.0
            )
            self.last_speaker_side_stats["speaker_cross_attn_valid_tokens"] = (
                float(token_mask.detach().float().sum(dim=1).mean().item()) if token_mask.dim() == 2 else 0.0
            )

    def _speaker_side_adaln_delta(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if (
            self._active_speaker_side_embedding is None
            or self._active_speaker_side_mask is None
            or self.speaker_side_adaln is None
            or self.speaker_side_gate_logits is None
            or str(layer_idx) not in self.speaker_side_adaln
        ):
            return None
        module = self.speaker_side_adaln[str(layer_idx)]
        params = module(self._active_speaker_side_embedding.to(dtype=module[0].weight.dtype))
        scale, shift = params.chunk(2, dim=-1)
        gate = torch.sigmoid(self.speaker_side_gate_logits[str(layer_idx)]).to(device=params.device, dtype=params.dtype)
        scale = torch.tanh(scale) * gate
        shift = shift * gate
        valid = self._active_speaker_side_mask.to(device=params.device, dtype=params.dtype).view(-1, 1, 1)
        return (
            hidden_states * scale.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(1)
            + shift.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(1)
        ) * valid.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def _refresh_speaker_cross_attn_stats(self) -> None:
        stats_rows = self._active_speaker_cross_attn_stats
        if not stats_rows:
            return
        aggregate_keys = (
            "speaker_cross_attn_gate_mean",
            "speaker_cross_attn_delta_norm",
            "speaker_cross_attn_raw_delta_norm",
            "speaker_cross_attn_hidden_norm",
            "speaker_cross_attn_delta_ratio",
            "speaker_cross_attn_output_scale",
            "speaker_cross_attn_runtime_scale_multiplier",
        )
        for key in aggregate_keys:
            values = [float(row[key]) for row in stats_rows if key in row]
            if values:
                self.last_speaker_side_stats[key] = float(sum(values) / len(values))
        self.last_speaker_side_stats["speaker_cross_attn_layer_counted"] = float(len(stats_rows))

    def set_speaker_cross_attn_runtime_scale_multiplier(self, multiplier: float) -> None:
        if self.speaker_cross_attn_layers is None:
            return
        for layer in self.speaker_cross_attn_layers.values():
            if hasattr(layer, "set_runtime_scale_multiplier"):
                layer.set_runtime_scale_multiplier(float(multiplier))

    def _speaker_cross_attn_delta(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            self._active_speaker_cross_attn_tokens is None
            or self._active_speaker_cross_attn_mask is None
            or self.speaker_cross_attn_layers is None
            or str(layer_idx) not in self.speaker_cross_attn_layers
        ):
            return None
        layer = self.speaker_cross_attn_layers[str(layer_idx)]
        out = layer(
            hidden_states,
            self._active_speaker_cross_attn_tokens.to(device=hidden_states.device),
            target_mask.to(device=hidden_states.device),
            self._active_speaker_cross_attn_mask.to(device=hidden_states.device),
        )
        layer_stats = dict(out.stats)
        for key, value in out.stats.items():
            short_key = key.replace("speaker_cross_attn_", "")
            layer_stats[f"speaker_cross_attn_layer_{layer_idx}_{short_key}"] = float(value)
        self._active_speaker_cross_attn_stats.append(layer_stats)
        self.last_speaker_side_stats.update(
            {
                key: float(value)
                for key, value in layer_stats.items()
                if key.startswith(f"speaker_cross_attn_layer_{layer_idx}_")
            }
        )
        self._refresh_speaker_cross_attn_stats()
        return out.delta.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def _make_speaker_side_kv_hook(self, layer_idx: int, which: str):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return output
            pair = self._active_speaker_side_kv.get(str(layer_idx))
            if pair is None:
                return output
            bias = pair[0] if which == "k" else pair[1]
            if output.dim() == 3:
                if output.shape[0] != bias.shape[0] or output.shape[-1] != bias.shape[-1]:
                    return output
                return output + bias.to(device=output.device, dtype=output.dtype).unsqueeze(1)
            if output.dim() == 2 and bias.shape[0] == 1 and output.shape[-1] == bias.shape[-1]:
                return output + bias[0].to(device=output.device, dtype=output.dtype)
            return output

        return hook

    def _forward_with_timbre_tokens(
        self,
        *args,
        timbre_tokens: torch.Tensor | None,
        target_position_mask: torch.Tensor,
        target_position_progress: torch.Tensor | None = None,
        prosody_tokens: torch.Tensor | None = None,
        prosody_batch_gate: torch.Tensor | None = None,
        source_semantic_memory: torch.Tensor | None = None,
        source_semantic_mask: torch.Tensor | None = None,
        content_cross_attn_memory: torch.Tensor | None = None,
        content_cross_attn_mask: torch.Tensor | None = None,
        content_token_ids: torch.Tensor | None = None,
        content_token_ids_mask: torch.Tensor | None = None,
        source_content_ids: torch.Tensor | None = None,
        source_content_ids_mask: torch.Tensor | None = None,
        vc_mode_id: torch.Tensor | None = None,
        role_ids: torch.Tensor | None = None,
        timbre_ref_prompt_positions: torch.Tensor | None = None,
        ref_speaker_prompt_slot_positions: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        channelwise_loss_weight = kwargs.pop("channelwise_loss_weight", None)
        # MossTTSDelayModel always requests hidden states from Qwen for its
        # multi-head projection, so forwarding this kwarg creates a duplicate.
        kwargs.pop("output_hidden_states", None)
        hidden_out_layers = kwargs.get("hidden_out_layers", None)
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if self.role_router is not None and input_ids is not None and role_ids is not None:
            kwargs["inputs_embeds"] = self.role_router.compute_input_embeddings(
                self.get_base_model(),
                input_ids,
                role_ids,
            )
        if (
            input_ids is not None
            and bool(self.timbre_memory_config.ref_speaker_prompt_slot)
            and int(self.timbre_memory_config.ref_speaker_prompt_tokens) > 0
        ):
            role_ids = self._inject_ref_speaker_prompt_embeds(
                kwargs,
                input_ids,
                timbre_ref_prompt_positions,
                ref_speaker_prompt_slot_positions=ref_speaker_prompt_slot_positions,
                role_ids=role_ids,
                ref_speaker_prompt_source_tokens=timbre_tokens,
            )
        base_labels = None if self.target_head_router is not None else labels
        if self.target_head_router is None:
            kwargs["channelwise_loss_weight"] = channelwise_loss_weight
        self.last_forward_debug.update(
            {
                "inner_labels_is_none": labels is None,
                "inner_labels_shape": tuple(labels.shape) if labels is not None else None,
                "inner_target_head_router": self.target_head_router is not None,
                "inner_role_ids_shape": tuple(role_ids.shape) if role_ids is not None else None,
                "inner_prosody_tokens_shape": tuple(prosody_tokens.shape) if prosody_tokens is not None else None,
                "inner_timbre_tokens_shape": tuple(timbre_tokens.shape) if timbre_tokens is not None else None,
                "inner_ref_speaker_prompt_slot_shape": tuple(ref_speaker_prompt_slot_positions.shape)
                if ref_speaker_prompt_slot_positions is not None
                else None,
                "inner_target_progress_shape": tuple(target_position_progress.shape)
                if target_position_progress is not None
                else None,
                "inner_source_semantic_memory_shape": tuple(source_semantic_memory.shape)
                if source_semantic_memory is not None
                else None,
                "inner_content_cross_attn_memory_shape": tuple(content_cross_attn_memory.shape)
                if content_cross_attn_memory is not None
                else None,
            }
        )
        self._active_timbre_tokens = timbre_tokens
        self._active_target_position_mask = target_position_mask.bool()
        self._active_target_position_progress = None if target_position_progress is None else target_position_progress.float()
        self._active_source_semantic_memory = source_semantic_memory
        self._active_source_semantic_mask = source_semantic_mask
        self._active_content_cross_attn_memory = content_cross_attn_memory
        self._active_content_cross_attn_mask = content_cross_attn_mask
        self._active_vc_mode_id = vc_mode_id
        active_dtype = timbre_tokens.dtype if timbre_tokens is not None else next(self.parameters()).dtype
        self._active_ref_speaker_adaln = self._prepare_ref_speaker_adaln(dtype=active_dtype)
        self._active_source_semantic_attentions = []
        self._active_source_semantic_adapter_stats = []
        self._active_content_cross_attn_attentions = []
        self._active_content_cross_attn_stats = []
        try:
            outputs = self._forward_model()(*args, labels=base_labels, **kwargs)
            outputs = self._apply_target_head_routing(
                outputs,
                labels=labels,
                channelwise_loss_weight=channelwise_loss_weight,
                hidden_out_layers=hidden_out_layers,
                target_position_mask=target_position_mask,
                prosody_tokens=prosody_tokens,
                timbre_tokens=timbre_tokens,
                prosody_batch_gate=prosody_batch_gate,
            )
            route_loss = self._routing_aux_loss()
            if self.last_source_prosody_gate_stats:
                self.last_route_stats.update(self.last_source_prosody_gate_stats)
            if route_loss is not None:
                if getattr(outputs, "loss", None) is None:
                    outputs.loss = route_loss
                else:
                    outputs.loss = outputs.loss + route_loss
            source_semantic_loss = self._source_semantic_progress_aux_loss(
                target_position_mask,
                source_semantic_mask,
            )
            if source_semantic_loss is not None:
                if getattr(outputs, "loss", None) is None:
                    outputs.loss = source_semantic_loss
                else:
                    outputs.loss = outputs.loss + source_semantic_loss
            content_cross_attn_loss = self._content_cross_attn_aux_loss(
                target_position_mask,
                content_cross_attn_mask,
                content_cross_attn_memory=content_cross_attn_memory,
                content_token_ids=content_token_ids,
                content_token_ids_mask=content_token_ids_mask,
                source_content_ids=source_content_ids,
                source_content_ids_mask=source_content_ids_mask,
            )
            if content_cross_attn_loss is not None:
                if getattr(outputs, "loss", None) is None:
                    outputs.loss = content_cross_attn_loss
                else:
                    outputs.loss = outputs.loss + content_cross_attn_loss
            self.last_forward_debug.update(
                {
                    "output_loss_is_none": getattr(outputs, "loss", None) is None,
                    "output_logits_len": len(getattr(outputs, "logits", []) or []),
                }
            )
            return outputs
        finally:
            self._active_timbre_tokens = None
            self._active_target_position_mask = None
            self._active_target_position_progress = None
            self._active_source_semantic_memory = None
            self._active_source_semantic_mask = None
            self._active_content_cross_attn_memory = None
            self._active_content_cross_attn_mask = None
            self._active_vc_mode_id = None
            self._active_ref_speaker_adaln = None
            self._active_speaker_side_embedding = None
            self._active_speaker_side_mask = None
            self._active_speaker_side_kv = {}
            self._active_speaker_cross_attn_tokens = None
            self._active_speaker_cross_attn_mask = None
            self._active_speaker_cross_attn_stats = []
            self._active_source_semantic_attentions = []
            self._active_source_semantic_adapter_stats = []
            self._active_content_cross_attn_attentions = []
            self._active_content_cross_attn_stats = []

    def _target_hidden_for_aux(
        self,
        outputs,
        target_position_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states or target_position_mask is None:
            return None
        last_hidden = hidden_states[-1]
        if last_hidden is None or last_hidden.dim() != 3:
            return None
        if target_position_mask.shape[1] != last_hidden.shape[1]:
            target_position_mask = target_position_mask[:, -last_hidden.shape[1] :]
        return self._select_padded_positions(last_hidden, target_position_mask)

    def _pooled_last_hidden_for_mask(
        self,
        outputs,
        position_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states or position_mask is None:
            return None
        last_hidden = hidden_states[-1]
        if last_hidden is None or last_hidden.dim() != 3:
            return None
        position_mask = self._align_time_mask(position_mask, last_hidden.shape[:2])
        if position_mask is None:
            return None
        mask = position_mask.to(device=last_hidden.device, dtype=last_hidden.dtype).unsqueeze(-1)
        lengths = mask.sum(dim=1)
        valid = lengths.squeeze(-1) > 0
        pooled = (last_hidden * mask).sum(dim=1) / lengths.clamp(min=1.0)
        return pooled, valid

    @staticmethod
    def _pool_sequence_embeddings(
        embeddings: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        mask = mask.to(device=embeddings.device).bool()
        weights = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        lengths = weights.sum(dim=1)
        valid = lengths.squeeze(-1) > 0
        pooled = (embeddings * weights).sum(dim=1) / lengths.clamp(min=1.0)
        return pooled, valid

    def _prosody_aux_loss(
        self,
        outputs,
        target_position_mask: torch.Tensor | None,
        *,
        vc_mode_id: torch.Tensor | None = None,
        source_logf0: torch.Tensor | None = None,
        source_logf0_mask: torch.Tensor | None = None,
        source_voiced_mask: torch.Tensor | None = None,
        source_voiced_mask_mask: torch.Tensor | None = None,
        source_energy: torch.Tensor | None = None,
        source_energy_mask: torch.Tensor | None = None,
        source_pause_mask: torch.Tensor | None = None,
        source_pause_mask_mask: torch.Tensor | None = None,
        source_duration: torch.Tensor | None = None,
        source_duration_mask: torch.Tensor | None = None,
        target_logf0: torch.Tensor | None = None,
        target_logf0_mask: torch.Tensor | None = None,
        target_voiced_mask: torch.Tensor | None = None,
        target_voiced_mask_mask: torch.Tensor | None = None,
        target_energy: torch.Tensor | None = None,
        target_energy_mask: torch.Tensor | None = None,
        target_pause_mask: torch.Tensor | None = None,
        target_pause_mask_mask: torch.Tensor | None = None,
        target_duration: torch.Tensor | None = None,
        target_duration_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        self.last_prosody_aux_loss = None
        self.last_prosody_aux_stats = {}
        if self.prosody_head is None or self.timbre_memory_config.prosody_loss_weight <= 0:
            return None
        selected = self._target_hidden_for_aux(outputs, target_position_mask)
        if selected is None:
            return None
        target_hidden, target_hidden_mask = selected
        predictions = self.prosody_head(target_hidden, target_hidden_mask)

        def select_batch(value: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor | None:
            if value is None:
                return None
            if value.shape[0] != mask.shape[0]:
                return value
            return value[mask.to(device=value.device)]

        def select_predictions(mask: torch.Tensor) -> dict[str, torch.Tensor]:
            return {key: select_batch(value, mask) for key, value in predictions.items() if value is not None}

        def prefixed_stat_name(label: str, stat_name: str) -> str:
            if stat_name.startswith("prosody_"):
                return f"prosody_{label}_{stat_name[len('prosody_'):]}"
            return f"prosody_{label}_{stat_name}"

        def compute_state(mask: torch.Tensor, *, label: str, use_target: bool) -> AuxiliaryLossState | None:
            if not bool(mask.any().item()):
                return None
            state = compute_prosody_proxy_loss(
                select_predictions(mask),
                source_logf0=select_batch(target_logf0 if use_target else source_logf0, mask),
                source_logf0_mask=select_batch(target_logf0_mask if use_target else source_logf0_mask, mask),
                source_voiced_mask=select_batch(target_voiced_mask if use_target else source_voiced_mask, mask),
                source_energy=select_batch(target_energy if use_target else source_energy, mask),
                source_energy_mask=select_batch(target_energy_mask if use_target else source_energy_mask, mask),
                source_pause_mask=select_batch(target_pause_mask if use_target else source_pause_mask, mask),
                source_duration=select_batch(target_duration if use_target else source_duration, mask),
                source_duration_mask=select_batch(target_duration_mask if use_target else source_duration_mask, mask),
                f0_weight=self.timbre_memory_config.prosody_f0_weight,
                voiced_weight=self.timbre_memory_config.prosody_voiced_weight,
                energy_weight=self.timbre_memory_config.prosody_energy_weight,
                pause_weight=self.timbre_memory_config.prosody_pause_weight,
                duration_weight=self.timbre_memory_config.prosody_duration_weight,
                normalize_f0=self.timbre_memory_config.prosody_normalize_f0,
                normalize_energy=self.timbre_memory_config.prosody_normalize_energy,
            )
            if state.loss is None:
                return None
            state.stats["prosody_teacher_is_target"] = 1.0 if use_target else 0.0
            state.stats["prosody_teacher_label_id"] = 1.0 if label == "target" else 0.0
            return state

        if vc_mode_id is not None and vc_mode_id.shape[0] == target_hidden.shape[0]:
            mode_ids = vc_mode_id.to(device=target_hidden.device).long().view(-1)
            no_text_id = int(self.MODE_TO_ID.get(VC_MODE_NO_TEXT, 2))
            source_mask = mode_ids.eq(no_text_id)
            target_mask = ~source_mask
            states: list[tuple[str, torch.Tensor, AuxiliaryLossState]] = []
            source_state = compute_state(source_mask, label="source", use_target=False)
            if source_state is not None:
                states.append(("source", source_mask, source_state))
            target_state = compute_state(target_mask, label="target", use_target=True)
            if target_state is not None:
                states.append(("target", target_mask, target_state))
            if not states:
                self.last_prosody_aux_stats = {
                    "prosody_mode_aware_source_samples": float(source_mask.detach().float().sum().item()),
                    "prosody_mode_aware_target_samples": float(target_mask.detach().float().sum().item()),
                    "prosody_mode_aware_skipped": 1.0,
                }
                return None
            weighted_terms = []
            total_count = 0.0
            for _label, mask, state in states:
                count = float(mask.detach().float().sum().item())
                total_count += count
                weighted_terms.append(state.loss * state.loss.new_tensor(count))
            raw_loss = torch.stack(weighted_terms).sum() / max(total_count, 1.0)
            weighted = raw_loss * float(self.timbre_memory_config.prosody_loss_weight)
            stats: dict[str, float] = {
                "prosody_mode_aware_loss": float(raw_loss.detach().item()),
                "prosody_loss_raw": float(raw_loss.detach().item()),
                "prosody_mode_aware_source_samples": float(source_mask.detach().float().sum().item()),
                "prosody_mode_aware_target_samples": float(target_mask.detach().float().sum().item()),
            }
            all_stat_names = set()
            for _label, _mask, state in states:
                all_stat_names.update(state.stats.keys())
            for stat_name in sorted(all_stat_names):
                if stat_name == "prosody_loss_raw":
                    continue
                stat_total = 0.0
                stat_count = 0.0
                for _label, mask, state in states:
                    if stat_name not in state.stats:
                        continue
                    count = float(mask.detach().float().sum().item())
                    stat_total += float(state.stats[stat_name]) * count
                    stat_count += count
                if stat_count > 0:
                    stats[stat_name] = stat_total / stat_count
            for label, _mask, state in states:
                for stat_name, stat_value in state.stats.items():
                    stats[prefixed_stat_name(label, stat_name)] = float(stat_value)
            stats["prosody_loss_weighted"] = float(weighted.detach().item())
            self.last_prosody_aux_loss = float(weighted.detach().item())
            self.last_prosody_aux_stats = stats
            return weighted

        state = compute_prosody_proxy_loss(
            predictions,
            source_logf0=source_logf0,
            source_logf0_mask=source_logf0_mask,
            source_voiced_mask=source_voiced_mask,
            source_energy=source_energy,
            source_energy_mask=source_energy_mask,
            source_pause_mask=source_pause_mask,
            source_duration=source_duration,
            source_duration_mask=source_duration_mask,
            f0_weight=self.timbre_memory_config.prosody_f0_weight,
            voiced_weight=self.timbre_memory_config.prosody_voiced_weight,
            energy_weight=self.timbre_memory_config.prosody_energy_weight,
            pause_weight=self.timbre_memory_config.prosody_pause_weight,
            duration_weight=self.timbre_memory_config.prosody_duration_weight,
            normalize_f0=self.timbre_memory_config.prosody_normalize_f0,
            normalize_energy=self.timbre_memory_config.prosody_normalize_energy,
        )
        _ = source_voiced_mask_mask, source_pause_mask_mask, target_voiced_mask_mask, target_pause_mask_mask
        if state.loss is None:
            return None
        weighted = state.loss * float(self.timbre_memory_config.prosody_loss_weight)
        stats = dict(state.stats)
        stats["prosody_loss_weighted"] = float(weighted.detach().item())
        self.last_prosody_aux_loss = float(weighted.detach().item())
        self.last_prosody_aux_stats = stats
        return weighted

    def _content_aux_loss(
        self,
        outputs,
        target_position_mask: torch.Tensor | None,
        *,
        source_content_embedding: torch.Tensor | None = None,
        source_content_embedding_mask: torch.Tensor | None = None,
        target_content_embedding: torch.Tensor | None = None,
        target_content_embedding_mask: torch.Tensor | None = None,
        source_content_ids: torch.Tensor | None = None,
        source_content_ids_mask: torch.Tensor | None = None,
        target_content_ids: torch.Tensor | None = None,
        target_content_ids_mask: torch.Tensor | None = None,
        content_token_ids: torch.Tensor | None = None,
        content_token_ids_mask: torch.Tensor | None = None,
        source_semantic_units: torch.Tensor | None = None,
        source_semantic_units_mask: torch.Tensor | None = None,
        target_semantic_units: torch.Tensor | None = None,
        target_semantic_units_mask: torch.Tensor | None = None,
        source_semantic_features: torch.Tensor | None = None,
        source_semantic_features_mask: torch.Tensor | None = None,
        target_semantic_features: torch.Tensor | None = None,
        target_semantic_features_mask: torch.Tensor | None = None,
        source_ref_codes: torch.Tensor | None = None,
        source_ref_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        self.last_content_aux_loss = None
        self.last_content_aux_stats = {}
        if (
            self.timbre_memory_config.content_loss_weight <= 0
            or (self.content_head is None and self.content_token_head is None and self.content_codec_head is None)
        ):
            return None
        selected = self._target_hidden_for_aux(outputs, target_position_mask)
        if selected is None:
            return None
        target_hidden, target_hidden_mask = selected
        terms: list[torch.Tensor] = []
        stats: dict[str, float] = {}
        if self.content_head is not None and self.timbre_memory_config.content_embedding_weight > 0:
            prediction = self.content_head(target_hidden, target_hidden_mask)
            positive = str(self.timbre_memory_config.content_positive or "source").strip().lower()
            embedding_states = []
            if positive in {"source", "source_or_target", "source_and_target", "average"}:
                embedding_states.append(
                    compute_content_embedding_loss(
                        prediction,
                        source_content_embedding,
                        source_content_embedding_mask,
                        stat_prefix="content_source_embedding",
                    )
                )
            if positive in {"target", "source_or_target", "source_and_target", "average"}:
                embedding_states.append(
                    compute_content_embedding_loss(
                        prediction,
                        target_content_embedding,
                        target_content_embedding_mask,
                        stat_prefix="content_target_embedding",
                    )
                )
            valid_embedding_losses = [state for state in embedding_states if state.loss is not None]
            if valid_embedding_losses:
                if positive == "source_or_target":
                    embedding_loss = valid_embedding_losses[0].loss
                else:
                    embedding_loss = torch.stack([state.loss for state in valid_embedding_losses]).mean()
                terms.append(float(self.timbre_memory_config.content_embedding_weight) * embedding_loss)
                for state in valid_embedding_losses:
                    stats.update(state.stats)
                stats["content_embedding_loss"] = float(embedding_loss.detach().item())
        if self.content_token_head is not None and self.timbre_memory_config.content_token_weight > 0:
            token_logits = self.content_token_head(target_hidden)
            positive = str(self.timbre_memory_config.content_positive or "source").strip().lower()
            token_states = []
            if positive in {"source", "source_or_target", "source_and_target", "average"}:
                token_states.append(
                    compute_content_token_loss(
                        token_logits,
                        source_content_ids,
                        source_content_ids_mask,
                        stat_prefix="content_source_token",
                    )
                )
            if positive in {"target", "source_or_target", "source_and_target", "average"}:
                token_states.append(
                    compute_content_token_loss(
                        token_logits,
                        target_content_ids,
                        target_content_ids_mask,
                        stat_prefix="content_target_token",
                    )
                )
            valid_token_losses = [state for state in token_states if state.loss is not None]
            if valid_token_losses:
                if positive == "source_or_target":
                    token_loss = valid_token_losses[0].loss
                else:
                    token_loss = torch.stack([state.loss for state in valid_token_losses]).mean()
                terms.append(float(self.timbre_memory_config.content_token_weight) * token_loss)
                for state in valid_token_losses:
                    stats.update(state.stats)
                stats["content_token_loss"] = float(token_loss.detach().item())
        if self.content_codec_head is not None and self.timbre_memory_config.content_source_codec_weight > 0:
            codec_logits = self.content_codec_head(target_hidden)
            codec_state = compute_source_codec_content_loss(
                codec_logits,
                self.content_source_codec_codebooks,
                source_ref_codes,
                source_ref_mask,
                audio_pad_code=int(self.get_base_model().config.audio_pad_code),
            )
            if codec_state.loss is not None:
                terms.append(float(self.timbre_memory_config.content_source_codec_weight) * codec_state.loss)
                stats.update(codec_state.stats)
        if not terms:
            return None
        raw_loss = torch.stack(terms).sum()
        weighted = raw_loss * float(self.timbre_memory_config.content_loss_weight)
        stats["content_loss_raw"] = float(raw_loss.detach().item())
        stats["content_loss_weighted"] = float(weighted.detach().item())
        self.last_content_aux_loss = float(weighted.detach().item())
        self.last_content_aux_stats = stats
        return weighted

    def _content_ctc_aux_loss(
        self,
        outputs,
        target_position_mask: torch.Tensor | None,
        *,
        content_token_ids: torch.Tensor | None = None,
        content_token_ids_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        self.last_content_ctc_aux_loss = None
        self.last_content_ctc_aux_stats = {}
        if self.content_ctc_head is None or self.timbre_memory_config.content_ctc_weight <= 0:
            return None
        selected = self._target_hidden_for_aux(outputs, target_position_mask)
        if selected is None:
            return None
        target_hidden, target_hidden_mask = selected
        logits = self.content_ctc_head(target_hidden)
        state = compute_content_ctc_loss(
            logits,
            target_hidden_mask,
            content_token_ids,
            content_token_ids_mask,
            blank_id=int(self.timbre_memory_config.content_ctc_blank_id),
        )
        if state.loss is None:
            return None
        weighted = state.loss * float(self.timbre_memory_config.content_ctc_weight)
        stats = dict(state.stats)
        stats["content_ctc_loss_weighted"] = float(weighted.detach().item())
        self.last_content_ctc_aux_loss = float(weighted.detach().item())
        self.last_content_ctc_aux_stats = stats
        return weighted

    def _ref_content_suppression_aux_loss(
        self,
        outputs,
        target_position_mask: torch.Tensor | None,
        *,
        timbre_ref_prompt_positions: torch.Tensor | None = None,
        timbre_ref_codes: torch.Tensor | None = None,
        timbre_ref_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        self.last_ref_content_suppression_loss = None
        self.last_ref_content_suppression_stats = {}
        weight = float(self.timbre_memory_config.ref_content_suppression_weight)
        if weight <= 0:
            return None
        target = self._pooled_last_hidden_for_mask(outputs, target_position_mask)
        if target is None:
            self.last_ref_content_suppression_stats = {"ref_content_suppression_skipped": 1.0}
            return None
        target_pooled, target_valid = target
        source = str(self.timbre_memory_config.ref_content_suppression_source or "auto").strip().lower()
        if source not in {"auto", "prompt_hidden", "codec_embedding"}:
            source = "auto"

        ref_pooled = None
        ref_valid = None
        resolved_source = ""
        if source in {"auto", "prompt_hidden"}:
            ref = self._pooled_last_hidden_for_mask(outputs, timbre_ref_prompt_positions)
            if ref is not None and bool(ref[1].any().item()):
                ref_pooled, ref_valid = ref
                resolved_source = "prompt_hidden"
        if ref_pooled is None and source in {"auto", "codec_embedding"} and timbre_ref_codes is not None:
            ref_embeddings = self._embed_timbre_ref_codes(timbre_ref_codes)
            ref_pooled, ref_valid = self._pool_sequence_embeddings(ref_embeddings, timbre_ref_mask)
            resolved_source = "codec_embedding"
        if ref_pooled is None or ref_valid is None:
            self.last_ref_content_suppression_stats = {
                "ref_content_suppression_skipped": 1.0,
                "ref_content_suppression_no_ref": 1.0,
            }
            return None

        joint_mask = target_valid.to(ref_valid.device) & ref_valid
        if not bool(joint_mask.any().item()):
            self.last_ref_content_suppression_stats = {
                "ref_content_suppression_skipped": 1.0,
                "ref_content_suppression_no_joint": 1.0,
            }
            return None
        target_norm = F.normalize(target_pooled[joint_mask].float(), dim=-1)
        ref_values = ref_pooled[joint_mask]
        if bool(self.timbre_memory_config.ref_content_suppression_detach_ref):
            ref_values = ref_values.detach()
        ref_norm = F.normalize(ref_values.float(), dim=-1)
        cosine = F.cosine_similarity(target_norm, ref_norm, dim=-1)
        margin = float(self.timbre_memory_config.ref_content_suppression_margin)
        raw_loss = F.relu(cosine - margin).mean()
        weighted = raw_loss * weight
        stats = {
            "ref_content_suppression_loss": float(raw_loss.detach().item()),
            "ref_content_suppression_loss_weighted": float(weighted.detach().item()),
            "ref_content_cos": float(cosine.detach().mean().item()),
            "ref_content_cos_max": float(cosine.detach().max().item()),
            "ref_content_margin": margin,
            "ref_content_suppression_samples": float(joint_mask.detach().float().sum().item()),
            "ref_content_suppression_source_id": 1.0 if resolved_source == "prompt_hidden" else 2.0,
        }
        self.last_ref_content_suppression_loss = float(weighted.detach().item())
        self.last_ref_content_suppression_stats = stats
        return weighted

    def _semantic_aux_loss(
        self,
        outputs,
        target_position_mask: torch.Tensor | None,
        *,
        vc_mode_id: torch.Tensor | None = None,
        source_semantic_units: torch.Tensor | None = None,
        source_semantic_units_mask: torch.Tensor | None = None,
        target_semantic_units: torch.Tensor | None = None,
        target_semantic_units_mask: torch.Tensor | None = None,
        source_semantic_features: torch.Tensor | None = None,
        source_semantic_features_mask: torch.Tensor | None = None,
        target_semantic_features: torch.Tensor | None = None,
        target_semantic_features_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        self.last_semantic_aux_loss = None
        self.last_semantic_aux_stats = {}
        if self.timbre_memory_config.semantic_loss_weight <= 0:
            return None
        selected = self._target_hidden_for_aux(outputs, target_position_mask)
        if selected is None:
            return None
        target_hidden, _target_hidden_mask = selected
        source = str(self.timbre_memory_config.semantic_source or "source").strip().lower()
        mode_aware = source in {"mode_aware", "auto"}

        def select_batch(value: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor | None:
            if value is None:
                return None
            if value.shape[0] != mask.shape[0]:
                return value
            return value[mask]

        if mode_aware:
            if vc_mode_id is None:
                source = "source"
                mode_aware = False
            else:
                mode_ids = vc_mode_id.to(device=target_hidden.device).long().view(-1)
                no_text_id = int(self.MODE_TO_ID.get(VC_MODE_NO_TEXT, 2))
                source_mask = mode_ids.eq(no_text_id)
                target_mask = ~source_mask
                states = []
                if self.semantic_token_head is not None:
                    logits = self.semantic_token_head(target_hidden)
                    if bool(source_mask.any().item()):
                        states.append(
                            compute_content_token_loss(
                                select_batch(logits, source_mask),
                                select_batch(source_semantic_units, source_mask),
                                select_batch(source_semantic_units_mask, source_mask),
                                stat_prefix="semantic_mode_aware_source_unit",
                            )
                        )
                    if bool(target_mask.any().item()):
                        states.append(
                            compute_content_token_loss(
                                select_batch(logits, target_mask),
                                select_batch(target_semantic_units, target_mask),
                                select_batch(target_semantic_units_mask, target_mask),
                                stat_prefix="semantic_mode_aware_target_unit",
                            )
                        )
                elif self.semantic_feature_head is not None:
                    prediction = self.semantic_feature_head(target_hidden)
                    if bool(source_mask.any().item()):
                        states.append(
                            compute_semantic_feature_loss(
                                select_batch(prediction, source_mask),
                                select_batch(source_semantic_features, source_mask),
                                select_batch(source_semantic_features_mask, source_mask),
                                loss_type=self.timbre_memory_config.semantic_feature_loss_type,
                                stat_prefix="semantic_mode_aware_source_feature",
                            )
                        )
                    if bool(target_mask.any().item()):
                        states.append(
                            compute_semantic_feature_loss(
                                select_batch(prediction, target_mask),
                                select_batch(target_semantic_features, target_mask),
                                select_batch(target_semantic_features_mask, target_mask),
                                loss_type=self.timbre_memory_config.semantic_feature_loss_type,
                                stat_prefix="semantic_mode_aware_target_feature",
                            )
                        )
                else:
                    return None
                valid_states = [state for state in states if state.loss is not None]
                if not valid_states:
                    return None
                raw_loss = torch.stack([state.loss for state in valid_states]).mean()
                weighted = raw_loss * float(self.timbre_memory_config.semantic_loss_weight)
                stats: dict[str, float] = {
                    "semantic_mode_aware_loss": float(raw_loss.detach().item()),
                    "semantic_loss_weighted": float(weighted.detach().item()),
                    "semantic_mode_aware_source_samples": float(source_mask.detach().float().sum().item()),
                    "semantic_mode_aware_target_samples": float(target_mask.detach().float().sum().item()),
                }
                for state in valid_states:
                    stats.update(state.stats)
                self.last_semantic_aux_loss = float(weighted.detach().item())
                self.last_semantic_aux_stats = stats
                return weighted

        use_target = source == "target"
        if self.semantic_token_head is not None:
            units = target_semantic_units if use_target else source_semantic_units
            units_mask = target_semantic_units_mask if use_target else source_semantic_units_mask
            logits = self.semantic_token_head(target_hidden)
            state = compute_content_token_loss(
                logits,
                units,
                units_mask,
                stat_prefix=f"semantic_{source}_unit",
            )
        elif self.semantic_feature_head is not None:
            features = target_semantic_features if use_target else source_semantic_features
            features_mask = target_semantic_features_mask if use_target else source_semantic_features_mask
            prediction = self.semantic_feature_head(target_hidden)
            state = compute_semantic_feature_loss(
                prediction,
                features,
                features_mask,
                loss_type=self.timbre_memory_config.semantic_feature_loss_type,
                stat_prefix=f"semantic_{source}_feature",
            )
        else:
            return None
        if state.loss is None:
            return None
        weighted = state.loss * float(self.timbre_memory_config.semantic_loss_weight)
        stats = dict(state.stats)
        stats["semantic_loss_weighted"] = float(weighted.detach().item())
        self.last_semantic_aux_loss = float(weighted.detach().item())
        self.last_semantic_aux_stats = stats
        return weighted

    def _progress_stop_aux_loss(
        self,
        outputs,
        target_position_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        self.last_progress_stop_aux_loss = None
        self.last_progress_stop_aux_stats = {}
        if self.progress_stop_head is None:
            return None
        if self.timbre_memory_config.progress_loss_weight <= 0 and self.timbre_memory_config.stop_loss_weight <= 0:
            return None
        selected = self._target_hidden_for_aux(outputs, target_position_mask)
        if selected is None:
            return None
        target_hidden, target_hidden_mask = selected
        predictions = self.progress_stop_head(target_hidden)
        state = compute_progress_stop_loss(
            predictions,
            target_hidden_mask,
            progress_weight=float(self.timbre_memory_config.progress_loss_weight),
            stop_weight=float(self.timbre_memory_config.stop_loss_weight),
        )
        if state.loss is None:
            return None
        stats = dict(state.stats)
        stats["progress_stop_loss_weighted"] = float(state.loss.detach().item())
        self.last_progress_stop_aux_loss = float(state.loss.detach().item())
        self.last_progress_stop_aux_stats = stats
        return state.loss

    def _progress_stop_inference_scores(self, outputs) -> dict[str, torch.Tensor] | None:
        if self.progress_stop_head is None:
            return None
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            return None
        last_hidden = hidden_states[-1]
        if last_hidden is None or last_hidden.dim() != 3 or last_hidden.shape[1] <= 0:
            return None
        predictions = self.progress_stop_head(last_hidden[:, -1:, :])
        progress_logits = predictions.get("progress_logits")
        stop_logit = predictions.get("stop_logit")
        if progress_logits is None or stop_logit is None:
            return None
        num_bins = int(progress_logits.shape[-1])
        denom = max(1, num_bins - 1)
        progress_bin = progress_logits.detach().float().argmax(dim=-1).squeeze(-1)
        progress_value = progress_bin.float().div(float(denom)).clamp(0.0, 1.0)
        stop_prob = torch.sigmoid(stop_logit.detach().float()).squeeze(-1)
        return {
            "progress_bin": progress_bin,
            "progress_value": progress_value,
            "stop_prob": stop_prob,
        }

    def _speaker_aux_loss(
        self,
        outputs,
        target_position_mask: torch.Tensor | None = None,
        source_speaker_embedding_path=None,
        timbre_ref_speaker_embedding_path=None,
        target_speaker_embedding_path=None,
        speaker_vec: torch.Tensor | None = None,
        speaker_vec_mask: torch.Tensor | None = None,
        speaker_vec_path=None,
        speaker_seq_features: torch.Tensor | None = None,
        speaker_seq_features_mask: torch.Tensor | None = None,
        source_speaker_audio_path=None,
        timbre_ref_speaker_audio_path=None,
        target_speaker_audio_path=None,
    ) -> torch.Tensor | None:
        self.last_speaker_aux_loss = None
        self.last_speaker_aux_stats = {}
        if self.speaker_projection is None:
            return None
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states or target_position_mask is None:
            return None
        last_hidden = hidden_states[-1]
        if last_hidden is None or last_hidden.dim() != 3:
            return None
        if target_position_mask.shape[1] != last_hidden.shape[1]:
            target_position_mask = target_position_mask[:, -last_hidden.shape[1] :]
        mask = target_position_mask.to(device=last_hidden.device, dtype=last_hidden.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (last_hidden * mask).sum(dim=1) / denom
        pred = F.normalize(self.speaker_projection(pooled).float(), dim=-1)

        batch_size = pred.shape[0]
        timbre_paths = self._resolve_speaker_paths(
            timbre_ref_speaker_embedding_path,
            timbre_ref_speaker_audio_path,
            batch_size,
        )
        target_paths = self._resolve_speaker_paths(target_speaker_embedding_path, target_speaker_audio_path, batch_size)
        source_paths = self._resolve_speaker_paths(source_speaker_embedding_path, source_speaker_audio_path, batch_size)

        aux_terms = []
        stats: dict[str, float] = {}

        def speaker_sim(paths) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            if paths is None:
                return None, None
            embedding, mask = self.speaker_encoder(paths, device=last_hidden.device, dtype=pred.dtype)
            if embedding is None or mask is None:
                return None, None
            return F.cosine_similarity(pred, embedding.float(), dim=-1), mask.to(device=last_hidden.device).bool()

        def speaker_embedding(paths) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            if paths is None:
                return None, None
            embedding, mask = self.speaker_encoder(paths, device=last_hidden.device, dtype=pred.dtype)
            if embedding is None or mask is None:
                return None, None
            return F.normalize(embedding.float(), dim=-1), mask.to(device=last_hidden.device).bool()

        need_positive = (
            self.timbre_memory_config.target_speaker_similarity_weight > 0
            or self.timbre_memory_config.source_speaker_suppression_weight > 0
        )
        ref_sim, ref_mask = speaker_sim(timbre_paths) if need_positive else (None, None)
        target_sim, target_mask = speaker_sim(target_paths) if need_positive else (None, None)
        positive_terms = []
        positive_sims = []
        positive_masks = []
        if ref_sim is not None and ref_mask is not None and bool(ref_mask.any().item()):
            positive_terms.append((1.0 - ref_sim[ref_mask]).mean())
            positive_sims.append(ref_sim)
            positive_masks.append(ref_mask)
            stats["ref_speaker_cos"] = float(ref_sim[ref_mask].detach().mean().item())
        if target_sim is not None and target_mask is not None and bool(target_mask.any().item()):
            positive_terms.append((1.0 - target_sim[target_mask]).mean())
            positive_sims.append(target_sim)
            positive_masks.append(target_mask)
            stats["target_speaker_cos"] = float(target_sim[target_mask].detach().mean().item())
        if self.timbre_memory_config.target_speaker_similarity_weight > 0 and positive_terms:
            term = torch.stack(positive_terms).mean()
            aux_terms.append(float(self.timbre_memory_config.target_speaker_similarity_weight) * term)
            valid_cos = []
            if ref_sim is not None and ref_mask is not None and bool(ref_mask.any().item()):
                valid_cos.append(ref_sim[ref_mask].detach())
            if target_sim is not None and target_mask is not None and bool(target_mask.any().item()):
                valid_cos.append(target_sim[target_mask].detach())
            if valid_cos:
                stats["positive_speaker_cos"] = float(torch.cat(valid_cos).mean().item())
            stats["speaker_positive_count"] = float(len(positive_terms))
        if self.timbre_memory_config.speaker_infonce_weight > 0 and timbre_paths is not None:
            ref_emb, ref_valid = speaker_embedding(timbre_paths)
            if ref_emb is not None and ref_valid is not None and bool(ref_valid.any().item()):
                valid_idx = torch.nonzero(ref_valid, as_tuple=False).flatten()
                pred_valid = pred.index_select(0, valid_idx)
                ref_valid_emb = ref_emb.index_select(0, valid_idx)
                temperature = max(1.0e-4, float(self.timbre_memory_config.speaker_infonce_temperature))
                term, logits, labels = self._speaker_infonce_loss_from_embeddings(
                    pred_valid,
                    ref_valid_emb,
                    temperature=temperature,
                )
                aux_terms.append(float(self.timbre_memory_config.speaker_infonce_weight) * term)
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    stats["speaker_infonce_loss"] = float(term.detach().item())
                    stats["speaker_infonce_acc"] = float((preds == labels).float().mean().item())
                    stats["speaker_infonce_batch"] = float(valid_idx.numel())
                    stats["speaker_infonce_denominator"] = float(logits.shape[1])
                    stats["speaker_infonce_negative_count"] = float(max(0, int(logits.shape[1]) - 1))
        if (
            self.timbre_memory_config.source_speaker_suppression_weight > 0
            and source_paths is not None
            and positive_sims
        ):
            source, source_mask = self.speaker_encoder(
                source_paths,
                device=last_hidden.device,
                dtype=pred.dtype,
            )
            if source is not None and source_mask is not None:
                if len(positive_sims) == 1:
                    positive_sim_for_margin = positive_sims[0]
                    positive_mask_for_margin = positive_masks[0]
                else:
                    stacked_masks = torch.stack(positive_masks, dim=0)
                    stacked_sims = torch.stack(positive_sims, dim=0)
                    positive_mask_for_margin = stacked_masks.any(dim=0)
                    floor = torch.full_like(stacked_sims, -1.0e4)
                    positive_sim_for_margin = torch.where(stacked_masks, stacked_sims, floor).max(dim=0).values
                source_mask = source_mask.to(device=last_hidden.device).bool()
                joint_mask = source_mask & positive_mask_for_margin
                if bool(joint_mask.any().item()):
                    source_sim = F.cosine_similarity(pred, source.float(), dim=-1)
                    margin = float(self.timbre_memory_config.speaker_loss_margin)
                    term = F.relu(source_sim[joint_mask] - positive_sim_for_margin[joint_mask] + margin).mean()
                    aux_terms.append(float(self.timbre_memory_config.source_speaker_suppression_weight) * term)
                    stats["source_speaker_cos"] = float(source_sim[joint_mask].detach().mean().item())
                    stats["source_minus_positive_cos"] = float(
                        (source_sim[joint_mask] - positive_sim_for_margin[joint_mask]).detach().mean().item()
                    )
                    if ref_sim is not None and ref_mask is not None:
                        ref_joint = source_mask & ref_mask
                        if bool(ref_joint.any().item()):
                            stats["source_minus_ref_cos"] = float(
                                (source_sim[ref_joint] - ref_sim[ref_joint]).detach().mean().item()
                            )
                    if target_sim is not None and target_mask is not None:
                        target_joint = source_mask & target_mask
                        if bool(target_joint.any().item()):
                            stats["source_minus_target_cos"] = float(
                                (source_sim[target_joint] - target_sim[target_joint]).detach().mean().item()
                            )
        if not aux_terms:
            return None
        aux_loss = torch.stack(aux_terms).sum()
        self.last_speaker_aux_loss = float(aux_loss.detach().item())
        self.last_speaker_aux_stats = stats
        return aux_loss

    def forward(
        self,
        *args,
        timbre_ref_codes: torch.Tensor | None = None,
        timbre_ref_mask: torch.Tensor | None = None,
        target_position_mask: torch.Tensor | None = None,
        source_prompt_positions: torch.Tensor | None = None,
        timbre_ref_prompt_positions: torch.Tensor | None = None,
        ref_speaker_prompt_slot_positions: torch.Tensor | None = None,
        vc_mode_id: torch.Tensor | None = None,
        source_speaker_embedding_path=None,
        timbre_ref_speaker_embedding_path=None,
        target_speaker_embedding_path=None,
        speaker_vec: torch.Tensor | None = None,
        speaker_vec_mask: torch.Tensor | None = None,
        speaker_vec_path=None,
        speaker_seq_features: torch.Tensor | None = None,
        speaker_seq_features_mask: torch.Tensor | None = None,
        source_speaker_audio_path=None,
        timbre_ref_speaker_audio_path=None,
        target_speaker_audio_path=None,
        role_ids: torch.Tensor | None = None,
        source_logf0: torch.Tensor | None = None,
        source_logf0_mask: torch.Tensor | None = None,
        source_voiced_mask: torch.Tensor | None = None,
        source_voiced_mask_mask: torch.Tensor | None = None,
        source_energy: torch.Tensor | None = None,
        source_energy_mask: torch.Tensor | None = None,
        source_pause_mask: torch.Tensor | None = None,
        source_pause_mask_mask: torch.Tensor | None = None,
        source_duration: torch.Tensor | None = None,
        source_duration_mask: torch.Tensor | None = None,
        target_logf0: torch.Tensor | None = None,
        target_logf0_mask: torch.Tensor | None = None,
        target_voiced_mask: torch.Tensor | None = None,
        target_voiced_mask_mask: torch.Tensor | None = None,
        target_energy: torch.Tensor | None = None,
        target_energy_mask: torch.Tensor | None = None,
        target_pause_mask: torch.Tensor | None = None,
        target_pause_mask_mask: torch.Tensor | None = None,
        target_duration: torch.Tensor | None = None,
        target_duration_mask: torch.Tensor | None = None,
        source_content_embedding: torch.Tensor | None = None,
        source_content_embedding_mask: torch.Tensor | None = None,
        target_content_embedding: torch.Tensor | None = None,
        target_content_embedding_mask: torch.Tensor | None = None,
        source_content_ids: torch.Tensor | None = None,
        source_content_ids_mask: torch.Tensor | None = None,
        target_content_ids: torch.Tensor | None = None,
        target_content_ids_mask: torch.Tensor | None = None,
        content_token_ids: torch.Tensor | None = None,
        content_token_ids_mask: torch.Tensor | None = None,
        source_semantic_units: torch.Tensor | None = None,
        source_semantic_units_mask: torch.Tensor | None = None,
        target_semantic_units: torch.Tensor | None = None,
        target_semantic_units_mask: torch.Tensor | None = None,
        source_semantic_features: torch.Tensor | None = None,
        source_semantic_features_mask: torch.Tensor | None = None,
        target_semantic_features: torch.Tensor | None = None,
        target_semantic_features_mask: torch.Tensor | None = None,
        source_ref_codes: torch.Tensor | None = None,
        source_ref_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        self.last_forward_debug = {
            "entry_labels_is_none": labels is None,
            "entry_timbre_ref_codes_is_none": timbre_ref_codes is None,
            "entry_target_position_mask_is_none": target_position_mask is None,
            "use_role_routing": bool(self.timbre_memory_config.use_role_routing),
            "target_head_routing": self.target_head_router is not None,
            "source_semantic_memory_enabled": self.source_semantic_memory_encoder is not None,
            "content_cross_attn_enabled": self.content_cross_attn_encoder is not None,
            "source_content_memory_type": self.source_content_memory_type,
        }
        self.last_source_semantic_aux_loss = None
        self.last_source_semantic_aux_stats = {}
        self.last_source_semantic_memory_shape = None
        self.last_content_cross_attn_aux_loss = None
        self.last_content_cross_attn_aux_stats = {}
        self.last_content_cross_attn_memory_shape = None
        self.last_ref_content_suppression_loss = None
        self.last_ref_content_suppression_stats = {}
        if target_position_mask is None and labels is not None:
            target_position_mask = (labels != -100).any(dim=-1)
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        resolved_role_ids = None
        prosody_tokens = None
        prosody_batch_gate = None
        if input_ids is not None and self.role_router is not None:
            resolved_role_ids = self._build_role_ids(
                input_ids,
                role_ids,
                source_prompt_positions,
                timbre_ref_prompt_positions,
                target_position_mask,
            )
            prosody_tokens = self._compute_source_prosody_tokens(
                input_ids,
                resolved_role_ids,
                source_prompt_positions,
            )
            if prosody_tokens is not None:
                prosody_batch_gate = self._source_prosody_batch_gate(
                    vc_mode_id,
                    batch_size=int(prosody_tokens.shape[0]),
                    device=prosody_tokens.device,
                    dtype=prosody_tokens.dtype,
                )
        side_pathway_enabled = bool(self.timbre_memory_config.speaker_side_pathway_enabled) or bool(
            self.timbre_memory_config.speaker_cross_attn_enabled
        )
        content_cross_attn_enabled = self.content_cross_attn_encoder is not None
        if (
            timbre_ref_codes is None
            and not side_pathway_enabled
            and not content_cross_attn_enabled
        ) or target_position_mask is None:
            if self.role_router is not None and input_ids is not None and resolved_role_ids is not None:
                kwargs["inputs_embeds"] = self.role_router.compute_input_embeddings(
                    self.get_base_model(),
                    input_ids,
                    resolved_role_ids,
                )
            outputs = self._forward_model()(*args, labels=labels, **kwargs)
            route_loss = self._routing_aux_loss()
            if route_loss is not None and getattr(outputs, "loss", None) is not None:
                outputs.loss = outputs.loss + route_loss
            return outputs

        source_semantic_memory = None
        source_semantic_memory_mask = None
        if self.source_semantic_memory_encoder is not None:
            source_semantic_memory, source_semantic_memory_mask = self._compute_source_semantic_memory(
                source_semantic_features,
                source_semantic_features_mask,
                content_token_ids=content_token_ids,
                content_token_ids_mask=content_token_ids_mask,
                source_semantic_units=source_semantic_units,
                source_semantic_units_mask=source_semantic_units_mask,
                source_ref_codes=source_ref_codes,
                source_ref_mask=source_ref_mask,
                input_ids=input_ids,
                source_prompt_positions=source_prompt_positions,
            )
        content_cross_attn_memory = None
        content_cross_attn_mask = None
        if self.content_cross_attn_encoder is not None:
            content_cross_attn_memory, content_cross_attn_mask = self._compute_content_cross_attn_memory(
                source_semantic_features,
                source_semantic_features_mask,
            )
        batch_size = int(target_position_mask.shape[0])
        condition_device = input_ids.device if input_ids is not None else next(self.parameters()).device
        condition_dtype = next(self.parameters()).dtype
        if side_pathway_enabled:
            self._prepare_speaker_side_condition(
                batch_size=batch_size,
                device=condition_device,
                dtype=condition_dtype,
                speaker_vec=speaker_vec,
                speaker_vec_mask=speaker_vec_mask,
                speaker_vec_path=speaker_vec_path,
                speaker_seq_features=speaker_seq_features,
                speaker_seq_features_mask=speaker_seq_features_mask,
                timbre_ref_speaker_embedding_path=timbre_ref_speaker_embedding_path,
                timbre_ref_speaker_audio_path=timbre_ref_speaker_audio_path,
            )
            timbre_tokens = None
        else:
            if self.legacy_timbre_memory_enabled:
                timbre_tokens = self._compute_timbre_tokens(
                    timbre_ref_codes,
                    timbre_ref_mask,
                    timbre_ref_speaker_embedding_path=timbre_ref_speaker_embedding_path,
                    timbre_ref_speaker_audio_path=timbre_ref_speaker_audio_path,
                )
            else:
                timbre_tokens = None
        outputs = self._forward_with_timbre_tokens(
            *args,
            timbre_tokens=timbre_tokens,
            target_position_mask=target_position_mask,
            prosody_tokens=prosody_tokens,
            prosody_batch_gate=prosody_batch_gate,
            source_semantic_memory=source_semantic_memory,
            source_semantic_mask=source_semantic_memory_mask,
            content_cross_attn_memory=content_cross_attn_memory,
            content_cross_attn_mask=content_cross_attn_mask,
            content_token_ids=content_token_ids,
            content_token_ids_mask=content_token_ids_mask,
            source_content_ids=source_content_ids,
            source_content_ids_mask=source_content_ids_mask,
            vc_mode_id=vc_mode_id,
            role_ids=resolved_role_ids,
            timbre_ref_prompt_positions=timbre_ref_prompt_positions,
            ref_speaker_prompt_slot_positions=ref_speaker_prompt_slot_positions,
            labels=labels,
            **kwargs,
        )

        aux_loss = self._speaker_aux_loss(
            outputs,
            target_position_mask=target_position_mask,
            source_speaker_embedding_path=source_speaker_embedding_path,
            timbre_ref_speaker_embedding_path=timbre_ref_speaker_embedding_path,
            target_speaker_embedding_path=target_speaker_embedding_path,
            source_speaker_audio_path=source_speaker_audio_path,
            timbre_ref_speaker_audio_path=timbre_ref_speaker_audio_path,
            target_speaker_audio_path=target_speaker_audio_path,
        )
        if aux_loss is not None and getattr(outputs, "loss", None) is not None:
            outputs.loss = outputs.loss + aux_loss
        prosody_loss = self._prosody_aux_loss(
            outputs,
            target_position_mask=target_position_mask,
            vc_mode_id=vc_mode_id,
            source_logf0=source_logf0,
            source_logf0_mask=source_logf0_mask,
            source_voiced_mask=source_voiced_mask,
            source_voiced_mask_mask=source_voiced_mask_mask,
            source_energy=source_energy,
            source_energy_mask=source_energy_mask,
            source_pause_mask=source_pause_mask,
            source_pause_mask_mask=source_pause_mask_mask,
            source_duration=source_duration,
            source_duration_mask=source_duration_mask,
            target_logf0=target_logf0,
            target_logf0_mask=target_logf0_mask,
            target_voiced_mask=target_voiced_mask,
            target_voiced_mask_mask=target_voiced_mask_mask,
            target_energy=target_energy,
            target_energy_mask=target_energy_mask,
            target_pause_mask=target_pause_mask,
            target_pause_mask_mask=target_pause_mask_mask,
            target_duration=target_duration,
            target_duration_mask=target_duration_mask,
        )
        if prosody_loss is not None and getattr(outputs, "loss", None) is not None:
            outputs.loss = outputs.loss + prosody_loss
        content_loss = self._content_aux_loss(
            outputs,
            target_position_mask=target_position_mask,
            source_content_embedding=source_content_embedding,
            source_content_embedding_mask=source_content_embedding_mask,
            target_content_embedding=target_content_embedding,
            target_content_embedding_mask=target_content_embedding_mask,
            source_content_ids=source_content_ids,
            source_content_ids_mask=source_content_ids_mask,
            target_content_ids=target_content_ids,
            target_content_ids_mask=target_content_ids_mask,
            source_ref_codes=source_ref_codes,
            source_ref_mask=source_ref_mask,
        )
        if content_loss is not None and getattr(outputs, "loss", None) is not None:
            outputs.loss = outputs.loss + content_loss
        content_ctc_loss = self._content_ctc_aux_loss(
            outputs,
            target_position_mask=target_position_mask,
            content_token_ids=content_token_ids,
            content_token_ids_mask=content_token_ids_mask,
        )
        if content_ctc_loss is not None and getattr(outputs, "loss", None) is not None:
            outputs.loss = outputs.loss + content_ctc_loss
        semantic_loss = self._semantic_aux_loss(
            outputs,
            target_position_mask=target_position_mask,
            vc_mode_id=vc_mode_id,
            source_semantic_units=source_semantic_units,
            source_semantic_units_mask=source_semantic_units_mask,
            target_semantic_units=target_semantic_units,
            target_semantic_units_mask=target_semantic_units_mask,
            source_semantic_features=source_semantic_features,
            source_semantic_features_mask=source_semantic_features_mask,
            target_semantic_features=target_semantic_features,
            target_semantic_features_mask=target_semantic_features_mask,
        )
        if semantic_loss is not None and getattr(outputs, "loss", None) is not None:
            outputs.loss = outputs.loss + semantic_loss
        ref_content_suppression_loss = self._ref_content_suppression_aux_loss(
            outputs,
            target_position_mask=target_position_mask,
            timbre_ref_prompt_positions=timbre_ref_prompt_positions,
            timbre_ref_codes=timbre_ref_codes,
            timbre_ref_mask=timbre_ref_mask,
        )
        if ref_content_suppression_loss is not None and getattr(outputs, "loss", None) is not None:
            outputs.loss = outputs.loss + ref_content_suppression_loss
        progress_stop_loss = self._progress_stop_aux_loss(
            outputs,
            target_position_mask=target_position_mask,
        )
        if progress_stop_loss is not None and getattr(outputs, "loss", None) is not None:
            outputs.loss = outputs.loss + progress_stop_loss
        return outputs

    @classmethod
    def from_pretrained_timbre_memory(
        cls,
        model: nn.Module,
        adapter_directory: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        config_overrides: dict[str, Any] | None = None,
    ) -> "MossCodecVCTimbreMemoryWrapper":
        adapter_path = Path(adapter_directory)
        with (adapter_path / "timbre_memory_config.json").open("r", encoding="utf-8") as f:
            raw_config = json.load(f)
        if "encoder_type" not in raw_config:
            raw_config["encoder_type"] = "perceiver"
        if config_overrides:
            saved_dim = int(raw_config.get("speaker_embedding_dim") or 0)
            override_dim = config_overrides.get("speaker_embedding_dim")
            if override_dim is not None and saved_dim > 0 and int(override_dim) != saved_dim:
                raise ValueError(
                    "Cannot override speaker_embedding_dim after training: "
                    f"adapter was trained with {saved_dim}, requested {override_dim}. "
                    "Use a speaker encoder with the same output dimension or retrain the timbre memory adapter."
                )
            raw_config.update({key: value for key, value in config_overrides.items() if value is not None})
        allowed = {field.name for field in fields(TimbreMemoryConfig)}
        config = TimbreMemoryConfig(**{key: value for key, value in raw_config.items() if key in allowed})
        config.enabled = True
        wrapper = cls(model, config)
        try:
            state = torch.load(adapter_path / "timbre_memory_adapter.pt", map_location=map_location, weights_only=True)
        except TypeError:
            state = torch.load(adapter_path / "timbre_memory_adapter.pt", map_location=map_location)
        timbre_memory_state = state.get("timbre_memory")
        if wrapper.timbre_memory is not None:
            if timbre_memory_state is None:
                raise ValueError("Checkpoint is missing timbre_memory weights for an enabled timbre memory adapter.")
            wrapper.timbre_memory.load_state_dict(timbre_memory_state)

        layer_adapter_state = state.get("layer_adapters")
        if len(wrapper.layer_adapters) > 0:
            if layer_adapter_state is None:
                raise ValueError("Checkpoint is missing layer_adapters weights for enabled timbre adapters.")
            wrapper.layer_adapters.load_state_dict(layer_adapter_state)
        elif layer_adapter_state is not None:
            wrapper.layer_adapters.load_state_dict(layer_adapter_state)
        if wrapper.speaker_projection is not None and state.get("speaker_projection") is not None:
            wrapper.speaker_projection.load_state_dict(state["speaker_projection"])
        if wrapper.prosody_head is not None and state.get("prosody_head") is not None:
            wrapper.prosody_head.load_state_dict(state["prosody_head"])
        if wrapper.content_head is not None and state.get("content_head") is not None:
            wrapper.content_head.load_state_dict(state["content_head"])
        if wrapper.content_ctc_head is not None and state.get("content_ctc_head") is not None:
            wrapper.content_ctc_head.load_state_dict(state["content_ctc_head"])
        if wrapper.content_token_head is not None and state.get("content_token_head") is not None:
            wrapper.content_token_head.load_state_dict(state["content_token_head"])
        if wrapper.content_codec_head is not None and state.get("content_codec_head") is not None:
            wrapper.content_codec_head.load_state_dict(state["content_codec_head"])
        if wrapper.semantic_token_head is not None and state.get("semantic_token_head") is not None:
            wrapper.semantic_token_head.load_state_dict(state["semantic_token_head"])
        if wrapper.semantic_feature_head is not None and state.get("semantic_feature_head") is not None:
            wrapper.semantic_feature_head.load_state_dict(state["semantic_feature_head"])
        if wrapper.progress_stop_head is not None and state.get("progress_stop_head") is not None:
            wrapper.progress_stop_head.load_state_dict(state["progress_stop_head"])
        if wrapper.source_semantic_memory_encoder is not None and state.get("source_semantic_memory_encoder") is not None:
            wrapper.source_semantic_memory_encoder.load_state_dict(state["source_semantic_memory_encoder"])
        if (
            wrapper.source_semantic_codec_residual_encoder is not None
            and state.get("source_semantic_codec_residual_encoder") is not None
        ):
            wrapper.source_semantic_codec_residual_encoder.load_state_dict(
                state["source_semantic_codec_residual_encoder"]
            )
        if wrapper.source_semantic_layer_adapters is not None and state.get("source_semantic_layer_adapters") is not None:
            wrapper.source_semantic_layer_adapters.load_state_dict(state["source_semantic_layer_adapters"])
        if wrapper.content_cross_attn_encoder is not None and state.get("content_cross_attn_encoder") is not None:
            wrapper.content_cross_attn_encoder.load_state_dict(state["content_cross_attn_encoder"])
        if wrapper.content_cross_attn_layers is not None and state.get("content_cross_attn_layers") is not None:
            wrapper.content_cross_attn_layers.load_state_dict(state["content_cross_attn_layers"])
        if wrapper.content_phoneme_classifier is not None and state.get("content_phoneme_classifier") is not None:
            wrapper.content_phoneme_classifier.load_state_dict(state["content_phoneme_classifier"])
        if wrapper.role_router is not None and state.get("role_router") is not None:
            wrapper.role_router.load_state_dict(state["role_router"])
        if wrapper.source_prosody_encoder is not None and state.get("source_prosody_encoder") is not None:
            wrapper.source_prosody_encoder.load_state_dict(state["source_prosody_encoder"])
        if wrapper.target_head_router is not None and state.get("target_head_router") is not None:
            wrapper.target_head_router.load_state_dict(state["target_head_router"])
        if wrapper.ref_speaker_prompt is not None and state.get("ref_speaker_prompt") is not None:
            wrapper.ref_speaker_prompt.load_state_dict(state["ref_speaker_prompt"])
        if wrapper.ref_speaker_adaln is not None and state.get("ref_speaker_adaln") is not None:
            wrapper.ref_speaker_adaln.load_state_dict(state["ref_speaker_adaln"])
        if wrapper.speaker_side_adaln is not None and state.get("speaker_side_adaln") is not None:
            wrapper.speaker_side_adaln.load_state_dict(state["speaker_side_adaln"])
        if wrapper.speaker_side_kv_bias is not None and state.get("speaker_side_kv_bias") is not None:
            wrapper.speaker_side_kv_bias.load_state_dict(state["speaker_side_kv_bias"])
        if wrapper.speaker_side_gate_logits is not None and state.get("speaker_side_gate_logits") is not None:
            wrapper.speaker_side_gate_logits.load_state_dict(state["speaker_side_gate_logits"])
        if wrapper.speaker_cross_attn_tokens is not None and state.get("speaker_cross_attn_tokens") is not None:
            wrapper.speaker_cross_attn_tokens.load_state_dict(state["speaker_cross_attn_tokens"])
        if (
            wrapper.speaker_cross_attn_seq_projector is not None
            and state.get("speaker_cross_attn_seq_projector") is not None
        ):
            wrapper.speaker_cross_attn_seq_projector.load_state_dict(state["speaker_cross_attn_seq_projector"])
        if wrapper.speaker_cross_attn_layers is not None and state.get("speaker_cross_attn_layers") is not None:
            wrapper.speaker_cross_attn_layers.load_state_dict(state["speaker_cross_attn_layers"])
        if wrapper.null_speaker_embedding is not None and state.get("null_speaker_embedding") is not None:
            wrapper.null_speaker_embedding.data.copy_(state["null_speaker_embedding"].to(wrapper.null_speaker_embedding))
        if state.get("speaker_infonce_negative_pool") is not None:
            wrapper.speaker_infonce_negative_pool = state["speaker_infonce_negative_pool"].detach().float()
        return wrapper

    @staticmethod
    def _resolve_attention_layer_indices(spec: str | None, num_layers: int) -> list[int]:
        raw = str(spec or "-4,-3,-2,-1").strip()
        out: list[int] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            idx = int(item)
            if idx < 0:
                idx = int(num_layers) + idx
            if 0 <= idx < int(num_layers) and idx not in out:
                out.append(idx)
        if not out and num_layers > 0:
            out.append(int(num_layers) - 1)
        return out

    def _capture_ref_speaker_prompt_attention(
        self,
        attentions,
        *,
        slot_positions: torch.Tensor | None,
        frame_index: int,
        decode_step: int,
        layer_spec: str | None,
    ) -> dict[str, Any] | None:
        if slot_positions is None or attentions is None:
            return None
        if not isinstance(attentions, (tuple, list)) or not attentions:
            return None
        slot_mask = slot_positions.detach().bool()
        layers: dict[str, Any] = {}
        for layer_idx in self._resolve_attention_layer_indices(layer_spec, len(attentions)):
            attn = attentions[layer_idx]
            if not torch.is_tensor(attn) or attn.dim() != 4:
                continue
            probs = attn.detach().float()[:, :, -1, :]
            key_len = int(probs.shape[-1])
            cur_slot = slot_mask.to(device=probs.device)
            if cur_slot.shape[1] < key_len:
                pad = torch.zeros(
                    cur_slot.shape[0],
                    key_len - cur_slot.shape[1],
                    dtype=torch.bool,
                    device=cur_slot.device,
                )
                cur_slot = torch.cat([cur_slot, pad], dim=1)
            elif cur_slot.shape[1] > key_len:
                cur_slot = cur_slot[:, :key_len]
            if cur_slot.shape[0] != probs.shape[0]:
                continue
            slot_mass = (probs * cur_slot[:, None, :].to(dtype=probs.dtype)).sum(dim=-1)
            slot_tokens = int(cur_slot.sum(dim=-1).float().mean().item())
            uniform_baseline = float(slot_tokens / max(1, key_len))
            max_head = float(slot_mass.max().item())
            layers[str(layer_idx)] = {
                "slot_attention_mean": float(slot_mass.mean().item()),
                "slot_attention_max_head": max_head,
                "slot_tokens": slot_tokens,
                "key_len": key_len,
                "uniform_baseline": uniform_baseline,
                "slot_attention_max_head_over_uniform": max_head / max(1e-8, uniform_baseline),
            }
        if not layers:
            return None
        values = [float(item["slot_attention_mean"]) for item in layers.values()]
        max_head_values = [float(item["slot_attention_max_head"]) for item in layers.values()]
        over_uniform_values = [float(item["slot_attention_max_head_over_uniform"]) for item in layers.values()]
        return {
            "frame_index": int(frame_index),
            "decode_step": int(decode_step),
            "slot_attention_mean": float(sum(values) / max(1, len(values))),
            "slot_attention_max_head": float(max(max_head_values)) if max_head_values else 0.0,
            "slot_attention_max_head_over_uniform": float(max(over_uniform_values)) if over_uniform_values else 0.0,
            "layers": layers,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 1000,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        min_audio_tokens: int = 0,
        source_semantic_progress_clock: str = "decode_step",
        source_semantic_release_after_progress: bool = False,
        source_semantic_release_start: float = 1.0,
        timbre_ref_codes: torch.Tensor | None = None,
        timbre_ref_mask: torch.Tensor | None = None,
        timbre_ref_speaker_embedding_path=None,
        timbre_ref_speaker_audio_path=None,
        speaker_vec: torch.Tensor | None = None,
        speaker_vec_mask: torch.Tensor | None = None,
        speaker_vec_path=None,
        speaker_seq_features: torch.Tensor | None = None,
        speaker_seq_features_mask: torch.Tensor | None = None,
        timbre_cfg_scale: float = 1.0,
        role_ids: torch.Tensor | None = None,
        ref_speaker_prompt_slot_positions: torch.Tensor | None = None,
        ref_speaker_prompt_attention_capture_frames: int = 0,
        ref_speaker_prompt_attention_layers: str | None = None,
        vc_mode_id: torch.Tensor | None = None,
        source_semantic_features: torch.Tensor | None = None,
        source_semantic_features_mask: torch.Tensor | None = None,
        content_token_ids: torch.Tensor | None = None,
        content_token_ids_mask: torch.Tensor | None = None,
        source_semantic_units: torch.Tensor | None = None,
        source_semantic_units_mask: torch.Tensor | None = None,
        source_ref_codes: torch.Tensor | None = None,
        source_ref_mask: torch.Tensor | None = None,
    ):
        self.last_ref_speaker_prompt_attention_stats = {}
        self.last_ref_speaker_prompt_slot_stats = {}
        self.last_progress_stop_infer_stats = {}
        side_pathway_enabled = bool(self.timbre_memory_config.speaker_side_pathway_enabled) or bool(
            self.timbre_memory_config.speaker_cross_attn_enabled
        )
        content_cross_attn_enabled = self.content_cross_attn_encoder is not None
        if timbre_ref_codes is None and not side_pathway_enabled and not content_cross_attn_enabled:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                text_temperature=text_temperature,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
            )

        from moss_tts_delay.inference_utils import find_last_equal_C, sample_token
        from tqdm import tqdm

        base_model = self.get_base_model()
        config = base_model.config
        source_semantic_progress_clock = str(source_semantic_progress_clock or "decode_step").strip().lower()
        if source_semantic_progress_clock not in {"decode_step", "gen_slot"}:
            raise ValueError(
                "source_semantic_progress_clock must be 'decode_step' or 'gen_slot', "
                f"got {source_semantic_progress_clock!r}"
            )
        release_start = min(max(float(source_semantic_release_start), 0.0), 1.0)
        if self.source_semantic_layer_adapters is not None:
            for adapter in self.source_semantic_layer_adapters.values():
                adapter.monotonic_release_after_progress = bool(source_semantic_release_after_progress)
                adapter.monotonic_release_start = release_start
        cfg_scale = float(timbre_cfg_scale)
        cfg_timbre_tokens = None
        if abs(cfg_scale - 1.0) > 1.0e-6 and not side_pathway_enabled and self.legacy_timbre_memory_enabled:
            cfg_timbre_tokens = self._compute_timbre_tokens(
                timbre_ref_codes,
                timbre_ref_mask,
                timbre_ref_speaker_embedding_path=timbre_ref_speaker_embedding_path,
                timbre_ref_speaker_audio_path=timbre_ref_speaker_audio_path,
                force_drop_speaker_condition=True,
            )
        if side_pathway_enabled:
            timbre_tokens = None
        elif self.legacy_timbre_memory_enabled:
            timbre_tokens = self._compute_timbre_tokens(
                timbre_ref_codes,
                timbre_ref_mask,
                timbre_ref_speaker_embedding_path=timbre_ref_speaker_embedding_path,
                timbre_ref_speaker_audio_path=timbre_ref_speaker_audio_path,
            )
        else:
            timbre_tokens = None
        prompt_role_ids = None
        prompt_source_positions = None
        prosody_tokens = None
        prosody_batch_gate = None
        if self.role_router is not None:
            if role_ids is None:
                prompt_role_ids = infer_prompt_role_ids_from_audio_spans(
                    input_ids,
                    audio_pad_code=int(config.audio_pad_code),
                )
            else:
                prompt_role_ids = role_ids.to(device=input_ids.device, dtype=torch.long)
                if prompt_role_ids.shape != input_ids.shape[:2]:
                    prompt_role_ids = self._align_time_values(prompt_role_ids, input_ids.shape[:2]).long()
            if (
                ref_speaker_prompt_slot_positions is None
                and bool(self.timbre_memory_config.ref_speaker_prompt_slot)
                and int(self.timbre_memory_config.ref_speaker_prompt_tokens) > 0
            ):
                ref_speaker_prompt_slot_positions = find_ref_speaker_prompt_slot_positions(
                    input_ids,
                    audio_start_token_id=int(config.audio_start_token_id),
                    audio_end_token_id=int(config.audio_end_token_id),
                    audio_gen_slot_token_id=(
                        int(getattr(config, "audio_user_slot_token_id", config.audio_assistant_gen_slot_token_id)),
                        int(config.audio_assistant_gen_slot_token_id),
                    ),
                    token_count=int(self.timbre_memory_config.ref_speaker_prompt_tokens),
                    occurrence=2,
                )
            if ref_speaker_prompt_slot_positions is not None:
                slot_mask_for_roles = ref_speaker_prompt_slot_positions.to(device=prompt_role_ids.device).bool()
                if slot_mask_for_roles.shape != prompt_role_ids.shape:
                    slot_mask_for_roles = self._align_time_mask(slot_mask_for_roles, prompt_role_ids.shape)
                if slot_mask_for_roles is not None:
                    prompt_role_ids = prompt_role_ids.clone()
                    prompt_role_ids[slot_mask_for_roles] = REF_CODEC
            prompt_source_positions = prompt_role_ids == SOURCE_CODEC
            prosody_tokens = self._compute_source_prosody_tokens(
                input_ids,
                prompt_role_ids,
                prompt_source_positions,
            )
            if prosody_tokens is not None:
                prosody_batch_gate = self._source_prosody_batch_gate(
                    vc_mode_id,
                    batch_size=int(prosody_tokens.shape[0]),
                    device=prosody_tokens.device,
                    dtype=prosody_tokens.dtype,
                )
        source_semantic_memory = None
        source_semantic_memory_mask = None
        if self.source_semantic_memory_encoder is not None:
            source_semantic_memory, source_semantic_memory_mask = self._compute_source_semantic_memory(
                source_semantic_features,
                source_semantic_features_mask,
                content_token_ids=content_token_ids,
                content_token_ids_mask=content_token_ids_mask,
                source_semantic_units=source_semantic_units,
                source_semantic_units_mask=source_semantic_units_mask,
                source_ref_codes=source_ref_codes,
                source_ref_mask=source_ref_mask,
                input_ids=input_ids,
                source_prompt_positions=prompt_source_positions,
            )
        content_cross_attn_memory = None
        content_cross_attn_mask = None
        if self.content_cross_attn_encoder is not None:
            content_cross_attn_memory, content_cross_attn_mask = self._compute_content_cross_attn_memory(
                source_semantic_features,
                source_semantic_features_mask,
            )
        if text_temperature > 0:
            text_do_sample = True
        else:
            text_temperature = 1
            text_do_sample = False
        if audio_temperature > 0:
            audio_do_sample = True
        else:
            audio_temperature = 1
            audio_do_sample = False

        past_key_values = None
        cfg_past_key_values = None
        device = input_ids.device
        current_input_ids = input_ids
        current_role_ids = prompt_role_ids
        current_attention_mask = attention_mask
        if current_attention_mask is None:
            current_attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=device)
        batch_size, seq_len, n_vq_with_text = input_ids.shape
        n_vq = n_vq_with_text - 1
        gen_slot_id = int(config.audio_assistant_gen_slot_token_id)

        generation_ids = input_ids[:]
        is_stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
        audio_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device)
        audio_gen_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device)
        semantic_target_steps = torch.zeros(batch_size, dtype=torch.int64, device=device)
        torch_int64_max = torch.iinfo(torch.int64).max
        delayed_lengths = torch.full((batch_size,), torch_int64_max, dtype=torch.int64, device=device)
        ref_slot_attention_rows: list[dict[str, Any]] = []
        ref_slot_attention_limit = max(0, int(ref_speaker_prompt_attention_capture_frames or 0))
        ref_slot_attention_layer_spec = ref_speaker_prompt_attention_layers

        is_continuation = (input_ids[:, -1, 0] == config.audio_start_token_id) | (
            input_ids[:, -1, 0] == config.audio_assistant_gen_slot_token_id
        )
        audio_start_indices = find_last_equal_C(input_ids[..., 0], config.audio_start_token_id)
        audio_start_mask = is_continuation & (audio_start_indices != -1)
        audio_lengths[audio_start_mask] = seq_len - audio_start_indices[audio_start_mask]
        if bool(audio_start_mask.any().item()):
            for batch_idx in torch.nonzero(audio_start_mask, as_tuple=False).reshape(-1).tolist():
                start_idx = int(audio_start_indices[batch_idx].item())
                if start_idx >= 0:
                    audio_gen_lengths[batch_idx] = int((input_ids[batch_idx, start_idx:, 0] == gen_slot_id).sum().item())
                    semantic_target_steps[batch_idx] = audio_gen_lengths[batch_idx]
        is_audio = audio_start_mask.clone()

        pre_exclude_mask0 = torch.tensor(
            [
                config.pad_token_id,
                config.audio_assistant_gen_slot_token_id,
                config.audio_assistant_delay_slot_token_id,
                config.audio_end_token_id,
            ],
            device=device,
        )
        pre_exclude_mask1 = torch.ones(config.language_config.vocab_size, device=device).bool()
        pre_exclude_mask1[[config.audio_assistant_gen_slot_token_id, config.audio_assistant_delay_slot_token_id]] = False
        semantic_progress_budget = int(min_audio_tokens) if int(min_audio_tokens) > 1 else int(max_new_tokens)
        semantic_progress_denom = max(1, semantic_progress_budget - 1)
        progress_stop_infer_enabled = self.progress_stop_head is not None and (
            float(self.timbre_memory_config.progress_loss_weight) > 0.0
            or float(self.timbre_memory_config.stop_loss_weight) > 0.0
        )
        progress_stop_forced = torch.zeros(batch_size, dtype=torch.int64, device=device)
        progress_stop_prob_max = torch.zeros(batch_size, dtype=torch.float32, device=device)
        progress_stop_value_max = torch.zeros(batch_size, dtype=torch.float32, device=device)
        progress_stop_steps = 0

        for time_step in tqdm(range(max_new_tokens), desc=f"Generating bs{batch_size} ..."):
            target_mask = torch.zeros(current_input_ids.shape[:2], dtype=torch.bool, device=device)
            if self.target_head_router is not None:
                target_mask[:, -1] = ~is_stopping
            elif time_step > 0:
                target_mask[:, -1] = ~is_stopping
            target_progress = None
            semantic_query_active = ~is_stopping
            if source_semantic_memory is not None:
                progress_value = semantic_target_steps.float().div(float(semantic_progress_denom)).clamp(0.0, 1.0)
                target_progress = torch.zeros(current_input_ids.shape[:2], dtype=torch.float32, device=device)
                target_progress[:, -1] = progress_value
            capture_ref_slot_attention = (
                ref_slot_attention_limit > 0
                and ref_speaker_prompt_slot_positions is not None
                and len(ref_slot_attention_rows) < ref_slot_attention_limit
                and bool(is_audio.any().item())
            )
            attention_kwargs = {"output_attentions": True} if capture_ref_slot_attention else {}
            if progress_stop_infer_enabled:
                attention_kwargs["output_hidden_states"] = True
            try:
                if side_pathway_enabled:
                    self._prepare_speaker_side_condition(
                        batch_size=batch_size,
                        device=device,
                        dtype=next(self.parameters()).dtype,
                        speaker_vec=speaker_vec,
                        speaker_vec_mask=speaker_vec_mask,
                        speaker_vec_path=speaker_vec_path,
                        speaker_seq_features=speaker_seq_features,
                        speaker_seq_features_mask=speaker_seq_features_mask,
                        timbre_ref_speaker_embedding_path=timbre_ref_speaker_embedding_path,
                        timbre_ref_speaker_audio_path=timbre_ref_speaker_audio_path,
                    )
                outputs = self._forward_with_timbre_tokens(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    timbre_tokens=timbre_tokens,
                    target_position_mask=target_mask,
                    target_position_progress=target_progress,
                    prosody_tokens=prosody_tokens,
                    prosody_batch_gate=prosody_batch_gate,
                    source_semantic_memory=source_semantic_memory,
                    source_semantic_mask=source_semantic_memory_mask,
                    content_cross_attn_memory=content_cross_attn_memory,
                    content_cross_attn_mask=content_cross_attn_mask,
                    content_token_ids=content_token_ids,
                    content_token_ids_mask=content_token_ids_mask,
                    vc_mode_id=vc_mode_id,
                    role_ids=current_role_ids,
                    timbre_ref_prompt_positions=None,
                    ref_speaker_prompt_slot_positions=ref_speaker_prompt_slot_positions,
                    **attention_kwargs,
                )
            except TypeError as exc:
                if not capture_ref_slot_attention or "output_attentions" not in str(exc):
                    raise
                self.last_ref_speaker_prompt_attention_stats = {
                    "requested_frames": int(ref_slot_attention_limit),
                    "captured_frames": 0,
                    "error": f"output_attentions unsupported: {exc}",
                }
                ref_slot_attention_limit = 0
                if side_pathway_enabled:
                    self._prepare_speaker_side_condition(
                        batch_size=batch_size,
                        device=device,
                        dtype=next(self.parameters()).dtype,
                        speaker_vec=speaker_vec,
                        speaker_vec_mask=speaker_vec_mask,
                        speaker_vec_path=speaker_vec_path,
                        speaker_seq_features=speaker_seq_features,
                        speaker_seq_features_mask=speaker_seq_features_mask,
                        timbre_ref_speaker_embedding_path=timbre_ref_speaker_embedding_path,
                        timbre_ref_speaker_audio_path=timbre_ref_speaker_audio_path,
                    )
                outputs = self._forward_with_timbre_tokens(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=progress_stop_infer_enabled,
                    timbre_tokens=timbre_tokens,
                    target_position_mask=target_mask,
                    target_position_progress=target_progress,
                    prosody_tokens=prosody_tokens,
                    prosody_batch_gate=prosody_batch_gate,
                    source_semantic_memory=source_semantic_memory,
                    source_semantic_mask=source_semantic_memory_mask,
                    content_cross_attn_memory=content_cross_attn_memory,
                    content_cross_attn_mask=content_cross_attn_mask,
                    content_token_ids=content_token_ids,
                    content_token_ids_mask=content_token_ids_mask,
                    vc_mode_id=vc_mode_id,
                    role_ids=current_role_ids,
                    timbre_ref_prompt_positions=None,
                    ref_speaker_prompt_slot_positions=ref_speaker_prompt_slot_positions,
                )
            if capture_ref_slot_attention:
                attention_row = self._capture_ref_speaker_prompt_attention(
                    getattr(outputs, "attentions", None),
                    slot_positions=ref_speaker_prompt_slot_positions,
                    frame_index=len(ref_slot_attention_rows),
                    decode_step=time_step,
                    layer_spec=ref_slot_attention_layer_spec,
                )
                if attention_row is not None:
                    ref_slot_attention_rows.append(attention_row)
            cfg_outputs = None
            if cfg_timbre_tokens is not None or (side_pathway_enabled and abs(cfg_scale - 1.0) > 1.0e-6):
                if side_pathway_enabled:
                    self._prepare_speaker_side_condition(
                        batch_size=batch_size,
                        device=device,
                        dtype=next(self.parameters()).dtype,
                        speaker_vec=speaker_vec,
                        speaker_vec_mask=speaker_vec_mask,
                        speaker_vec_path=speaker_vec_path,
                        speaker_seq_features=speaker_seq_features,
                        speaker_seq_features_mask=speaker_seq_features_mask,
                        timbre_ref_speaker_embedding_path=timbre_ref_speaker_embedding_path,
                        timbre_ref_speaker_audio_path=timbre_ref_speaker_audio_path,
                        force_drop_speaker_condition=True,
                    )
                cfg_outputs = self._forward_with_timbre_tokens(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=cfg_past_key_values,
                    use_cache=True,
                    timbre_tokens=cfg_timbre_tokens,
                    target_position_mask=target_mask,
                    target_position_progress=target_progress,
                    prosody_tokens=prosody_tokens,
                    prosody_batch_gate=prosody_batch_gate,
                    source_semantic_memory=source_semantic_memory,
                    source_semantic_mask=source_semantic_memory_mask,
                    content_cross_attn_memory=content_cross_attn_memory,
                    content_cross_attn_mask=content_cross_attn_mask,
                    content_token_ids=content_token_ids,
                    content_token_ids_mask=content_token_ids_mask,
                    vc_mode_id=vc_mode_id,
                    role_ids=current_role_ids,
                    timbre_ref_prompt_positions=None,
                    ref_speaker_prompt_slot_positions=ref_speaker_prompt_slot_positions,
                )
                cfg_past_key_values = cfg_outputs.past_key_values
            past_key_values = outputs.past_key_values

            raw_logits = outputs.logits
            if cfg_outputs is not None:
                raw_logits = [
                    uncond_logit + cfg_scale * (cond_logit - uncond_logit)
                    for cond_logit, uncond_logit in zip(outputs.logits, cfg_outputs.logits)
                ]
            next_token_logits = [
                logit[:, -1, :] / text_temperature if logit_idx == 0 else logit[:, -1, :] / audio_temperature
                for logit_idx, logit in enumerate(raw_logits)
            ]
            next_token_logits[0] = next_token_logits[0].clone()
            next_text_token = torch.full((batch_size,), config.pad_token_id, device=device)
            next_text_token[~is_stopping & (delayed_lengths < n_vq)] = config.audio_assistant_delay_slot_token_id
            is_audio_eos = ~is_stopping & (delayed_lengths == n_vq)
            next_text_token[is_audio_eos] = config.audio_end_token_id
            is_audio[is_audio_eos] = False
            sampling_text_mask = ~is_stopping & (delayed_lengths > n_vq)
            next_token_logits[0][~is_audio] = next_token_logits[0][~is_audio].index_fill(
                -1, pre_exclude_mask0, float("-inf")
            )
            next_token_logits[0][is_audio] = next_token_logits[0][is_audio].masked_fill(
                pre_exclude_mask1, float("-inf")
            )
            if time_step == 0:
                next_token_logits[0][..., 151662] = float("-inf")
            if time_step <= n_vq:
                next_token_logits[0][..., config.im_end_token_id] = float("-inf")
            if min_new_tokens > 0 and time_step < min_new_tokens:
                next_token_logits[0][..., config.im_end_token_id] = float("-inf")
            if min_audio_tokens > 0:
                prevent_early_delay = sampling_text_mask & is_audio & (audio_gen_lengths < int(min_audio_tokens))
                if bool(prevent_early_delay.any().item()):
                    next_token_logits[0][
                        prevent_early_delay,
                        int(config.audio_assistant_delay_slot_token_id),
                    ] = float("-inf")
            progress_stop_trigger = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if progress_stop_infer_enabled and bool((sampling_text_mask & is_audio).any().item()):
                progress_stop_scores = self._progress_stop_inference_scores(outputs)
                if progress_stop_scores is not None:
                    stop_prob = progress_stop_scores["stop_prob"].to(device=device)
                    progress_value = progress_stop_scores["progress_value"].to(device=device)
                    progress_stop_prob_max = torch.maximum(progress_stop_prob_max, stop_prob)
                    progress_stop_value_max = torch.maximum(progress_stop_value_max, progress_value)
                    progress_stop_steps += 1
                    progress_stop_trigger = (
                        sampling_text_mask
                        & is_audio
                        & ((stop_prob > 0.5) | (progress_value >= 0.98))
                    )
                    if bool(progress_stop_trigger.any().item()):
                        progress_stop_forced[progress_stop_trigger] += 1

            next_text_token[sampling_text_mask] = sample_token(
                logits=next_token_logits[0][sampling_text_mask],
                top_p=text_top_p,
                top_k=text_top_k,
                do_sample=text_do_sample,
            )
            if bool(progress_stop_trigger.any().item()):
                next_text_token[progress_stop_trigger] = config.audio_assistant_delay_slot_token_id
            is_audio[next_text_token == config.audio_start_token_id] = True
            is_stopping[next_text_token == config.im_end_token_id] = True

            next_audio_tokens = torch.full((batch_size, n_vq), config.audio_pad_code, device=device)
            pre_audio_mask = audio_lengths.unsqueeze(1) > torch.arange(n_vq, dtype=int, device=device).expand(
                batch_size, n_vq
            )
            post_audio_mask = torch.arange(n_vq, dtype=int, device=device).expand(
                batch_size, n_vq
            ) > delayed_lengths.unsqueeze(1) - 1
            post_audio_mask[delayed_lengths == torch_int64_max] = True
            sampling_audio_mask = pre_audio_mask & post_audio_mask
            next_audio_tokens[~sampling_audio_mask] = config.audio_pad_code

            if sampling_audio_mask.sum() > 0:
                audio_ch0_logits = next_token_logits[1][sampling_audio_mask[:, 0]]
                audio_logits = torch.stack(next_token_logits[2:], dim=1)[sampling_audio_mask[:, 1:]]
                audio_ch0_logits[..., config.audio_pad_code] = float("-inf")
                audio_logits[..., config.audio_pad_code] = float("-inf")
                next_audio_tokens[:, 0][sampling_audio_mask[:, 0]] = sample_token(
                    logits=audio_ch0_logits,
                    prev_tokens=generation_ids[:, :, 1],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p,
                    top_k=audio_top_k,
                    do_sample=audio_do_sample,
                )
                next_audio_tokens[:, 1:][sampling_audio_mask[:, 1:]] = sample_token(
                    logits=audio_logits,
                    prev_tokens=generation_ids[:, :, 2:],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p,
                    top_k=audio_top_k,
                    do_sample=audio_do_sample,
                )

            audio_lengths[
                (next_text_token == config.audio_start_token_id)
                | (next_text_token == config.audio_assistant_gen_slot_token_id)
                | (next_text_token == config.audio_assistant_delay_slot_token_id)
            ] += 1
            audio_gen_lengths[next_text_token == config.audio_start_token_id] = 0
            audio_gen_lengths[next_text_token == config.audio_assistant_gen_slot_token_id] += 1
            audio_gen_lengths[next_text_token == config.audio_end_token_id] = 0
            audio_lengths[next_text_token == config.audio_end_token_id] = 0
            delayed_lengths[
                (delayed_lengths == torch_int64_max) & (next_text_token == config.audio_assistant_delay_slot_token_id)
            ] = 0
            delayed_lengths[delayed_lengths != torch_int64_max] += 1
            delayed_lengths[delayed_lengths > n_vq] = torch_int64_max
            if source_semantic_progress_clock == "gen_slot":
                semantic_target_steps[next_text_token == gen_slot_id] += 1
            else:
                semantic_target_steps[semantic_query_active] += 1

            current_input_ids = torch.cat([next_text_token[:, None, None], next_audio_tokens[:, None, :]], dim=2)
            if self.role_router is not None:
                current_role_ids = torch.full(
                    current_input_ids.shape[:2],
                    TARGET_CODEC,
                    dtype=torch.long,
                    device=device,
                )
            current_attention_mask = torch.cat([current_attention_mask, (~is_stopping).unsqueeze(-1)], dim=-1)
            generation_ids = torch.cat([generation_ids, current_input_ids], dim=1)
            if is_stopping.sum() == batch_size:
                break

        if progress_stop_infer_enabled:
            forced_rows = progress_stop_forced.detach().cpu()
            self.last_progress_stop_infer_stats = {
                "enabled": 1.0,
                "steps": float(progress_stop_steps),
                "forced_rows": float((forced_rows > 0).sum().item()),
                "forced_total": float(forced_rows.sum().item()),
                "stop_prob_max_mean": float(progress_stop_prob_max.detach().float().mean().item()),
                "stop_prob_max_max": float(progress_stop_prob_max.detach().float().max().item()),
                "progress_value_max_mean": float(progress_stop_value_max.detach().float().mean().item()),
                "progress_value_max_max": float(progress_stop_value_max.detach().float().max().item()),
            }
        start_indices = find_last_equal_C(input_ids[..., 0], config.im_start_token_id) + 3
        start_lengths = seq_len - start_indices
        output = []
        for start_idx, start_length, cur_generation_ids in zip(start_indices, start_lengths, generation_ids):
            output.append((start_length, cur_generation_ids[start_idx:]))
        if ref_slot_attention_limit > 0:
            layer_values: dict[str, dict[str, list[float]]] = {}
            for row in ref_slot_attention_rows:
                for layer, payload in (row.get("layers") or {}).items():
                    bucket = layer_values.setdefault(
                        layer,
                        {
                            "mean": [],
                            "max_head": [],
                            "uniform": [],
                            "over_uniform": [],
                        },
                    )
                    bucket["mean"].append(float(payload.get("slot_attention_mean", 0.0)))
                    bucket["max_head"].append(float(payload.get("slot_attention_max_head", 0.0)))
                    bucket["uniform"].append(float(payload.get("uniform_baseline", 0.0)))
                    bucket["over_uniform"].append(float(payload.get("slot_attention_max_head_over_uniform", 0.0)))
            by_layer = {
                layer: {
                    "frames": len(values["mean"]),
                    "slot_attention_mean": float(sum(values["mean"]) / max(1, len(values["mean"]))),
                    "slot_attention_max": float(max(values["mean"])) if values["mean"] else 0.0,
                    "slot_attention_max_head_mean": float(sum(values["max_head"]) / max(1, len(values["max_head"]))),
                    "slot_attention_max_head_max": float(max(values["max_head"])) if values["max_head"] else 0.0,
                    "uniform_baseline_mean": float(sum(values["uniform"]) / max(1, len(values["uniform"]))),
                    "max_head_over_uniform_mean": float(
                        sum(values["over_uniform"]) / max(1, len(values["over_uniform"]))
                    ),
                    "max_head_over_uniform_max": float(max(values["over_uniform"])) if values["over_uniform"] else 0.0,
                }
                for layer, values in sorted(layer_values.items(), key=lambda item: int(item[0]))
            }
            frame_values = [float(row.get("slot_attention_mean", 0.0)) for row in ref_slot_attention_rows]
            frame_max_head_values = [float(row.get("slot_attention_max_head", 0.0)) for row in ref_slot_attention_rows]
            frame_over_uniform_values = [
                float(row.get("slot_attention_max_head_over_uniform", 0.0)) for row in ref_slot_attention_rows
            ]
            self.last_ref_speaker_prompt_attention_stats = {
                "requested_frames": int(ref_slot_attention_limit),
                "captured_frames": int(len(ref_slot_attention_rows)),
                "slot_attention_mean": float(sum(frame_values) / max(1, len(frame_values))) if frame_values else 0.0,
                "slot_attention_max_frame": float(max(frame_values)) if frame_values else 0.0,
                "slot_attention_max_head_mean": float(
                    sum(frame_max_head_values) / max(1, len(frame_max_head_values))
                )
                if frame_max_head_values
                else 0.0,
                "slot_attention_max_head_max": float(max(frame_max_head_values)) if frame_max_head_values else 0.0,
                "max_head_over_uniform_mean": float(
                    sum(frame_over_uniform_values) / max(1, len(frame_over_uniform_values))
                )
                if frame_over_uniform_values
                else 0.0,
                "max_head_over_uniform_max": float(max(frame_over_uniform_values)) if frame_over_uniform_values else 0.0,
                "layers": by_layer,
                "frames": ref_slot_attention_rows,
            }
        return output

    def _save_peft_adapter_without_state_dict_gather(self, save_path: Path) -> bool:
        """Save PEFT LoRA files from replicated LoRA params without calling model.state_dict()."""
        if not hasattr(self.model, "peft_config"):
            return False
        try:
            from peft.utils.save_and_load import get_peft_model_state_dict
            from safetensors.torch import save_file
        except ImportError:
            return False

        adapter_names = list(getattr(self.model, "peft_config", {}).keys())
        if not adapter_names:
            return False
        if len(adapter_names) != 1 or adapter_names[0] != "default":
            raise RuntimeError(f"Manual PEFT save currently expects a single default adapter; got {adapter_names}")

        adapter_name = "default"
        manual_state_dict: dict[str, torch.Tensor] = {}
        empty_lora_tensors: list[tuple[str, tuple[int, ...]]] = []
        for name, value in self.model.named_parameters():
            if "lora_" not in name or adapter_name not in name:
                continue
            tensor = value.detach()
            if tensor.numel() == 0:
                empty_lora_tensors.append((name, tuple(tensor.shape)))
                continue
            manual_state_dict[name] = tensor.cpu().contiguous()
        if empty_lora_tensors:
            sample = ", ".join(f"{name}{shape}" for name, shape in empty_lora_tensors[:8])
            raise RuntimeError(
                "Cannot save LoRA adapter from empty FSDP shards. "
                "LoRA A/B modules must be excluded from FSDP wrapping before training. "
                f"empty_count={len(empty_lora_tensors)} sample={sample}"
            )
        if not manual_state_dict:
            raise RuntimeError("Cannot save PEFT adapter: no LoRA tensors found in model.named_parameters().")

        peft_state_dict = get_peft_model_state_dict(
            self.model,
            state_dict=manual_state_dict,
            adapter_name=adapter_name,
            save_embedding_layers=False,
        )
        if not peft_state_dict:
            raise RuntimeError("Cannot save PEFT adapter: get_peft_model_state_dict returned an empty state dict.")
        normalized_peft_state_dict: dict[str, torch.Tensor] = {}
        for key, value in peft_state_dict.items():
            normalized_key = key.replace("._fsdp_wrapped_module.", ".")
            normalized_peft_state_dict[normalized_key] = value

        save_path.mkdir(parents=True, exist_ok=True)
        save_file(normalized_peft_state_dict, str(save_path / "adapter_model.safetensors"))

        peft_config = self.model.peft_config[adapter_name]
        old_inference_mode = getattr(peft_config, "inference_mode", None)
        peft_config.inference_mode = True
        peft_config.save_pretrained(str(save_path))
        if old_inference_mode is not None:
            peft_config.inference_mode = old_inference_mode

        with (save_path / "README.md").open("w", encoding="utf-8") as f:
            f.write(
                "# MOSS-CodecVC LoRA Adapter\n\n"
                "This checkpoint was saved with a rank-safe trainable-only saver. "
                "It contains PEFT LoRA weights plus MOSS-CodecVC Ver2 timbre/routing adapter weights.\n"
            )
        return True

    def _has_trainable_lora_parameters(self) -> bool:
        for name, param in self.model.named_parameters():
            if "lora_" in name and bool(param.requires_grad):
                return True
        return False

    def _copy_frozen_peft_adapter_from_fallback(self, save_path: Path) -> bool:
        fallback_dir = self.peft_adapter_fallback_directory
        if fallback_dir is None:
            return False
        fallback_path = Path(fallback_dir)
        adapter_file = fallback_path / "adapter_model.safetensors"
        adapter_config = fallback_path / "adapter_config.json"
        if not adapter_file.exists() or not adapter_config.exists():
            return False
        save_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter_file, save_path / adapter_file.name)
        shutil.copy2(adapter_config, save_path / adapter_config.name)
        readme_file = fallback_path / "README.md"
        if readme_file.exists():
            shutil.copy2(readme_file, save_path / readme_file.name)
        return True

    def save_pretrained(self, save_directory: str | Path, *args, **kwargs) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        full_state_dict = kwargs.pop("state_dict", None)
        saved_peft_adapter = False
        try:
            saved_peft_adapter = self._save_peft_adapter_without_state_dict_gather(save_path)
        except RuntimeError:
            if self._has_trainable_lora_parameters() or not self._copy_frozen_peft_adapter_from_fallback(save_path):
                raise
            saved_peft_adapter = True
        if (
            not saved_peft_adapter
            and not self._has_trainable_lora_parameters()
            and self._copy_frozen_peft_adapter_from_fallback(save_path)
        ):
            saved_peft_adapter = True
        if not saved_peft_adapter and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_path, *args, **kwargs)

        def extract_module_state(module_name: str, module: nn.Module | None) -> dict[str, torch.Tensor] | None:
            if module is None:
                return None
            if full_state_dict is not None:
                prefix = f"{module_name}."
                extracted: dict[str, torch.Tensor] = {}
                for key, value in full_state_dict.items():
                    if not torch.is_tensor(value):
                        continue
                    normalized_key = str(key)
                    changed = True
                    while changed:
                        changed = False
                        for wrapper_prefix in ("module.", "_fsdp_wrapped_module."):
                            if normalized_key.startswith(wrapper_prefix):
                                normalized_key = normalized_key[len(wrapper_prefix) :]
                                changed = True
                    if normalized_key.startswith(prefix):
                        extracted[normalized_key[len(prefix) :]] = value.detach().cpu()
                if extracted:
                    return extracted
            return {key: value.detach().cpu() for key, value in module.state_dict().items()}

        torch.save(
            {
                "timbre_memory": extract_module_state("timbre_memory", self.timbre_memory),
                "layer_adapters": extract_module_state("layer_adapters", self.layer_adapters),
                "adapter_layer_indices": self.adapter_layer_indices,
                "speaker_projection": extract_module_state("speaker_projection", self.speaker_projection),
                "prosody_head": extract_module_state("prosody_head", self.prosody_head),
                "content_head": extract_module_state("content_head", self.content_head),
                "content_ctc_head": extract_module_state("content_ctc_head", self.content_ctc_head),
                "content_token_head": extract_module_state("content_token_head", self.content_token_head),
                "content_codec_head": extract_module_state("content_codec_head", self.content_codec_head),
                "semantic_token_head": extract_module_state("semantic_token_head", self.semantic_token_head),
                "semantic_feature_head": extract_module_state("semantic_feature_head", self.semantic_feature_head),
                "progress_stop_head": extract_module_state("progress_stop_head", self.progress_stop_head),
                "source_semantic_memory_encoder": extract_module_state(
                    "source_semantic_memory_encoder",
                    self.source_semantic_memory_encoder,
                ),
                "source_semantic_codec_residual_encoder": extract_module_state(
                    "source_semantic_codec_residual_encoder",
                    self.source_semantic_codec_residual_encoder,
                ),
                "source_semantic_layer_adapters": extract_module_state(
                    "source_semantic_layer_adapters",
                    self.source_semantic_layer_adapters,
                ),
                "content_cross_attn_encoder": extract_module_state(
                    "content_cross_attn_encoder",
                    self.content_cross_attn_encoder,
                ),
                "content_cross_attn_layers": extract_module_state(
                    "content_cross_attn_layers",
                    self.content_cross_attn_layers,
                ),
                "content_phoneme_classifier": extract_module_state(
                    "content_phoneme_classifier",
                    self.content_phoneme_classifier,
                ),
                "role_router": extract_module_state("role_router", self.role_router),
                "source_prosody_encoder": extract_module_state("source_prosody_encoder", self.source_prosody_encoder),
                "target_head_router": extract_module_state("target_head_router", self.target_head_router),
                "ref_speaker_prompt": extract_module_state("ref_speaker_prompt", self.ref_speaker_prompt),
                "ref_speaker_adaln": extract_module_state("ref_speaker_adaln", self.ref_speaker_adaln),
                "speaker_side_adaln": extract_module_state("speaker_side_adaln", self.speaker_side_adaln),
                "speaker_side_kv_bias": extract_module_state("speaker_side_kv_bias", self.speaker_side_kv_bias),
                "speaker_side_gate_logits": extract_module_state(
                    "speaker_side_gate_logits",
                    self.speaker_side_gate_logits,
                ),
                "speaker_cross_attn_tokens": extract_module_state(
                    "speaker_cross_attn_tokens",
                    self.speaker_cross_attn_tokens,
                ),
                "speaker_cross_attn_seq_projector": extract_module_state(
                    "speaker_cross_attn_seq_projector",
                    self.speaker_cross_attn_seq_projector,
                ),
                "speaker_cross_attn_layers": extract_module_state(
                    "speaker_cross_attn_layers",
                    self.speaker_cross_attn_layers,
                ),
                "null_speaker_embedding": None
                if self.null_speaker_embedding is None
                else self.null_speaker_embedding.detach().cpu(),
                "speaker_infonce_negative_pool": self.speaker_infonce_negative_pool.detach().float().cpu(),
            },
            save_path / "timbre_memory_adapter.pt",
        )
        with (save_path / "timbre_memory_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(self.timbre_memory_config), f, indent=2, ensure_ascii=False)
