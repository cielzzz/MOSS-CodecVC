#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import unicodedata
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config
from moss_codecvc.moss_codec import ensure_moss_on_path
from moss_codecvc.models import (
    MossCodecVCTimbreMemoryWrapper,
    MossCodecVCTimbreSFTDataset,
    TimbreMemoryConfig,
)


def import_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


infer_mod = import_file("moss_codecvc_infer_helpers", ROOT / "scripts/003001_infer_moss_codecvc.py")
train_mod = import_file("moss_codecvc_train_helpers", ROOT / "scripts/002002_train_moss_codecvc_lora.py")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Teacher-forced greedy decode probe for the Ver2.3 content CTC head.")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--model-path", required=True, help="LoRA adapter dir or full model dir.")
    ap.add_argument("--base-model-path", default=None)
    ap.add_argument("--jsonl", required=True, help="Train-ready JSONL to probe, e.g. no-text tiny split.")
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument(
        "--summary-json",
        default=None,
        help="Aggregate metrics JSON. Defaults to <output-jsonl without .jsonl>.summary.json.",
    )
    ap.add_argument("--max-rows", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-vq", type=int, default=None)
    ap.add_argument("--source-gate-floor", type=float, default=None)
    ap.add_argument("--speaker-encoder-type", default="speechbrain_ecapa")
    ap.add_argument(
        "--speaker-encoder-path",
        default="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb",
    )
    ap.add_argument("--speaker-embedding-dim", type=int, default=192)
    ap.add_argument("--allow-mixed-content-tokenizers", action="store_true")
    return ap.parse_args()


def collapse_ctc(ids: list[int], blank_id: int) -> list[int]:
    out: list[int] = []
    prev: int | None = None
    for token in ids:
        token = int(token)
        if token != blank_id and token != prev:
            out.append(token)
        prev = token
    return out


def strip_target(ids: torch.Tensor, mask: torch.Tensor | None, blank_id: int) -> list[int]:
    values = ids.detach().cpu().long().tolist()
    if mask is None:
        return [int(v) for v in values if int(v) >= 0 and int(v) != blank_id]
    keep = mask.detach().cpu().bool().tolist()
    return [int(v) for v, ok in zip(values, keep) if ok and int(v) >= 0 and int(v) != blank_id]


def token_edit_distance(pred: list[int], target: list[int]) -> int:
    prev = list(range(len(target) + 1))
    for i, pred_token in enumerate(pred, start=1):
        cur = [i] + [0] * len(target)
        for j, target_token in enumerate(target, start=1):
            cost = 0 if pred_token == target_token else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return int(prev[-1])


def token_error_rate(pred: list[int], target: list[int]) -> float | None:
    if not target:
        return None
    return float(token_edit_distance(pred, target)) / float(len(target))


def decode_tokens(tokenizer: Any, ids: list[int], offset: int) -> str | None:
    raw_ids = [int(token) - int(offset) for token in ids if int(token) - int(offset) >= 0]
    if tokenizer is None:
        return None
    if not raw_ids:
        return ""
    try:
        return str(tokenizer.decode(raw_ids, skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(raw_ids))
    except Exception as exc:
        return f"<decode_error:{type(exc).__name__}:{exc}>"


def decoded_text_diagnostics(text: str | None, *, collapsed_len: int, target_len: int) -> dict[str, Any]:
    value = "" if text is None else str(text).strip()
    alnum_count = sum(unicodedata.category(char)[:1] in {"L", "N"} for char in value)
    punctuation_count = sum(unicodedata.category(char).startswith("P") for char in value)
    empty_prediction = int(collapsed_len == 0 or not value)
    punctuation_only = int(bool(value) and alnum_count == 0 and punctuation_count > 0)
    single_token_collapse = int(target_len >= 4 and collapsed_len <= 1)
    severe_length_collapse = int(target_len >= 5 and collapsed_len <= max(1, int(0.2 * target_len)))
    labels = []
    if empty_prediction:
        labels.append("empty_or_blank")
    if punctuation_only:
        labels.append("punctuation_only")
    if single_token_collapse:
        labels.append("single_token")
    if severe_length_collapse:
        labels.append("severe_short")
    return {
        "decoded_text_nonspace_len": len("".join(value.split())),
        "decoded_text_alnum_count": int(alnum_count),
        "decoded_text_punctuation_count": int(punctuation_count),
        "empty_prediction": bool(empty_prediction),
        "punctuation_only_collapse": bool(punctuation_only),
        "single_token_collapse": bool(single_token_collapse),
        "severe_length_collapse": bool(severe_length_collapse),
        "collapse_diagnosis": labels or ["none"],
    }


class SentencePieceDecoder:
    def __init__(self, model_path: str) -> None:
        import sentencepiece as spm

        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return str(self.processor.decode([int(x) for x in ids]))


def resolve_display_tokenizer(records: list[dict[str, Any]], fallback_tokenizer: Any) -> Any:
    for record in records:
        if str(record.get("content_tokenizer") or "").lower() == "sentencepiece":
            vocab_path = record.get("content_vocab_path")
            if vocab_path:
                try:
                    return SentencePieceDecoder(str(vocab_path))
                except Exception as exc:
                    print(f"[ctc-probe] failed to load sentencepiece decoder {vocab_path}: {exc}", file=sys.stderr)
                    return fallback_tokenizer
    return fallback_tokenizer


class DefaultNoneNamespace(argparse.Namespace):
    """Keep this legacy probe compatible as the training config surface grows."""

    def __getattr__(self, name: str) -> None:
        del name
        return None


def make_timbre_config_args(args: argparse.Namespace) -> argparse.Namespace:
    return DefaultNoneNamespace(
        version="ver2",
        use_timbre_memory=None,
        timbre_memory_tokens=None,
        timbre_adapter_layers=None,
        timbre_adapter_heads=None,
        timbre_adapter_dim=None,
        timbre_adapter_dropout=None,
        timbre_adapter_init_gate=None,
        timbre_encoder_type=None,
        timbre_encoder_layers=None,
        timbre_conformer_kernel_size=None,
        timbre_speaker_conditioning=None,
        target_speaker_similarity_weight=0.0,
        source_speaker_suppression_weight=0.0,
        speaker_encoder_type=args.speaker_encoder_type,
        speaker_encoder_path=args.speaker_encoder_path,
        speaker_embedding_dim=args.speaker_embedding_dim,
        speaker_loss_margin=0.0,
        enable_role_routing=None,
        enable_target_head_routing=None,
        lambda_route=0.0,
        routing_gate_lr_multiplier=1.0,
        lambda_prosody=0.0,
        lambda_content=0.0,
        prosody_f0_weight=0.0,
        prosody_voiced_weight=0.0,
        prosody_energy_weight=0.0,
        prosody_pause_weight=0.0,
        prosody_duration_weight=0.0,
        prosody_normalize_f0=None,
        prosody_normalize_energy=None,
        content_positive=None,
        content_embedding_dim=0,
        content_embedding_weight=0.0,
        content_ctc_weight=1.0,
        content_ctc_vocab_size=None,
        content_ctc_blank_id=None,
        content_ctc_token_offset=None,
        content_token_vocab_size=0,
        content_token_weight=0.0,
        content_source_codec_weight=0.0,
        content_source_codec_codebooks=None,
        semantic_loss_weight=0.0,
        semantic_mode=None,
        semantic_source=None,
        semantic_vocab_size=0,
        semantic_feature_dim=0,
        semantic_feature_loss_type=None,
        progress_loss_weight=0.0,
        stop_loss_weight=0.0,
        progress_num_bins=None,
        prosody_memory_tokens=None,
        source_prosody_encoder_type=None,
        source_prosody_encoder_layers=None,
        source_prosody_conv_kernel_size=None,
    )


def saved_timbre_config(model_path: Path) -> tuple[dict[str, Any] | None, TimbreMemoryConfig | None]:
    config_path = model_path / "timbre_memory_config.json"
    if not config_path.exists():
        return None, None
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {field.name: field for field in fields(TimbreMemoryConfig)}
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in allowed or value is None:
            continue
        normalized[key] = value
    config = TimbreMemoryConfig(**normalized)
    config.enabled = True
    return raw, config


def null_default_overrides(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize historical JSON nulls when a newer dataclass added a concrete default."""

    if raw_config is None:
        return {}
    overrides: dict[str, Any] = {}
    for field in fields(TimbreMemoryConfig):
        if raw_config.get(field.name, MISSING) is not None:
            continue
        if field.default is not MISSING and field.default is not None:
            overrides[field.name] = field.default
    return overrides


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rows = int(summary["rows"])
    ter_count = int(summary["ter_count"])
    total_frames = int(summary["total_frames"])
    target_tokens = int(summary["target_tokens"])
    collapsed_tokens = int(summary["collapsed_tokens"])
    summary.update(
        {
            "exact_rate": float(summary["exact_matches"]) / max(1, rows),
            "mean_row_greedy_ter": float(summary["ter_sum"]) / max(1, ter_count),
            "greedy_ter": float(summary["edit_distance"]) / max(1, target_tokens),
            "blank_frame_rate": float(summary["blank_frames"]) / max(1, total_frames),
            "nonblank_frame_rate": float(summary["nonblank_frames"]) / max(1, total_frames),
            "blank_posterior_mean": float(summary["blank_posterior_sum"]) / max(1, total_frames),
            "nonblank_posterior_mean": float(summary["nonblank_posterior_sum"]) / max(1, total_frames),
            "collapsed_len_mean": float(collapsed_tokens) / max(1, rows),
            "collapsed_to_target_token_ratio": float(collapsed_tokens) / max(1, target_tokens),
            "collapsed_punctuation_share": float(summary["collapsed_punctuation_tokens"])
            / max(1, collapsed_tokens),
            "empty_prediction_rate": float(summary["empty_predictions"]) / max(1, rows),
            "punctuation_only_collapse_rate": float(summary["punctuation_only_collapses"]) / max(1, rows),
            "single_token_collapse_rate": float(summary["single_token_collapses"]) / max(1, rows),
            "severe_length_collapse_rate": float(summary["severe_length_collapses"]) / max(1, rows),
            "blank_dominant_rate": float(summary["blank_dominant_rows"]) / max(1, rows),
            "dominant_nonblank_frame_share_mean": float(summary["dominant_nonblank_frame_share_sum"])
            / max(1, rows),
        }
    )
    if summary["collapsed_len_min"] is None:
        summary["collapsed_len_min"] = 0
    return summary


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    model_path = Path(args.model_path).expanduser()
    codec_path = deep_get(cfg, "paths.codec") or deep_get(cfg, "moss.codec_path")
    moss_root = deep_get(cfg, "moss.root")
    if moss_root:
        ensure_moss_on_path(moss_root)

    from moss_tts_delay.finetuning.dataset import MossTTSSFTDataset
    from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
    from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor
    from peft import PeftConfig, PeftModel

    class LazyMossTTSSFTDataset(MossTTSSFTDataset):
        def __init__(self, records, processor, n_vq=None) -> None:
            self.records = records
            self.processor = processor
            self.n_vq = n_vq
            self._audio_cache = {}

    device_arg = infer_mod.normalize_device_arg(args.device)
    device = torch.device(device_arg if torch.cuda.is_available() or not device_arg.startswith("cuda") else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    is_lora_adapter = (model_path / "adapter_config.json").exists()
    peft_model_path = infer_mod.prepare_peft_adapter_dir_for_inference(model_path) if is_lora_adapter else model_path
    if is_lora_adapter:
        peft_cfg = PeftConfig.from_pretrained(str(peft_model_path))
        base_model_path = args.base_model_path or peft_cfg.base_model_name_or_path
        if not base_model_path:
            raise ValueError("--base-model-path is required when adapter_config.json does not specify a base model path")
        processor_path = base_model_path
        model_load_path = base_model_path
    else:
        processor_path = str(model_path)
        model_load_path = str(model_path)

    processor = MossTTSDelayProcessor.from_pretrained(processor_path, codec_path=codec_path, trust_remote_code=True)
    n_vq = int(args.n_vq or deep_get(cfg, "model.n_vq", getattr(processor, "n_vq", 32)))
    records = train_mod.load_records(str(Path(args.jsonl).expanduser()), int(args.max_rows))
    display_tokenizer = resolve_display_tokenizer(records, processor.tokenizer)
    raw_saved_config, timbre_config = saved_timbre_config(model_path)
    if timbre_config is None:
        timbre_config = train_mod.resolve_timbre_memory_config(cfg, make_timbre_config_args(args))
        timbre_config.content_ctc_weight = 1.0
    train_mod.validate_content_tokenizer_consistency(records, enabled=True, allow_mixed=bool(args.allow_mixed_content_tokenizers))
    manifest_vocab = train_mod.infer_content_ctc_vocab_size_from_manifest(records)
    uses_manifest_tokens = manifest_vocab > 1
    saved_vocab = int(timbre_config.content_ctc_vocab_size)
    if saved_vocab > 1 and manifest_vocab > 1 and saved_vocab != manifest_vocab:
        raise ValueError(
            "checkpoint/manifest content CTC vocabulary mismatch: "
            f"checkpoint={saved_vocab} manifest={manifest_vocab}"
        )
    if timbre_config.content_ctc_vocab_size <= 1:
        if manifest_vocab > 1:
            timbre_config.content_ctc_vocab_size = int(manifest_vocab)
        else:
            timbre_config.content_ctc_vocab_size = int(len(processor.tokenizer)) + int(timbre_config.content_ctc_token_offset)

    base_dataset = LazyMossTTSSFTDataset(records=records, processor=processor, n_vq=n_vq)
    dataset = MossCodecVCTimbreSFTDataset(
        records=records,
        base_dataset=base_dataset,
        n_vq=n_vq,
        audio_pad_code=int(processor.model_config.audio_pad_code),
        content_tokenizer=processor.tokenizer if not uses_manifest_tokens else None,
        content_ctc_token_offset=int(timbre_config.content_ctc_token_offset),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_fn)

    model = MossTTSDelayModel.from_pretrained(model_load_path, torch_dtype=dtype, trust_remote_code=True)
    if not hasattr(model, "prepare_inputs_for_generation"):
        model.prepare_inputs_for_generation = lambda *a, **kw: kw
    if not hasattr(model, "generation_config"):
        model.generation_config = GenerationConfig.from_model_config(model.config)
    if is_lora_adapter:
        model = PeftModel.from_pretrained(model, str(peft_model_path))
    has_saved_adapter = (model_path / "timbre_memory_config.json").exists() and (
        model_path / "timbre_memory_adapter.pt"
    ).exists()
    if has_saved_adapter:
        model = MossCodecVCTimbreMemoryWrapper.from_pretrained_timbre_memory(
            model,
            model_path,
            map_location="cpu",
            config_overrides=null_default_overrides(raw_saved_config),
        )
    else:
        model = MossCodecVCTimbreMemoryWrapper(model, timbre_config)
    if model.content_ctc_head is None:
        raise RuntimeError("content_ctc_head is not available; adapter may not contain the CTC head")
    model = model.to(device).eval()
    infer_mod.apply_source_gate_floor_for_inference(model, args.source_gate_floor)

    out_path = Path(args.output_jsonl).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_json).expanduser()
        if args.summary_json
        else out_path.with_suffix("").with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    blank_id = int(model.timbre_memory_config.content_ctc_blank_id)
    token_offset = int(model.timbre_memory_config.content_ctc_token_offset)
    content_cross_attn_mode = model.content_cross_attn_encoder is not None
    input_source = "source_bnf_content_cross_attn_memory" if content_cross_attn_mode else "legacy_target_hidden"
    summary: dict[str, Any] = {
        "schema_version": 2,
        "model_path": str(model_path),
        "input_jsonl": str(Path(args.jsonl).expanduser()),
        "output_jsonl": str(out_path),
        "content_ctc_input_source": input_source,
        "content_cross_attn_mode": bool(content_cross_attn_mode),
        "input_rows": 0,
        "rows": 0,
        "no_text_rows": 0,
        "text_bypassed_rows": 0,
        "exact_matches": 0,
        "ter_sum": 0.0,
        "ter_count": 0,
        "edit_distance": 0,
        "target_tokens": 0,
        "blank_frames": 0,
        "nonblank_frames": 0,
        "total_frames": 0,
        "blank_posterior_sum": 0.0,
        "nonblank_posterior_sum": 0.0,
        "collapsed_tokens": 0,
        "collapsed_punctuation_tokens": 0,
        "collapsed_len_min": None,
        "collapsed_len_max": 0,
        "empty_predictions": 0,
        "punctuation_only_collapses": 0,
        "single_token_collapses": 0,
        "severe_length_collapses": 0,
        "blank_dominant_rows": 0,
        "dominant_nonblank_frame_share_sum": 0.0,
    }
    print(
        f"[ctc-probe] input_source={input_source} model={model_path} "
        f"rows={len(records)} batch_size={args.batch_size}",
        flush=True,
    )
    with out_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        record_cursor = 0
        for batch in loader:
            batch_size = int(batch["input_ids"].shape[0])
            batch_records = [records[record_cursor + row_offset] for row_offset in range(batch_size)]
            record_cursor += batch_size
            summary["input_rows"] += batch_size
            batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
            labels = batch.get("labels")
            target_mask = (labels != -100).any(dim=-1) if labels is not None else None
            target_ids = batch.get("content_token_ids")
            target_ids_mask = batch.get("content_token_ids_mask")

            if content_cross_attn_mode:
                mode_ids = batch.get("vc_mode_id")
                if mode_ids is None or int(mode_ids.numel()) != batch_size:
                    raise RuntimeError("content-cross-attn CTC probe requires one vc_mode_id per batch row")
                no_text_id = int(model.MODE_TO_ID["no_text"])
                active_mask = mode_ids.long().view(-1).eq(no_text_id)
                summary["no_text_rows"] += int(active_mask.sum().item())
                summary["text_bypassed_rows"] += int((~active_mask).sum().item())
                if not bool(active_mask.any().item()):
                    continue
                active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
                source_features = batch.get("source_semantic_features")
                source_feature_mask = batch.get("source_semantic_features_mask")
                if source_features is None or source_feature_mask is None:
                    raise RuntimeError("content-cross-attn CTC probe requires source BNF features and mask")
                source_features = source_features.index_select(0, active_indices)
                source_feature_mask = source_feature_mask.index_select(0, active_indices).bool()
                if not bool(source_feature_mask.any(dim=1).all().item()):
                    bad = torch.nonzero(~source_feature_mask.any(dim=1), as_tuple=False).flatten().tolist()
                    raise RuntimeError(f"no_text rows are missing source BNF frames: selected_rows={bad}")
                with autocast_context(device):
                    hidden, hidden_mask = model._compute_content_cross_attn_memory(
                        source_features,
                        source_feature_mask,
                    )
                    if hidden is None:
                        raise RuntimeError("content_cross_attn_encoder returned no memory")
                    if hidden_mask is None:
                        hidden_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                    expected_hidden = int(model.content_ctc_head.net[0].normalized_shape[0])
                    if int(hidden.shape[-1]) != expected_hidden:
                        raise RuntimeError(
                            f"CTC memory/head width mismatch: memory={hidden.shape[-1]} head={expected_hidden}"
                        )
                    logits = model.content_ctc_head(hidden).float()
                batch_records = [batch_records[int(index)] for index in active_indices.detach().cpu().tolist()]
                if target_ids is not None:
                    target_ids = target_ids.index_select(0, active_indices)
                if target_ids_mask is not None:
                    target_ids_mask = target_ids_mask.index_select(0, active_indices)
            else:
                forward_kwargs = train_mod.build_forward_kwargs_from_batch(
                    batch,
                    timbre_memory_enabled=True,
                    channelwise_loss_weight=None,
                )
                with autocast_context(device):
                    outputs = model(**forward_kwargs)
                    selected = model._target_hidden_for_aux(outputs, target_mask)
                    if selected is None:
                        continue
                    hidden, hidden_mask = selected
                    logits = model.content_ctc_head(hidden).float()

            pred_ids = logits.argmax(dim=-1)
            blank_posteriors = (logits[..., blank_id] - logits.logsumexp(dim=-1)).exp()
            for idx in range(int(pred_ids.shape[0])):
                record = batch_records[idx]
                valid_mask = hidden_mask[idx].bool()
                greedy = [int(x) for x in pred_ids[idx][valid_mask].detach().cpu().tolist()]
                collapsed = collapse_ctc(greedy, blank_id)
                target = strip_target(
                    target_ids[idx],
                    target_ids_mask[idx] if target_ids_mask is not None else None,
                    blank_id,
                ) if target_ids is not None else []
                edit_distance = token_edit_distance(collapsed, target) if target else None
                ter = (float(edit_distance) / float(len(target))) if edit_distance is not None else None
                exact = bool(target and collapsed == target)
                blank_frames = sum(1 for token in greedy if int(token) == blank_id)
                total_frames = len(greedy)
                nonblank_frames = total_frames - blank_frames
                row_blank_posterior = float(blank_posteriors[idx][valid_mask].mean().item()) if total_frames else None
                row_nonblank_posterior = None if row_blank_posterior is None else 1.0 - row_blank_posterior
                nonblank_token_counts: dict[int, int] = {}
                for token in greedy:
                    if token != blank_id:
                        nonblank_token_counts[token] = nonblank_token_counts.get(token, 0) + 1
                dominant_token = max(nonblank_token_counts, key=nonblank_token_counts.get) if nonblank_token_counts else None
                dominant_share = (
                    float(nonblank_token_counts[dominant_token]) / float(nonblank_frames)
                    if dominant_token is not None and nonblank_frames > 0
                    else 0.0
                )
                punctuation_flags = []
                for token in collapsed:
                    token_text = decode_tokens(display_tokenizer, [token], token_offset)
                    token_value = "" if token_text is None else str(token_text).strip()
                    punctuation_flags.append(
                        bool(token_value)
                        and all(unicodedata.category(char).startswith(("P", "S")) for char in token_value)
                    )
                punctuation_tokens = sum(punctuation_flags)
                target_text = decode_tokens(display_tokenizer, target, token_offset)
                pred_text = decode_tokens(display_tokenizer, collapsed, token_offset)
                text_diagnostics = decoded_text_diagnostics(
                    pred_text,
                    collapsed_len=len(collapsed),
                    target_len=len(target),
                )
                row = {
                    "sample_id": record.get("sample_id"),
                    "mode": record.get("moss_codecvc_mode") or record.get("mode"),
                    "language": record.get("language"),
                    "content_ref_text": record.get("content_ref_text"),
                    "content_ctc_input_source": input_source,
                    "ctc_blank_id": blank_id,
                    "ctc_token_offset": token_offset,
                    "valid_bnf_frames": total_frames if content_cross_attn_mode else None,
                    "target_len": len(target),
                    "greedy_len": len(greedy),
                    "blank_frames": blank_frames,
                    "blank_frame_rate": (float(blank_frames) / float(total_frames)) if total_frames else None,
                    "nonblank_frames": nonblank_frames,
                    "nonblank_frame_rate": (float(nonblank_frames) / float(total_frames)) if total_frames else None,
                    "blank_posterior_mean": row_blank_posterior,
                    "nonblank_posterior_mean": row_nonblank_posterior,
                    "dominant_nonblank_frame_token_id": dominant_token,
                    "dominant_nonblank_frame_share": dominant_share,
                    "collapsed_len": len(collapsed),
                    "collapsed_to_target_ratio": (float(len(collapsed)) / float(len(target))) if target else None,
                    "collapsed_punctuation_tokens": punctuation_tokens,
                    "collapsed_punctuation_share": (
                        float(punctuation_tokens) / float(len(collapsed)) if collapsed else None
                    ),
                    "exact_match": exact,
                    "edit_distance": edit_distance,
                    "token_error_rate": ter,
                    "target_text": target_text,
                    "pred_text": pred_text,
                    "target_ids": target,
                    "pred_ids": collapsed,
                }
                row.update(text_diagnostics)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["rows"] += 1
                if not content_cross_attn_mode:
                    summary["no_text_rows"] += int(str(row["mode"] or "").strip().lower() == "no_text")
                summary["exact_matches"] += int(exact)
                summary["blank_frames"] += int(blank_frames)
                summary["nonblank_frames"] += int(nonblank_frames)
                summary["total_frames"] += int(total_frames)
                if row_blank_posterior is not None:
                    summary["blank_posterior_sum"] += row_blank_posterior * total_frames
                    summary["nonblank_posterior_sum"] += row_nonblank_posterior * total_frames
                summary["target_tokens"] += len(target)
                summary["collapsed_tokens"] += len(collapsed)
                summary["collapsed_punctuation_tokens"] += punctuation_tokens
                summary["collapsed_len_min"] = (
                    len(collapsed)
                    if summary["collapsed_len_min"] is None
                    else min(int(summary["collapsed_len_min"]), len(collapsed))
                )
                summary["collapsed_len_max"] = max(int(summary["collapsed_len_max"]), len(collapsed))
                summary["empty_predictions"] += int(text_diagnostics["empty_prediction"])
                summary["punctuation_only_collapses"] += int(text_diagnostics["punctuation_only_collapse"])
                summary["single_token_collapses"] += int(text_diagnostics["single_token_collapse"])
                summary["severe_length_collapses"] += int(text_diagnostics["severe_length_collapse"])
                summary["blank_dominant_rows"] += int(total_frames > 0 and blank_frames / total_frames >= 0.95)
                summary["dominant_nonblank_frame_share_sum"] += dominant_share
                if ter is not None:
                    summary["ter_sum"] += float(ter)
                    summary["ter_count"] += 1
                    summary["edit_distance"] += int(edit_distance)
    summary = finalize_summary(summary)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[ctc-probe] wrote={out_path} summary={summary_path} rows={summary['rows']} "
        f"input_source={input_source} greedy_ter={summary['greedy_ter']:.4f} "
        f"mean_row_ter={summary['mean_row_greedy_ter']:.4f} exact={summary['exact_rate']:.4f} "
        f"blank={summary['blank_frame_rate']:.4f} nonblank={summary['nonblank_frame_rate']:.4f} "
        f"nonblank_post={summary['nonblank_posterior_mean']:.4f} "
        f"punctuation_only={summary['punctuation_only_collapse_rate']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
