#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config
from moss_codecvc.moss_codec import ensure_moss_on_path
from moss_codecvc.models import MossCodecVCTimbreMemoryWrapper, MossCodecVCTimbreSFTDataset


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


def token_error_rate(pred: list[int], target: list[int]) -> float | None:
    if not target:
        return None
    prev = list(range(len(target) + 1))
    for i, pred_token in enumerate(pred, start=1):
        cur = [i] + [0] * len(target)
        for j, target_token in enumerate(target, start=1):
            cost = 0 if pred_token == target_token else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return float(prev[-1]) / float(len(target))


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


def make_timbre_config_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
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
    timbre_config = train_mod.resolve_timbre_memory_config(cfg, make_timbre_config_args(args))
    timbre_config.content_ctc_weight = 1.0
    train_mod.validate_content_tokenizer_consistency(records, enabled=True, allow_mixed=bool(args.allow_mixed_content_tokenizers))
    manifest_vocab = train_mod.infer_content_ctc_vocab_size_from_manifest(records)
    uses_manifest_tokens = manifest_vocab > 1
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
    if is_lora_adapter and (model_path / "timbre_memory_config.json").exists() and (model_path / "timbre_memory_adapter.pt").exists():
        overrides = {
            "speaker_encoder_type": args.speaker_encoder_type,
            "speaker_encoder_path": args.speaker_encoder_path,
            "speaker_embedding_dim": int(args.speaker_embedding_dim),
        }
        model = MossCodecVCTimbreMemoryWrapper.from_pretrained_timbre_memory(model, model_path, map_location="cpu", config_overrides=overrides)
    else:
        model = MossCodecVCTimbreMemoryWrapper(model, timbre_config)
    if model.content_ctc_head is None:
        raise RuntimeError("content_ctc_head is not available; adapter may not contain the CTC head")
    model = model.to(device).eval()
    infer_mod.apply_source_gate_floor_for_inference(model, args.source_gate_floor)

    out_path = Path(args.output_jsonl).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blank_id = int(model.timbre_memory_config.content_ctc_blank_id)
    offset = int(model.timbre_memory_config.content_ctc_token_offset)
    summary = {"rows": 0, "exact": 0, "ter_sum": 0.0, "ter_count": 0, "blank_frames": 0, "total_frames": 0}
    with out_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        record_cursor = 0
        for batch in loader:
            batch_size = int(batch["input_ids"].shape[0])
            batch_records = [records[record_cursor + offset] for offset in range(batch_size)]
            record_cursor += batch_size
            batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
            labels = batch.get("labels")
            target_mask = (labels != -100).any(dim=-1) if labels is not None else None
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=labels,
                timbre_ref_codes=batch.get("timbre_ref_codes"),
                timbre_ref_mask=batch.get("timbre_ref_mask"),
                timbre_ref_speaker_embedding_path=batch.get("timbre_ref_speaker_embedding_path"),
                timbre_ref_speaker_audio_path=batch.get("timbre_ref_speaker_audio_path"),
                source_prompt_positions=batch.get("source_prompt_positions"),
                timbre_ref_prompt_positions=batch.get("timbre_ref_prompt_positions"),
                role_ids=batch.get("role_ids"),
                target_position_mask=target_mask,
            )
            selected = model._target_hidden_for_aux(outputs, target_mask)
            if selected is None:
                continue
            hidden, hidden_mask = selected
            pred_ids = model.content_ctc_head(hidden).float().argmax(dim=-1)
            target_ids = batch.get("content_token_ids")
            target_ids_mask = batch.get("content_token_ids_mask")
            for idx in range(int(pred_ids.shape[0])):
                record = batch_records[idx]
                greedy = [int(x) for x in pred_ids[idx][hidden_mask[idx]].detach().cpu().tolist()]
                collapsed = collapse_ctc(greedy, blank_id)
                target = strip_target(target_ids[idx], target_ids_mask[idx] if target_ids_mask is not None else None, blank_id) if target_ids is not None else []
                ter = token_error_rate(collapsed, target)
                exact = bool(target and collapsed == target)
                blank_frames = sum(1 for token in greedy if int(token) == blank_id)
                total_frames = len(greedy)
                row = {
                    "sample_id": record.get("sample_id"),
                    "mode": record.get("moss_codecvc_mode") or record.get("mode"),
                    "language": record.get("language"),
                    "content_ref_text": record.get("content_ref_text"),
                    "ctc_blank_id": blank_id,
                    "ctc_token_offset": offset,
                    "target_len": len(target),
                    "greedy_len": len(greedy),
                    "blank_frames": blank_frames,
                    "blank_frame_rate": (float(blank_frames) / float(total_frames)) if total_frames else None,
                    "collapsed_len": len(collapsed),
                    "exact_match": exact,
                    "token_error_rate": ter,
                    "target_text": decode_tokens(display_tokenizer, target, offset),
                    "pred_text": decode_tokens(display_tokenizer, collapsed, offset),
                    "target_ids": target,
                    "pred_ids": collapsed,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["rows"] += 1
                summary["exact"] += int(exact)
                summary["blank_frames"] += int(blank_frames)
                summary["total_frames"] += int(total_frames)
                if ter is not None:
                    summary["ter_sum"] += float(ter)
                    summary["ter_count"] += 1
    exact_rate = summary["exact"] / max(1, summary["rows"])
    mean_ter = summary["ter_sum"] / max(1, summary["ter_count"])
    blank_rate = summary["blank_frames"] / max(1, summary["total_frames"])
    print(f"[ctc-probe] wrote={out_path} rows={summary['rows']} exact_rate={exact_rate:.4f} mean_token_error_rate={mean_ter:.4f} blank_frame_rate={blank_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
