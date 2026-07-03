#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config
from moss_codecvc.io_utils import load_torch_file
from moss_codecvc.moss_codec import ensure_moss_on_path
from moss_codecvc.models import MossCodecVCTimbreMemoryWrapper, MossCodecVCTimbreSFTDataset
from moss_codecvc.models.auxiliary_losses import ContentCTCHead, compute_content_ctc_loss


DEFAULT_MODEL = ROOT / "outputs/lora_runs/ver2_3_debug_resume_evalfix/ablation_a_ce_route/step-1000"
DEFAULT_JSONL = (
    ROOT
    / "trainset/zh45w_en22w_no_text/sft/"
    "moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody."
    "with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl"
)
DEFAULT_DUMP = ROOT / "outputs/debug_ctc/ctc_head_only_overfit_hidden.pt"
DEFAULT_METRICS = ROOT / "outputs/debug_ctc/ctc_head_only_overfit_metrics.jsonl"
DEFAULT_REPORT = ROOT / "outputs/debug_ctc/ctc_head_only_overfit_report.json"


def import_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


probe_mod = import_file("moss_codecvc_ctc_probe_helpers", ROOT / "scripts/004019_ctc_greedy_decode_probe.py")
infer_mod = probe_mod.infer_mod
train_mod = probe_mod.train_mod


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Freeze teacher-forced H_target and overfit only a ContentCTCHead.")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL), help="LoRA adapter dir or full model dir used to dump H_target.")
    ap.add_argument("--base-model-path", default=None)
    ap.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    ap.add_argument("--dump-path", default=str(DEFAULT_DUMP))
    ap.add_argument("--output-jsonl", default=str(DEFAULT_METRICS))
    ap.add_argument("--output-report", default=str(DEFAULT_REPORT))
    ap.add_argument("--max-rows", type=int, default=128)
    ap.add_argument("--dump-batch-size", type=int, default=1)
    ap.add_argument("--train-batch-size", type=int, default=16)
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
    ap.add_argument("--reuse-dump", action="store_true")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--adapter-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--init-from-model-head", action="store_true")
    ap.add_argument("--save-head", default=str(ROOT / "outputs/debug_ctc/ctc_head_only_overfit_head.pt"))
    return ap.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_timbre_config_args(args: argparse.Namespace) -> argparse.Namespace:
    return probe_mod.make_timbre_config_args(args)


def resolve_device(device_arg: str) -> torch.device:
    normalized = infer_mod.normalize_device_arg(device_arg)
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def load_model_and_loader(args: argparse.Namespace, device: torch.device):
    cfg = load_config(args.config)
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

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_path = Path(args.model_path).expanduser()
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
    train_mod.validate_content_tokenizer_consistency(
        records,
        enabled=True,
        allow_mixed=bool(args.allow_mixed_content_tokenizers),
    )
    manifest_vocab = int(train_mod.infer_content_ctc_vocab_size_from_manifest(records))
    if manifest_vocab <= 1:
        raise ValueError("content_ctc_vocab_size could not be inferred from manifest")
    uses_manifest_tokens = manifest_vocab > 1

    timbre_config = train_mod.resolve_timbre_memory_config(cfg, make_timbre_config_args(args))
    timbre_config.content_ctc_weight = 1.0
    timbre_config.content_ctc_vocab_size = int(manifest_vocab)
    timbre_config.content_ctc_blank_id = int(getattr(timbre_config, "content_ctc_blank_id", 0))
    timbre_config.content_ctc_token_offset = int(getattr(timbre_config, "content_ctc_token_offset", 1))

    base_dataset = LazyMossTTSSFTDataset(records=records, processor=processor, n_vq=n_vq)
    dataset = MossCodecVCTimbreSFTDataset(
        records=records,
        base_dataset=base_dataset,
        n_vq=n_vq,
        audio_pad_code=int(processor.model_config.audio_pad_code),
        content_tokenizer=processor.tokenizer if not uses_manifest_tokens else None,
        content_ctc_token_offset=int(timbre_config.content_ctc_token_offset),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.dump_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=dataset.collate_fn,
    )

    base_model = MossTTSDelayModel.from_pretrained(model_load_path, torch_dtype=dtype, trust_remote_code=True)
    if not hasattr(base_model, "prepare_inputs_for_generation"):
        base_model.prepare_inputs_for_generation = lambda *a, **kw: kw
    if not hasattr(base_model, "generation_config"):
        base_model.generation_config = GenerationConfig.from_model_config(base_model.config)
    if is_lora_adapter:
        base_model = PeftModel.from_pretrained(base_model, str(peft_model_path))
    if is_lora_adapter and (model_path / "timbre_memory_config.json").exists() and (model_path / "timbre_memory_adapter.pt").exists():
        overrides = {
            "speaker_encoder_type": args.speaker_encoder_type,
            "speaker_encoder_path": args.speaker_encoder_path,
            "speaker_embedding_dim": int(args.speaker_embedding_dim),
        }
        model = MossCodecVCTimbreMemoryWrapper.from_pretrained_timbre_memory(
            base_model,
            model_path,
            map_location="cpu",
            config_overrides=overrides,
        )
    else:
        model = MossCodecVCTimbreMemoryWrapper(base_model, timbre_config)
    model = model.to(device).eval()
    infer_mod.apply_source_gate_floor_for_inference(model, args.source_gate_floor)
    display_tokenizer = probe_mod.resolve_display_tokenizer(records, processor.tokenizer)
    return model, loader, records, display_tokenizer, {
        "vocab_size": manifest_vocab,
        "blank_id": int(timbre_config.content_ctc_blank_id),
        "token_offset": int(timbre_config.content_ctc_token_offset),
    }


def dump_hidden(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model, loader, records, display_tokenizer, ctc_meta = load_model_and_loader(args, device)
    hidden_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    meta_rows: list[dict[str, Any]] = []
    model_head_state = None
    if getattr(model, "content_ctc_head", None) is not None:
        model_head_state = {key: value.detach().cpu() for key, value in model.content_ctc_head.state_dict().items()}

    record_cursor = 0
    with torch.inference_mode():
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
            target_ids = batch.get("content_token_ids")
            target_ids_mask = batch.get("content_token_ids_mask")
            if target_ids is None:
                continue
            for idx in range(batch_size):
                cur_hidden = hidden[idx][hidden_mask[idx]].detach().cpu().to(torch.float16)
                cur_target = probe_mod.strip_target(
                    target_ids[idx],
                    target_ids_mask[idx] if target_ids_mask is not None else None,
                    int(ctc_meta["blank_id"]),
                )
                if cur_hidden.numel() == 0 or not cur_target or len(cur_target) > int(cur_hidden.shape[0]):
                    continue
                record = batch_records[idx]
                hidden_rows.append(cur_hidden)
                target_rows.append(torch.tensor(cur_target, dtype=torch.long))
                meta_rows.append(
                    {
                        "sample_id": record.get("sample_id"),
                        "mode": record.get("moss_codecvc_mode") or record.get("mode"),
                        "language": record.get("language"),
                        "content_ref_text": record.get("content_ref_text"),
                        "target_text": probe_mod.decode_tokens(display_tokenizer, cur_target, int(ctc_meta["token_offset"])),
                        "hidden_len": int(cur_hidden.shape[0]),
                        "target_len": len(cur_target),
                    }
                )
            if len(hidden_rows) >= int(args.max_rows):
                break

    if not hidden_rows:
        raise RuntimeError("no valid hidden/target rows were dumped")
    hidden_padded = pad_sequence(hidden_rows, batch_first=True, padding_value=0.0)
    hidden_mask = pad_sequence(
        [torch.ones(row.shape[0], dtype=torch.bool) for row in hidden_rows],
        batch_first=True,
        padding_value=False,
    )
    target_padded = pad_sequence(target_rows, batch_first=True, padding_value=int(ctc_meta["blank_id"]))
    target_mask = pad_sequence(
        [torch.ones(row.shape[0], dtype=torch.bool) for row in target_rows],
        batch_first=True,
        padding_value=False,
    )
    dump = {
        "hidden": hidden_padded,
        "hidden_mask": hidden_mask,
        "target_ids": target_padded,
        "target_mask": target_mask,
        "records": meta_rows,
        "ctc_meta": ctc_meta,
        "model_path": str(Path(args.model_path).expanduser().resolve(strict=False)),
        "jsonl": str(Path(args.jsonl).expanduser().resolve(strict=False)),
        "model_head_state": model_head_state,
    }
    dump_path = Path(args.dump_path).expanduser()
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, dump_path)
    print(f"[ctc-head-overfit] dumped rows={len(meta_rows)} hidden={tuple(hidden_padded.shape)} path={dump_path}")
    return dump


def load_or_dump_hidden(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dump_path = Path(args.dump_path).expanduser()
    if bool(args.reuse_dump) and dump_path.exists():
        return load_torch_file(dump_path, map_location="cpu")
    return dump_hidden(args, device)


def make_batches(num_rows: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    indices = list(range(num_rows))
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def collapse_and_score(
    pred_ids: torch.Tensor,
    hidden_mask: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    blank_id: int,
) -> tuple[int, float, int, int, list[dict[str, Any]]]:
    exact = 0
    ter_sum = 0.0
    ter_count = 0
    blank_frames = 0
    total_frames = 0
    rows: list[dict[str, Any]] = []
    for idx in range(int(pred_ids.shape[0])):
        greedy = [int(x) for x in pred_ids[idx][hidden_mask[idx]].detach().cpu().tolist()]
        collapsed = probe_mod.collapse_ctc(greedy, int(blank_id))
        target = probe_mod.strip_target(target_ids[idx], target_mask[idx], int(blank_id))
        ter = probe_mod.token_error_rate(collapsed, target)
        is_exact = bool(target and collapsed == target)
        exact += int(is_exact)
        if ter is not None:
            ter_sum += float(ter)
            ter_count += 1
        blank_frames += sum(1 for item in greedy if int(item) == int(blank_id))
        total_frames += len(greedy)
        rows.append(
            {
                "target_ids": target,
                "pred_ids": collapsed,
                "exact_match": is_exact,
                "token_error_rate": ter,
            }
        )
    return exact, ter_sum, ter_count, blank_frames, total_frames, rows


def evaluate(
    head: ContentCTCHead,
    dump: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    blank_id: int,
    token_offset: int,
    step: int,
) -> dict[str, Any]:
    head.eval()
    hidden = dump["hidden"]
    hidden_mask = dump["hidden_mask"]
    target_ids = dump["target_ids"]
    target_mask = dump["target_mask"]
    records = dump.get("records") or []
    total = int(hidden.shape[0])
    exact = 0
    ter_sum = 0.0
    ter_count = 0
    blank_frames = 0
    total_frames = 0
    blank_prob_sum = 0.0
    prob_frames = 0
    examples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            cur_hidden = hidden[start:end].to(device=device, dtype=next(head.parameters()).dtype)
            cur_hidden_mask = hidden_mask[start:end].to(device=device)
            cur_target = target_ids[start:end].to(device=device)
            cur_target_mask = target_mask[start:end].to(device=device)
            logits = head(cur_hidden).float()
            probs = torch.softmax(logits, dim=-1)
            valid_probs = probs[..., int(blank_id)][cur_hidden_mask]
            blank_prob_sum += float(valid_probs.sum().detach().item())
            prob_frames += int(valid_probs.numel())
            pred = logits.argmax(dim=-1)
            batch_exact, batch_ter_sum, batch_ter_count, batch_blank, batch_frames, rows = collapse_and_score(
                pred,
                cur_hidden_mask,
                cur_target,
                cur_target_mask,
                blank_id=int(blank_id),
            )
            exact += batch_exact
            ter_sum += batch_ter_sum
            ter_count += batch_ter_count
            blank_frames += batch_blank
            total_frames += batch_frames
            if len(examples) < 5:
                for local_idx, row in enumerate(rows):
                    record = records[start + local_idx] if start + local_idx < len(records) else {}
                    item = {
                        "sample_id": record.get("sample_id"),
                        "content_ref_text": record.get("content_ref_text"),
                        "target_text": record.get("target_text"),
                        "pred_text": None,
                        "target_ids": row["target_ids"],
                        "pred_ids": row["pred_ids"],
                        "exact_match": row["exact_match"],
                        "token_error_rate": row["token_error_rate"],
                    }
                    examples.append(item)
                    if len(examples) >= 5:
                        break
    decoder = None
    vocab_path = ""
    for record in records:
        del record
    try:
        first_meta_path = ""
        # The target_text is already decoded in the dump; only pred_text needs a decoder.
        source_jsonl = Path(str(dump.get("jsonl") or "")).expanduser()
        if source_jsonl.exists():
            with source_jsonl.open("r", encoding="utf-8") as handle:
                first = json.loads(handle.readline())
                vocab_path = str(first.get("content_vocab_path") or "")
        if vocab_path:
            decoder = probe_mod.SentencePieceDecoder(vocab_path)
    except Exception:
        decoder = None
    if decoder is not None:
        for item in examples:
            item["pred_text"] = probe_mod.decode_tokens(decoder, item["pred_ids"], int(token_offset))
    return {
        "step": int(step),
        "rows": total,
        "exact_rate": float(exact) / max(1, total),
        "mean_token_error_rate": float(ter_sum) / max(1, ter_count),
        "blank_frame_rate": float(blank_frames) / max(1, total_frames),
        "blank_posterior_mean": float(blank_prob_sum) / max(1, prob_frames),
        "nonblank_posterior_mean": 1.0 - (float(blank_prob_sum) / max(1, prob_frames)),
        "examples": examples,
    }


def train_head(args: argparse.Namespace, dump: dict[str, Any], device: torch.device) -> dict[str, Any]:
    ctc_meta = dump["ctc_meta"]
    blank_id = int(ctc_meta["blank_id"])
    token_offset = int(ctc_meta["token_offset"])
    vocab_size = int(ctc_meta["vocab_size"])
    hidden = dump["hidden"]
    hidden_mask = dump["hidden_mask"]
    target_ids = dump["target_ids"]
    target_mask = dump["target_mask"]
    head = ContentCTCHead(
        hidden_size=int(hidden.shape[-1]),
        vocab_size=vocab_size,
        adapter_dim=int(args.adapter_dim),
        dropout=float(args.dropout),
    )
    if bool(args.init_from_model_head) and dump.get("model_head_state"):
        head.load_state_dict(dump["model_head_state"], strict=True)
    head = head.to(device)
    if device.type == "cuda":
        head = head.to(dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = random.Random(int(args.seed))
    metrics_path = Path(args.output_jsonl).expanduser()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    last_metrics: dict[str, Any] | None = None
    with metrics_path.open("w", encoding="utf-8") as handle:
        for step in range(1, int(args.steps) + 1):
            head.train()
            step_losses = []
            for batch_indices in make_batches(int(hidden.shape[0]), int(args.train_batch_size), rng):
                idx = torch.tensor(batch_indices, dtype=torch.long)
                cur_hidden = hidden.index_select(0, idx).to(device=device, dtype=next(head.parameters()).dtype)
                cur_hidden_mask = hidden_mask.index_select(0, idx).to(device=device)
                cur_target = target_ids.index_select(0, idx).to(device=device)
                cur_target_mask = target_mask.index_select(0, idx).to(device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = head(cur_hidden)
                state = compute_content_ctc_loss(
                    logits,
                    cur_hidden_mask,
                    cur_target,
                    cur_target_mask,
                    blank_id=blank_id,
                )
                if state.loss is None:
                    continue
                state.loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
                step_losses.append(float(state.loss.detach().cpu().item()))
            if step == 1 or step % int(args.eval_every) == 0 or step == int(args.steps):
                metrics = evaluate(
                    head,
                    dump,
                    device=device,
                    batch_size=int(args.train_batch_size),
                    blank_id=blank_id,
                    token_offset=token_offset,
                    step=step,
                )
                metrics["train_ctc_loss_mean"] = sum(step_losses) / max(1, len(step_losses))
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                handle.flush()
                last_metrics = metrics
                print(
                    "[ctc-head-overfit] "
                    f"step={step} loss={metrics['train_ctc_loss_mean']:.4f} "
                    f"exact={metrics['exact_rate']:.4f} ter={metrics['mean_token_error_rate']:.4f} "
                    f"blank_frame={metrics['blank_frame_rate']:.4f} nonblank_post={metrics['nonblank_posterior_mean']:.4f}",
                    flush=True,
                )
    save_head = Path(args.save_head).expanduser()
    save_head.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()}, "ctc_meta": ctc_meta}, save_head)
    report = {
        "status": "complete",
        "dump_path": str(Path(args.dump_path).expanduser()),
        "metrics_jsonl": str(metrics_path),
        "save_head": str(save_head),
        "rows": int(hidden.shape[0]),
        "hidden_shape": list(hidden.shape),
        "target_shape": list(target_ids.shape),
        "ctc_meta": ctc_meta,
        "final": last_metrics,
        "acceptance": {
            "non_empty_decode": bool(last_metrics and last_metrics.get("blank_frame_rate", 1.0) < 1.0),
            "exact_gt_zero": bool(last_metrics and last_metrics.get("exact_rate", 0.0) > 0.0),
            "ter_below_one": bool(last_metrics and last_metrics.get("mean_token_error_rate", 1.0) < 1.0),
        },
    }
    report_path = Path(args.output_report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    dump = load_or_dump_hidden(args, device)
    report = train_head(args, dump, device)
    print(json.dumps({"report": str(Path(args.output_report).expanduser()), "final": report.get("final"), "acceptance": report.get("acceptance")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
