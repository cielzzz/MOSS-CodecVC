#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torchaudio
from transformers import GenerationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_CONFIG = ROOT / "configs/remote_full.yaml"
DEFAULT_BASE_MODEL = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-TTS"
)
DEFAULT_SPEAKER_ENCODER = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb"
)
DEFAULT_SOURCE_CONTENT_SPM = ROOT / "trainset/shared_content_tokenizers/spm_multilingual_byte_fallback_v1.model"


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else value


def env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def env_float(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_bool_or_none(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def checkpoint_timbre_side_only(model_path: str) -> bool:
    config_path = Path(model_path).expanduser() / "timbre_memory_config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return False
    return bool(config.get("timbre_side_only", False))


def load_infer_module():
    path = ROOT / "scripts/003001_infer_moss_codecvc.py"
    spec = importlib.util.spec_from_file_location("moss_codecvc_single_infer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load infer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem[:180] or "case"


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted_case_ids = set(args.case_id)
    selected: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()

    for row in rows:
        case_id = str(row.get("case_id") or "")
        mode = str(row.get("mode") or "")
        cell = str(row.get("cell") or "")
        if wanted_case_ids and case_id not in wanted_case_ids:
            continue
        if args.mode != "all" and mode != args.mode:
            continue
        if mode not in {"no_text", "text"}:
            continue
        cell_key = f"{mode}:{cell}"
        if args.per_mode > 0 and mode_counts[mode] >= args.per_mode:
            continue
        if args.per_cell > 0 and cell_counts[cell_key] >= args.per_cell:
            continue

        selected.append(row)
        mode_counts[mode] += 1
        cell_counts[cell_key] += 1
        if args.max_cases > 0 and len(selected) >= args.max_cases:
            break

    if args.num_shards > 1:
        selected = [
            row for idx, row in enumerate(selected) if idx % args.num_shards == args.shard_index
        ]
    return selected


def required_text(row: dict[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    if text and text != "<NO_TEXT>":
        return text
    return str(row.get("content_ref_text") or row.get("source_text") or "").strip()


def source_content_text_with_key(row: dict[str, Any]) -> tuple[str, str]:
    for key in ("source_text", "content_ref_text", "asr_src_text", "source_asr_text"):
        text = str(row.get(key) or "").strip()
        if text and text != "<NO_TEXT>":
            return text, key
    return "", ""


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_generation_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    print(f"[persistent-infer] generation seed={int(seed)}", flush=True)


class PersistentCodecVCInfer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        set_generation_seed(args.seed)
        self.mod = load_infer_module()
        cfg = self.mod.load_config(args.config)
        self.cfg = cfg
        self.n_vq = args.n_vq or int(self.mod.deep_get(cfg, "moss.default_n_vq", 32))
        self.mod.ensure_moss_on_path(self.mod.deep_get(cfg, "moss.root"))
        self.mod.patch_torchaudio_load_with_soundfile_fallback()

        from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
        from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor
        from peft import PeftConfig, PeftModel

        device_arg = self.mod.normalize_device_arg(args.device)
        self.device = torch.device(
            device_arg if torch.cuda.is_available() or not device_arg.startswith("cuda") else "cpu"
        )
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        model_path = Path(args.model_path)
        self.model_path = model_path
        is_lora_adapter = (model_path / "adapter_config.json").exists()
        peft_model_path = (
            self.mod.prepare_peft_adapter_dir_for_inference(model_path) if is_lora_adapter else model_path
        )
        if is_lora_adapter:
            peft_cfg = PeftConfig.from_pretrained(str(peft_model_path))
            base_model_path = args.base_model_path or peft_cfg.base_model_name_or_path
            if not base_model_path:
                raise ValueError("--base-model-path is required for this LoRA adapter.")
            processor_path = base_model_path
            model_load_path = base_model_path
        else:
            processor_path = args.model_path
            model_load_path = args.model_path

        codec_path = self.mod.deep_get(cfg, "moss.codec_path")
        self.processor = MossTTSDelayProcessor.from_pretrained(
            processor_path,
            codec_path=codec_path,
            trust_remote_code=True,
        )
        self.processor.audio_tokenizer.to(self.device)
        model = MossTTSDelayModel.from_pretrained(
            model_load_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if not hasattr(model, "prepare_inputs_for_generation"):
            model.prepare_inputs_for_generation = lambda *a, **kw: kw
        if not hasattr(model, "generation_config"):
            model.generation_config = GenerationConfig.from_model_config(model.config)
        if is_lora_adapter:
            print(f"[persistent-infer] loading PEFT adapter from {peft_model_path}", flush=True)
            model = PeftModel.from_pretrained(model, str(peft_model_path))
            print("[persistent-infer] PEFT adapter loaded", flush=True)
        self.use_timbre_memory = (
            is_lora_adapter
            and not args.disable_timbre_memory
            and (model_path / "timbre_memory_config.json").exists()
            and (model_path / "timbre_memory_adapter.pt").exists()
        )
        if args.timbre_side_only and not self.use_timbre_memory:
            raise ValueError("--timbre-side-only requires an adapter with timbre memory; otherwise timbre conditioning is removed.")
        if self.use_timbre_memory:
            print("[persistent-infer] loading timbre memory adapter", flush=True)
            overrides = {}
            if args.speaker_encoder_type:
                overrides["speaker_encoder_type"] = args.speaker_encoder_type
            if args.speaker_encoder_path:
                overrides["speaker_encoder_path"] = args.speaker_encoder_path
            if args.speaker_embedding_dim is not None:
                overrides["speaker_embedding_dim"] = int(args.speaker_embedding_dim)
            model = self.mod.MossCodecVCTimbreMemoryWrapper.from_pretrained_timbre_memory(
                model,
                model_path,
                map_location="cpu",
                config_overrides=overrides or None,
            )
            print("[persistent-infer] timbre memory adapter loaded", flush=True)
        print(f"[persistent-infer] moving model to {self.device}", flush=True)
        self.model = model.to(self.device).eval()
        print(f"[persistent-infer] model on {self.device}", flush=True)
        self.mod.apply_source_gate_floor_for_inference(self.model, args.source_gate_floor)
        monotonic_bias_strength = (
            0.0 if args.disable_source_semantic_monotonic_bias else args.source_semantic_monotonic_bias_strength
        )
        self.mod.apply_source_semantic_debug_overrides(
            self.model,
            position_scale=args.source_semantic_position_scale,
            monotonic_bias_strength=monotonic_bias_strength,
            monotonic_bias_width=args.source_semantic_monotonic_bias_width,
        )
        print(
            "[persistent-infer] model loaded "
            f"model_path={args.model_path} device={self.device} n_vq={self.n_vq} "
            f"timbre_memory={self.use_timbre_memory} timbre_side_only={bool(args.timbre_side_only)}",
            flush=True,
        )

    def _gen_kwargs(
        self,
        *,
        no_text: bool,
        text: str | None,
        source_audio: str,
        source_codes: torch.Tensor,
    ) -> tuple[dict[str, Any], int, int]:
        args = self.args
        cfg = self.cfg
        n_vq = self.n_vq
        text_temperature = (
            args.temperature
            if args.temperature is not None
            else float(self.mod.deep_get(cfg, "inference.temperature", 0.8))
        )
        text_top_p = args.top_p if args.top_p is not None else float(self.mod.deep_get(cfg, "inference.top_p", 0.9))
        text_top_k = args.top_k if args.top_k is not None else int(self.mod.deep_get(cfg, "inference.top_k", 50))
        no_text_budget_frames = int(
            math.ceil(float(source_codes.shape[0]) * max(0.01, float(args.no_text_duration_budget_ratio)))
        )
        no_text_base_new_tokens = no_text_budget_frames + int(n_vq)
        no_text_soft_budget = bool(no_text and args.no_text_soft_duration_budget)
        config_max_new_tokens = int(self.mod.deep_get(cfg, "inference.max_new_tokens", 2048))
        if args.max_new_tokens is not None:
            max_new_tokens = int(args.max_new_tokens)
            reason = "explicit"
        elif no_text:
            token_margin = max(0, int(args.no_text_max_token_margin))
            if no_text_soft_budget:
                soft_margin = (
                    int(args.no_text_soft_extra_token_margin)
                    if args.no_text_soft_extra_token_margin is not None
                    else int(n_vq)
                )
                token_margin = max(token_margin, max(0, soft_margin))
            max_new_tokens = no_text_base_new_tokens + token_margin
            reason = (
                f"ceil(source_codec_len({int(source_codes.shape[0])})*"
                f"ratio({float(args.no_text_duration_budget_ratio):.3f}))+n_vq({int(n_vq)})+margin({token_margin})"
            )
        elif args.text_auto_max_new_tokens:
            max_new_tokens, reason = self.mod.estimate_text_max_new_tokens(
                text,
                n_vq=n_vq,
                source_codec_len=int(source_codes.shape[0]),
                source_audio_seconds=self.mod.audio_duration_seconds(source_audio),
                config_ceiling=config_max_new_tokens,
                cjk_chars_per_second=float(args.text_cjk_chars_per_second),
                latin_words_per_second=float(args.text_latin_words_per_second),
                duration_margin=float(args.text_duration_margin),
                extra_tokens=int(args.text_extra_new_tokens),
                min_tokens=int(args.text_min_new_tokens_floor),
            )
        else:
            max_new_tokens = config_max_new_tokens
            reason = "config"
        if args.min_new_tokens is not None:
            min_new_tokens = max(0, int(args.min_new_tokens))
        elif no_text_soft_budget:
            min_new_tokens = 0
        elif no_text and args.max_new_tokens is None:
            min_new_tokens = max_new_tokens
        else:
            min_new_tokens = 0
        if args.min_audio_tokens is not None:
            min_audio_tokens = max(0, int(args.min_audio_tokens))
        elif no_text_soft_budget:
            min_audio_tokens = max(
                0,
                int(math.floor(float(source_codes.shape[0]) * max(0.0, float(args.no_text_soft_min_audio_ratio)))),
            )
        elif no_text:
            min_audio_tokens = int(source_codes.shape[0])
        else:
            min_audio_tokens = 0

        audio_temperature = args.no_text_audio_temperature if no_text and args.no_text_audio_temperature is not None else args.audio_temperature
        audio_top_p = args.no_text_audio_top_p if no_text and args.no_text_audio_top_p is not None else args.audio_top_p
        audio_top_k = args.no_text_audio_top_k if no_text and args.no_text_audio_top_k is not None else args.audio_top_k
        audio_repetition_penalty = (
            args.no_text_audio_repetition_penalty
            if no_text and args.no_text_audio_repetition_penalty is not None
            else args.audio_repetition_penalty
        )
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "text_temperature": text_temperature,
            "text_top_p": text_top_p,
            "text_top_k": text_top_k,
            "audio_temperature": (
                float(audio_temperature)
                if audio_temperature is not None
                else float(self.mod.deep_get(cfg, "inference.audio_temperature", 1.7))
            ),
            "audio_top_p": (
                float(audio_top_p)
                if audio_top_p is not None
                else float(self.mod.deep_get(cfg, "inference.audio_top_p", 0.8))
            ),
            "audio_top_k": (
                int(audio_top_k)
                if audio_top_k is not None
                else int(self.mod.deep_get(cfg, "inference.audio_top_k", 25))
            ),
            "audio_repetition_penalty": (
                float(audio_repetition_penalty)
                if audio_repetition_penalty is not None
                else float(self.mod.deep_get(cfg, "inference.audio_repetition_penalty", 1.0))
            ),
        }
        print(
            "[persistent-infer] generation budget "
            f"max_new_tokens={max_new_tokens} ({reason}) "
            f"min_new_tokens={min_new_tokens} min_audio_tokens={min_audio_tokens} "
            f"audio_top_k={gen_kwargs['audio_top_k']}",
            flush=True,
        )
        return gen_kwargs, min_new_tokens, min_audio_tokens

    def run_case(self, row: dict[str, Any], output_wav: Path) -> dict[str, Any]:
        args = self.args
        mode = str(row.get("mode") or "")
        no_text = mode == "no_text"
        source_audio = str(row.get("source_audio") or "")
        timbre_ref_audio = str(row.get("timbre_ref_audio") or "")
        if not source_audio or not Path(source_audio).exists():
            raise FileNotFoundError(f"source_audio not found: {source_audio}")
        if not timbre_ref_audio or not Path(timbre_ref_audio).exists():
            raise FileNotFoundError(f"timbre_ref_audio not found: {timbre_ref_audio}")
        text = None if no_text else required_text(row)
        if not no_text and not text:
            raise ValueError(f"text mode row has empty text: {row.get('case_id')}")
        source_content_text, _source_content_key = source_content_text_with_key(row)

        source_codes, timbre_codes = self.processor.encode_audios_from_path(
            [source_audio, timbre_ref_audio],
            n_vq=self.n_vq,
        )
        vc_mode = self.mod.VC_MODE_NO_TEXT if no_text else self.mod.VC_MODE_TEXT
        if args.instruction:
            instruction = args.instruction
        elif vc_mode == self.mod.VC_MODE_TEXT:
            instruction = self.mod.deep_get(self.cfg, "instruction.text_prosody") or self.mod.deep_get(
                self.cfg, "instruction.default"
            )
        else:
            instruction = (
                self.mod.deep_get(self.cfg, "instruction.no_text")
                or self.mod.deep_get(self.cfg, "instruction.prosody_no_timbre")
                or self.mod.deep_get(self.cfg, "instruction.default")
            )
        instruction = self.mod.apply_vc_mode_token(instruction, vc_mode, enabled=not args.disable_mode_token)
        prompt_text = args.no_text_placeholder if no_text else text
        prompt_references = [source_codes] if args.timbre_side_only else [source_codes, timbre_codes]
        user_message = self.processor.build_user_message(
            text=prompt_text,
            reference=prompt_references,
            instruction=instruction,
            tokens=int(source_codes.shape[0]),
            language=args.language,
            quality="high",
        )
        inputs = self.processor([[user_message]], mode="generation", n_vq=self.n_vq)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        gen_kwargs, min_new_tokens, min_audio_tokens = self._gen_kwargs(
            no_text=no_text,
            text=text,
            source_audio=source_audio,
            source_codes=source_codes,
        )

        print(
            "[persistent-infer] encoded codec shapes "
            f"case={row.get('case_id')} source={tuple(source_codes.shape)} "
            f"timbre_ref={tuple(timbre_codes.shape)} prompt_refs={len(prompt_references)} "
            f"timbre_side_only={bool(args.timbre_side_only)} prompt={tuple(inputs['input_ids'].shape)}",
            flush=True,
        )
        if self.use_timbre_memory:
            mode_id = int(getattr(self.model, "MODE_TO_ID", {}).get(vc_mode, 0))
            if mode_id > 0:
                gen_kwargs["vc_mode_id"] = torch.tensor([mode_id], dtype=torch.long, device=self.device)
            timbre_codes_for_memory = torch.as_tensor(
                timbre_codes,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            gen_kwargs["timbre_ref_codes"] = timbre_codes_for_memory
            gen_kwargs["timbre_ref_mask"] = torch.ones(
                timbre_codes_for_memory.shape[:2],
                dtype=torch.bool,
                device=self.device,
            )
            if args.timbre_ref_speaker_embedding_path:
                gen_kwargs["timbre_ref_speaker_embedding_path"] = [args.timbre_ref_speaker_embedding_path]
            gen_kwargs["timbre_ref_speaker_audio_path"] = [timbre_ref_audio]
            if getattr(self.model.timbre_memory_config, "use_role_routing", False):
                prompt_role_ids = self.mod.infer_prompt_role_ids_from_audio_spans(
                    inputs["input_ids"],
                    audio_pad_code=int(self.model.config.audio_pad_code),
                )
                print(
                    "[persistent-infer] role routing "
                    f"counts={self.mod.count_roles(prompt_role_ids).as_dict()}",
                    flush=True,
                )
            source_semantic_encoder = getattr(self.model, "source_semantic_memory_encoder", None)
            text_gate = float(getattr(self.model.timbre_memory_config, "source_semantic_text_gate", 0.0))
            learned_text_gate = bool(
                getattr(self.model.timbre_memory_config, "source_semantic_allow_learned_text_gate", False)
            )
            source_content_memory_type = str(
                getattr(self.model.timbre_memory_config, "source_content_memory_type", "hubert_continuous")
                or "hubert_continuous"
            ).strip().lower()
            source_content_memory_type = self.mod.normalize_source_content_memory_type(source_content_memory_type)
            should_use_source_semantic = (
                source_semantic_encoder is not None
                and not args.disable_source_semantic_memory
                and (no_text or abs(text_gate) > 1.0e-6 or learned_text_gate)
            )
            if should_use_source_semantic:
                if self.mod.is_continuous_source_memory_type(source_content_memory_type):
                    if args.source_semantic_feature_path:
                        source_semantic_features = self.mod.load_source_semantic_feature_tensor(
                            args.source_semantic_feature_path
                        )
                        source_semantic_origin = args.source_semantic_feature_path
                    else:
                        semantic_device = (
                            str(self.device)
                            if args.source_semantic_device == "same"
                            else self.mod.normalize_device_arg(args.source_semantic_device)
                        )
                        semantic_model_name = self.mod.resolve_source_semantic_model_name_for_memory(
                            source_content_memory_type,
                            args.source_semantic_model_name_or_path,
                        )
                        source_semantic_features = self.mod.extract_source_semantic_features_online(
                            audio_path=source_audio,
                            model_name_or_path=semantic_model_name,
                            cache_dir=args.source_semantic_cache_dir,
                            local_files_only=bool(args.source_semantic_local_files_only),
                            layer=int(args.source_semantic_layer),
                            device=semantic_device,
                            dtype_name=args.source_semantic_dtype,
                            downsample_stride=int(args.source_semantic_downsample_stride),
                        )
                        source_semantic_origin = f"online:{semantic_model_name}:layer{int(args.source_semantic_layer)}"
                    expected_dim = int(
                        getattr(
                            self.model.timbre_memory_config,
                            "source_semantic_feature_dim",
                            source_semantic_features.shape[-1],
                        )
                    )
                    if int(source_semantic_features.shape[-1]) != expected_dim:
                        raise ValueError(
                            "source semantic feature dim mismatch: "
                            f"got {source_semantic_features.shape[-1]}, expected {expected_dim}"
                        )
                    source_semantic_features = source_semantic_features.unsqueeze(0).to(device=self.device)
                    gen_kwargs["source_semantic_features"] = source_semantic_features
                    gen_kwargs["source_semantic_features_mask"] = torch.ones(
                        source_semantic_features.shape[:2],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    if self.mod.source_codec_residual_memory_enabled(self.model):
                        source_codes_for_memory = self.mod.add_source_ref_codes_for_memory(
                            gen_kwargs,
                            source_codes=source_codes,
                            device=self.device,
                        )
                        print(
                            "[persistent-infer] source codec residual memory "
                            f"codes={tuple(source_codes_for_memory.shape)} "
                            f"weight={float(getattr(self.model.timbre_memory_config, 'source_codec_residual_memory_weight', 0.0)):.4g}",
                            flush=True,
                        )
                    print(
                        "[persistent-infer] source semantic memory "
                        f"type={source_content_memory_type} features={tuple(source_semantic_features.shape)} "
                        f"origin={source_semantic_origin}",
                        flush=True,
                    )
                elif source_content_memory_type in {"text_tokens", "semantic_units"}:
                    if args.source_content_token_ids_path:
                        content_ids = self.mod.load_content_token_ids(args.source_content_token_ids_path)
                        token_origin = args.source_content_token_ids_path
                    elif args.source_content_token_ids:
                        content_ids = self.mod.parse_content_token_ids(args.source_content_token_ids)
                        token_origin = "cli_ids"
                    elif source_content_memory_type == "text_tokens" and source_content_text:
                        content_ids = self.mod.encode_source_content_text(
                            source_content_text,
                            args.source_content_spm_model,
                        )
                        token_origin = f"text:{args.source_content_spm_model}"
                    else:
                        raise ValueError(
                            f"source_content_memory_type={source_content_memory_type} needs content ids/text"
                        )
                    content_ids = content_ids.unsqueeze(0).to(device=self.device)
                    content_mask = torch.ones(content_ids.shape, dtype=torch.bool, device=self.device)
                    if source_content_memory_type == "text_tokens":
                        gen_kwargs["content_token_ids"] = content_ids
                        gen_kwargs["content_token_ids_mask"] = content_mask
                    else:
                        gen_kwargs["source_semantic_units"] = content_ids
                        gen_kwargs["source_semantic_units_mask"] = content_mask
                    print(
                        "[persistent-infer] source content memory "
                        f"type={source_content_memory_type} tokens={int(content_ids.shape[1])} "
                        f"origin={token_origin}",
                        flush=True,
                    )
                elif source_content_memory_type == "codec_bottleneck":
                    source_codes_for_memory = self.mod.add_source_ref_codes_for_memory(
                        gen_kwargs,
                        source_codes=source_codes,
                        device=self.device,
                    )
                    print(
                        "[persistent-infer] source codec bottleneck memory "
                        f"codes={tuple(source_codes_for_memory.shape)}",
                        flush=True,
                    )
                else:
                    raise ValueError(f"Unsupported source_content_memory_type: {source_content_memory_type}")
            gen_kwargs["source_semantic_progress_clock"] = str(args.source_semantic_progress_clock)
            gen_kwargs["source_semantic_release_after_progress"] = bool(args.source_semantic_release_after_progress)
            gen_kwargs["source_semantic_release_start"] = float(args.source_semantic_release_start)
            if min_new_tokens > 0:
                gen_kwargs["min_new_tokens"] = min_new_tokens
            if min_audio_tokens > 0:
                gen_kwargs["min_audio_tokens"] = min_audio_tokens

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        generation_stats = self.mod.generation_structure_stats(
            output,
            processor=self.processor,
            config=self.model.config,
            n_vq=self.n_vq,
            min_audio_tokens=min_audio_tokens,
        )
        if args.debug_generation_structure:
            self.mod.print_generation_structure(
                output,
                processor=self.processor,
                config=self.model.config,
                n_vq=self.n_vq,
                min_audio_tokens=min_audio_tokens,
            )
        messages = self.processor.decode(output)
        wavs = []
        for message in messages:
            if message is None:
                continue
            for cur_wav in message.to_dict().get("audio_codes_list", []):
                if torch.is_tensor(cur_wav):
                    wavs.append(cur_wav)
        if not wavs:
            raise RuntimeError("No audio waveform decoded from model output.")
        if len(wavs) > 1:
            lengths = [int(item.reshape(-1).numel()) for item in wavs]
            print(
                "[persistent-infer] decoded segments "
                f"count={len(wavs)} lengths={lengths} policy={args.audio_segment_policy}",
                flush=True,
            )
            if args.audio_segment_policy == "first":
                wavs = [wavs[0]]
            elif args.audio_segment_policy == "longest":
                wavs = [wavs[max(range(len(wavs)), key=lambda idx: lengths[idx])]]
        wav = torch.cat([item.reshape(-1).cpu() for item in wavs], dim=0)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_wav), wav.view(1, -1), int(self.processor.model_config.sampling_rate))
        print(f"[persistent-infer] wrote {output_wav.resolve()}", flush=True)
        del inputs, output, wavs, wav
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        first_stats = generation_stats[0] if generation_stats else {}
        return {
            "generation_max_new_tokens": int(gen_kwargs.get("max_new_tokens", 0)),
            "generation_min_new_tokens": int(min_new_tokens),
            "generation_min_audio_tokens": int(min_audio_tokens),
            "generation_structure": first_stats,
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run SeedTTS validation inference with one persistent model load per shard."
    )
    ap.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION_JSONL))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--manifest-jsonl", default="")
    ap.add_argument("--mode", choices=("all", "no_text", "text"), default="all")
    ap.add_argument("--per-mode", type=int, default=0)
    ap.add_argument("--per-cell", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--case-id", action="append", default=[])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--config", default=env_str("CONFIG", str(DEFAULT_CONFIG)))
    ap.add_argument("--base-model-path", default=env_str("BASE_MODEL_PATH", str(DEFAULT_BASE_MODEL)))
    ap.add_argument("--device", default=env_str("DEVICE", "cuda:0"))
    ap.add_argument("--seed", type=int, default=env_int("SEED", None))
    ap.add_argument("--n-vq", type=int, default=env_int("N_VQ", None))
    ap.add_argument("--max-new-tokens", type=int, default=env_int("MAX_NEW_TOKENS", None))
    ap.add_argument("--min-new-tokens", type=int, default=env_int("MIN_NEW_TOKENS", None))
    ap.add_argument("--min-audio-tokens", type=int, default=env_int("MIN_AUDIO_TOKENS", None))
    ap.add_argument("--text-auto-max-new-tokens", action=argparse.BooleanOptionalAction, default=env_bool("TEXT_AUTO_MAX_NEW_TOKENS", True))
    ap.add_argument("--text-cjk-chars-per-second", type=float, default=env_float("TEXT_CJK_CHARS_PER_SECOND", 5.2))
    ap.add_argument("--text-latin-words-per-second", type=float, default=env_float("TEXT_LATIN_WORDS_PER_SECOND", 2.8))
    ap.add_argument("--text-duration-margin", type=float, default=env_float("TEXT_DURATION_MARGIN", 1.15))
    ap.add_argument("--text-extra-new-tokens", type=int, default=env_int("TEXT_EXTRA_NEW_TOKENS", 48))
    ap.add_argument("--text-min-new-tokens-floor", type=int, default=env_int("TEXT_MIN_NEW_TOKENS_FLOOR", 96))
    ap.add_argument("--no-text-placeholder", default=env_str("NO_TEXT_PLACEHOLDER", "<NO_TEXT>"))
    ap.add_argument("--no-text-max-token-margin", type=int, default=env_int("NO_TEXT_MAX_TOKEN_MARGIN", 0))
    ap.add_argument("--no-text-duration-budget-ratio", type=float, default=env_float("NO_TEXT_DURATION_BUDGET_RATIO", 1.0))
    ap.add_argument("--no-text-soft-duration-budget", action="store_true", default=env_bool("NO_TEXT_SOFT_DURATION_BUDGET", False))
    ap.add_argument("--no-text-soft-min-audio-ratio", type=float, default=env_float("NO_TEXT_SOFT_MIN_AUDIO_RATIO", 0.5))
    ap.add_argument("--no-text-soft-extra-token-margin", type=int, default=env_int("NO_TEXT_SOFT_EXTRA_TOKEN_MARGIN", None))
    ap.add_argument("--temperature", type=float, default=env_float("TEMPERATURE", None))
    ap.add_argument("--top-p", type=float, default=env_float("TOP_P", None))
    ap.add_argument("--top-k", type=int, default=env_int("TOP_K", None))
    ap.add_argument("--audio-temperature", type=float, default=env_float("AUDIO_TEMPERATURE", None))
    ap.add_argument("--audio-top-p", type=float, default=env_float("AUDIO_TOP_P", None))
    ap.add_argument("--audio-top-k", type=int, default=env_int("AUDIO_TOP_K", None))
    ap.add_argument("--audio-repetition-penalty", type=float, default=env_float("AUDIO_REPETITION_PENALTY", None))
    ap.add_argument("--no-text-audio-temperature", type=float, default=env_float("NO_TEXT_AUDIO_TEMPERATURE", None))
    ap.add_argument("--no-text-audio-top-p", type=float, default=env_float("NO_TEXT_AUDIO_TOP_P", None))
    ap.add_argument("--no-text-audio-top-k", type=int, default=env_int("NO_TEXT_AUDIO_TOP_K", None))
    ap.add_argument("--no-text-audio-repetition-penalty", type=float, default=env_float("NO_TEXT_AUDIO_REPETITION_PENALTY", None))
    ap.add_argument("--source-gate-floor", type=float, default=env_float("NO_TEXT_SOURCE_GATE_FLOOR", None))
    ap.add_argument("--disable-timbre-memory", action="store_true", default=env_bool("DISABLE_TIMBRE_MEMORY", False))
    ap.add_argument("--timbre-side-only", action=argparse.BooleanOptionalAction, default=env_bool_or_none("TIMBRE_SIDE_ONLY"))
    ap.add_argument("--speaker-encoder-type", default=env_str("SPEAKER_ENCODER_TYPE", "speechbrain_ecapa"))
    ap.add_argument("--speaker-encoder-path", default=env_str("SPEAKER_ENCODER_PATH", str(DEFAULT_SPEAKER_ENCODER)))
    ap.add_argument("--speaker-embedding-dim", type=int, default=env_int("SPEAKER_EMBEDDING_DIM", 192))
    ap.add_argument("--timbre-ref-speaker-embedding-path", default=env_str("TIMBRE_REF_SPEAKER_EMBEDDING_PATH", ""))
    ap.add_argument("--language", default=env_str("LANGUAGE", None))
    ap.add_argument("--instruction", default=env_str("INSTRUCTION", None))
    ap.add_argument("--disable-mode-token", action="store_true", default=env_bool("DISABLE_MODE_TOKEN", False))
    ap.add_argument("--audio-segment-policy", choices=("all", "first", "longest"), default=env_str("AUDIO_SEGMENT_POLICY", "all"))
    ap.add_argument("--disable-source-semantic-memory", action="store_true", default=env_bool("DISABLE_SOURCE_SEMANTIC_MEMORY", False))
    ap.add_argument("--source-semantic-feature-path", default=env_str("SOURCE_SEMANTIC_FEATURE_PATH", ""))
    ap.add_argument("--source-content-token-ids", default=env_str("SOURCE_CONTENT_TOKEN_IDS", ""))
    ap.add_argument("--source-content-token-ids-path", default=env_str("SOURCE_CONTENT_TOKEN_IDS_PATH", ""))
    ap.add_argument("--source-content-spm-model", default=env_str("SOURCE_CONTENT_SPM_MODEL", str(DEFAULT_SOURCE_CONTENT_SPM)))
    ap.add_argument("--source-semantic-model-name-or-path", default=env_str("SOURCE_SEMANTIC_MODEL_NAME_OR_PATH", ""))
    ap.add_argument("--source-semantic-cache-dir", default=env_str("SOURCE_SEMANTIC_CACHE_DIR", ""))
    ap.add_argument("--source-semantic-local-files-only", action=argparse.BooleanOptionalAction, default=env_bool("SOURCE_SEMANTIC_LOCAL_FILES_ONLY", True))
    ap.add_argument("--source-semantic-layer", type=int, default=env_int("SOURCE_SEMANTIC_LAYER", 9))
    ap.add_argument("--source-semantic-device", default=env_str("SOURCE_SEMANTIC_DEVICE", "same"))
    ap.add_argument("--source-semantic-dtype", choices=("auto", "float32", "float16", "bfloat16"), default=env_str("SOURCE_SEMANTIC_DTYPE", "auto"))
    ap.add_argument("--source-semantic-downsample-stride", type=int, default=env_int("SOURCE_SEMANTIC_DOWNSAMPLE_STRIDE", 1))
    ap.add_argument("--source-semantic-position-scale", type=float, default=env_float("SOURCE_SEMANTIC_POSITION_SCALE", None))
    ap.add_argument("--source-semantic-monotonic-bias-strength", type=float, default=env_float("SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH", None))
    ap.add_argument("--source-semantic-monotonic-bias-width", type=float, default=env_float("SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH", None))
    ap.add_argument("--disable-source-semantic-monotonic-bias", action="store_true", default=env_bool("DISABLE_SOURCE_SEMANTIC_MONOTONIC_BIAS", False))
    ap.add_argument("--source-semantic-progress-clock", choices=("decode_step", "gen_slot"), default=env_str("SOURCE_SEMANTIC_PROGRESS_CLOCK", "decode_step"))
    ap.add_argument("--source-semantic-release-after-progress", action="store_true", default=env_bool("SOURCE_SEMANTIC_RELEASE_AFTER_PROGRESS", False))
    ap.add_argument("--source-semantic-release-start", type=float, default=env_float("SOURCE_SEMANTIC_RELEASE_START", 1.0))
    ap.add_argument("--debug-generation-structure", action="store_true", default=env_bool("DEBUG_GENERATION_STRUCTURE", False))
    args = ap.parse_args()
    if args.num_shards < 1:
        ap.error("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        ap.error("--shard-index must be in [0, --num-shards)")
    if args.timbre_side_only is None:
        args.timbre_side_only = checkpoint_timbre_side_only(args.model_path)
    if not args.source_semantic_model_name_or_path:
        args.source_semantic_model_name_or_path = str(load_infer_module().DEFAULT_HUBERT_MODEL)
    if not args.source_semantic_cache_dir:
        args.source_semantic_cache_dir = str(load_infer_module().DEFAULT_HUBERT_CACHE_DIR)
    return args


def main() -> int:
    args = parse_args()
    validation_jsonl = Path(args.validation_jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    manifest = Path(args.manifest_jsonl).expanduser() if args.manifest_jsonl else output_dir / "manifest.jsonl"
    rows = list(iter_jsonl(validation_jsonl))
    selected = select_rows(rows, args)
    if not selected:
        print("[persistent-valid] no rows selected", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[persistent-valid] selected={len(selected)} shard={args.shard_index}/{args.num_shards} "
        f"output_dir={output_dir} manifest={manifest}",
        flush=True,
    )
    for row in selected:
        print(
            f"[persistent-valid] selected case_id={row.get('case_id')} "
            f"mode={row.get('mode')} cell={row.get('cell')}",
            flush=True,
        )
    if args.dry_run:
        return 0

    engine = PersistentCodecVCInfer(args)
    failures = 0
    for row in selected:
        case_id = str(row.get("case_id") or "")
        output_wav = output_dir / f"{safe_stem(case_id)}.wav"
        content_text, content_text_key = source_content_text_with_key(row)
        manifest_row = {
            "case_id": case_id,
            "mode": row.get("mode"),
            "cell": row.get("cell"),
            "source_audio": row.get("source_audio"),
            "timbre_ref_audio": row.get("timbre_ref_audio"),
            "text": row.get("text"),
            "content_ref_text": row.get("content_ref_text"),
            "source_content_text": content_text,
            "source_content_text_key": content_text_key,
            "content_ref_text_source": row.get("content_ref_text_source") or row.get("eval_text_source"),
            "content_asr_backend": row.get("content_asr_backend"),
            "content_asr_model": row.get("content_asr_model"),
            "source_asr_backend": row.get("source_asr_backend"),
            "source_asr_model": row.get("source_asr_model"),
            "output_wav": str(output_wav),
            "seed": args.seed,
            "source_semantic_monotonic_bias_strength": (
                0.0 if args.disable_source_semantic_monotonic_bias else args.source_semantic_monotonic_bias_strength
            ),
            "source_semantic_monotonic_bias_width": args.source_semantic_monotonic_bias_width,
            "source_semantic_progress_clock": args.source_semantic_progress_clock,
            "source_semantic_release_after_progress": bool(args.source_semantic_release_after_progress),
            "source_semantic_release_start": args.source_semantic_release_start,
            "no_text_soft_duration_budget": bool(args.no_text_soft_duration_budget),
            "no_text_soft_min_audio_ratio": args.no_text_soft_min_audio_ratio,
            "no_text_soft_extra_token_margin": args.no_text_soft_extra_token_margin,
        }
        if output_wav.exists() and not args.overwrite:
            manifest_row.update({"status": "skipped_exists", "elapsed_sec": 0.0})
            append_jsonl(manifest, manifest_row)
            print(f"[persistent-valid] skip existing {case_id} -> {output_wav}", flush=True)
            continue
        start = time.time()
        try:
            print(f"[persistent-valid] run {case_id} mode={row.get('mode')}", flush=True)
            run_stats = engine.run_case(row, output_wav)
            elapsed = round(time.time() - start, 3)
            manifest_row.update(
                {
                    "status": "ok" if output_wav.exists() else "failed",
                    "returncode": 0 if output_wav.exists() else 1,
                    "elapsed_sec": elapsed,
                    "output_exists": output_wav.exists(),
                }
            )
            manifest_row.update(run_stats)
            print(f"[persistent-valid] done {case_id} status={manifest_row['status']} elapsed={elapsed}s", flush=True)
        except Exception as exc:
            elapsed = round(time.time() - start, 3)
            failures += 1
            manifest_row.update(
                {
                    "status": "failed",
                    "returncode": 1,
                    "elapsed_sec": elapsed,
                    "output_exists": output_wav.exists(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[persistent-valid] failed {case_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                append_jsonl(manifest, manifest_row)
                break
        append_jsonl(manifest, manifest_row)
    print(f"[persistent-valid] complete total={len(selected)} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
