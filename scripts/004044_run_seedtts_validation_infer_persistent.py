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

import numpy as np
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
    return bool(
        config.get("timbre_side_only", False)
        or config.get("speaker_side_pathway_enabled", False)
        or config.get("speaker_cross_attn_enabled", False)
    )


def checkpoint_speaker_side_pathway(model_path: str) -> bool:
    config_path = Path(model_path).expanduser() / "timbre_memory_config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return False
    return bool(config.get("speaker_side_pathway_enabled", False) or config.get("speaker_cross_attn_enabled", False))


def checkpoint_progress_stop_enabled(model_path: str) -> bool:
    config_path = Path(model_path).expanduser() / "timbre_memory_config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return False
    return float(config.get("progress_loss_weight", 0.0) or 0.0) > 0.0 or float(
        config.get("stop_loss_weight", 0.0) or 0.0
    ) > 0.0


def load_speaker_seq_tensor(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise FileNotFoundError(f"speaker_seq_path not found: {path_obj}")
    if path_obj.suffix.lower() == ".npz":
        payload = np.load(path_obj)
        if "speaker_seq_features" in payload.files:
            arr = payload["speaker_seq_features"]
        elif "features" in payload.files:
            arr = payload["features"]
        else:
            arr = payload[payload.files[0]]
    else:
        arr = np.load(path_obj)
    tensor = torch.as_tensor(arr, dtype=torch.float32, device=device)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.dim() != 2:
        raise ValueError(f"speaker_seq_path must contain [T, D], got {tuple(tensor.shape)} from {path_obj}")
    mask = torch.ones((1, tensor.shape[0]), dtype=torch.bool, device=device)
    return tensor.unsqueeze(0), mask


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
        if (
            args.filter_v2_real_no_text_ref_content_leak
            and mode == "no_text"
            and is_v2_real_no_text_ref_content_leak(row)
        ):
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


def nested_get(row: dict[str, Any], key: str) -> Any | None:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_content_text(text: Any) -> str:
    out: list[str] = []
    for ch in str(text or "").lower():
        code = ord(ch)
        if ch.isalnum() or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            out.append(ch)
    return "".join(out)


def is_v2_real_no_text_ref_content_leak(row: dict[str, Any]) -> bool:
    ref_text = normalize_content_text(nested_get(row, "timbre_ref_text"))
    target_text = normalize_content_text(nested_get(row, "target_text"))
    return bool(ref_text and target_text and ref_text == target_text)


def audio_path_with_meta_fallback(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value:
        return str(value)
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict) and meta.get(key):
        return str(meta[key])
    return ""


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
        self._source_semantic_online_cache: dict[tuple[str, str, bool, str, str], tuple[Any, torch.nn.Module, torch.dtype, int]] = {}
        cfg = self.mod.load_config(args.config)
        self.cfg = cfg
        self.n_vq = args.n_vq or int(self.mod.deep_get(cfg, "moss.default_n_vq", 32))
        self.mod.ensure_moss_on_path(self.mod.deep_get(cfg, "moss.root"))
        self.mod.patch_torchaudio_load_with_soundfile_fallback()

        from moss_tts_delay.configuration_moss_tts import MossTTSDelayConfig
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
        model_load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if args.attn_implementation:
            attn_impl = str(args.attn_implementation)
            model_config = MossTTSDelayConfig.from_pretrained(model_load_path, trust_remote_code=True)
            for cfg_obj in (model_config, getattr(model_config, "language_config", None)):
                if cfg_obj is None:
                    continue
                setattr(cfg_obj, "_attn_implementation", attn_impl)
                setattr(cfg_obj, "_attn_implementation_internal", attn_impl)
                setattr(cfg_obj, "attn_implementation", attn_impl)
            model_load_kwargs["config"] = model_config
            model_load_kwargs["attn_implementation"] = attn_impl
            print(
                f"[persistent-infer] loading base model with attn_implementation={args.attn_implementation}",
                flush=True,
            )
        model = MossTTSDelayModel.from_pretrained(model_load_path, **model_load_kwargs)
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
        timbre_cfg = getattr(self.model, "timbre_memory_config", None)
        self.stop_head_budget = bool(
            getattr(self.model, "progress_stop_head", None) is not None
            and timbre_cfg is not None
            and (
                float(getattr(timbre_cfg, "progress_loss_weight", 0.0) or 0.0) > 0.0
                or float(getattr(timbre_cfg, "stop_loss_weight", 0.0) or 0.0) > 0.0
            )
        )
        if self.stop_head_budget:
            print("[persistent-infer] stop-head budget enabled; min_audio_tokens will not force no-text length", flush=True)
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

    @torch.no_grad()
    def _extract_source_semantic_features_online_cached(
        self,
        *,
        audio_path: str,
        model_name_or_path: str,
        cache_dir: str,
        local_files_only: bool,
        layer: int,
        device: str,
        dtype_name: str,
        downsample_stride: int,
    ) -> torch.Tensor:
        key = (
            str(model_name_or_path),
            str(Path(cache_dir).expanduser()),
            bool(local_files_only),
            str(device),
            str(dtype_name),
        )
        cached = self._source_semantic_online_cache.get(key)
        if cached is None:
            from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

            common = {
                "cache_dir": str(Path(cache_dir).expanduser()),
                "local_files_only": bool(local_files_only),
                "trust_remote_code": True,
            }
            try:
                processor = AutoProcessor.from_pretrained(model_name_or_path, **common)
            except Exception:
                processor = AutoFeatureExtractor.from_pretrained(model_name_or_path, **common)
            dtype = self.mod._semantic_torch_dtype(dtype_name, device)
            model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype, use_safetensors=False, **common)
            model.eval().to(device)
            for param in model.parameters():
                param.requires_grad = False
            sr = self.mod._processor_sample_rate(processor)
            cached = (processor, model, dtype, int(sr))
            self._source_semantic_online_cache[key] = cached
            print(
                "[persistent-infer] cached source semantic extractor "
                f"model={model_name_or_path} device={device} dtype={dtype} sr={int(sr)}",
                flush=True,
            )
        processor, model, dtype, sr = cached
        audio = self.mod._read_audio_mono_for_semantic(audio_path, sr)
        inputs = processor(audio.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
        model_inputs = {}
        for name, value in inputs.items():
            if torch.is_tensor(value):
                if name == "input_values":
                    value = value.to(device=device, dtype=dtype)
                else:
                    value = value.to(device=device)
            model_inputs[name] = value
        outputs = model(**model_inputs, output_hidden_states=True)
        if int(layer) == -1 or getattr(outputs, "hidden_states", None) is None:
            features = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states
            idx = int(layer)
            if idx < 0:
                idx = len(hidden_states) + idx
            if idx < 0 or idx >= len(hidden_states):
                raise ValueError(f"source semantic layer {layer} outside hidden_states={len(hidden_states)}")
            features = hidden_states[idx]
        features = features.squeeze(0).detach().float().cpu()
        stride = max(1, int(downsample_stride))
        if stride > 1:
            features = features[::stride].contiguous()
        if features.dim() != 2 or features.numel() == 0:
            raise RuntimeError(f"empty source semantic features for {audio_path}: shape={tuple(features.shape)}")
        return features.contiguous()

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
        stop_head_budget_active = bool(self.stop_head_budget and self.use_timbre_memory)
        if stop_head_budget_active:
            if args.min_new_tokens is None:
                min_new_tokens = 0
            if args.min_audio_tokens is None:
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
            f"stop_head_budget={int(stop_head_budget_active)} "
            f"audio_top_k={gen_kwargs['audio_top_k']} "
            f"timbre_cfg_scale={float(args.timbre_cfg_scale):.3f}",
            flush=True,
        )
        return gen_kwargs, min_new_tokens, min_audio_tokens

    def run_case(self, row: dict[str, Any], output_wav: Path) -> dict[str, Any]:
        args = self.args
        mode = str(row.get("mode") or "")
        no_text = mode == "no_text"
        source_audio = audio_path_with_meta_fallback(row, "source_audio")
        timbre_ref_audio = audio_path_with_meta_fallback(row, "timbre_ref_audio")
        speaker_vec_path = str(row.get("speaker_vec_path") or row.get("timbre_ref_speaker_vec_path") or "")
        if args.speaker_vec_path:
            speaker_vec_path = str(args.speaker_vec_path)
        speaker_seq_path = str(
            row.get("speaker_seq_path")
            or row.get("timbre_ref_speaker_seq_path")
            or row.get("speaker_seq_features_path")
            or ""
        )
        if args.speaker_seq_path:
            speaker_seq_path = str(args.speaker_seq_path)
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
        ref_prompt_slot_enabled, ref_prompt_slot_tokens = (
            self.mod.resolve_ref_speaker_prompt_slot(args, self.model) if self.use_timbre_memory else (False, 0)
        )
        ref_prompt_permutation = (
            self.mod.resolve_ref_prompt_codec_permutation(args, self.model)
            if self.use_timbre_memory
            else {
                "enabled": False,
                "min_seconds": 2.0,
                "max_seconds": 4.0,
                "frame_rate": 12.5,
                "seed": 1234,
                "mode": "shuffle",
                "block_seconds": 0.4,
                "bootstrap": "off",
            }
        )
        ref_prompt_slot_codes = None
        if ref_prompt_slot_enabled:
            ref_prompt_slot_codes = self.mod.make_ref_speaker_prompt_slot_codes(
                source_codes,
                n_vq=self.n_vq,
                audio_pad_code=int(self.model.config.audio_pad_code),
                token_count=ref_prompt_slot_tokens,
                slot_code=int(getattr(self.model.timbre_memory_config, "ref_speaker_prompt_slot_code", -1)),
                slot_pack_mode=str(getattr(self.model.timbre_memory_config, "ref_speaker_prompt_slot_pack_mode", "pad")),
            )
        timbre_prompt_codes = timbre_codes
        ref_prompt_permutation_stats = {
            "enabled": 0,
            "source_frames": int(timbre_codes.shape[0]),
            "prompt_frames": int(timbre_codes.shape[0]),
            "start": 0,
            "shuffled": 0,
        }
        should_permute_ref_prompt = (
            bool(ref_prompt_permutation["enabled"])
            and ref_prompt_slot_codes is None
            and not bool(args.timbre_side_only)
        )
        if should_permute_ref_prompt:
            timbre_prompt_codes, stats = self.mod.permute_ref_prompt_codes(
                timbre_codes,
                enabled=True,
                min_seconds=float(ref_prompt_permutation["min_seconds"]),
                max_seconds=float(ref_prompt_permutation["max_seconds"]),
                frame_rate=float(ref_prompt_permutation["frame_rate"]),
                seed=int(ref_prompt_permutation["seed"]),
                mode=str(ref_prompt_permutation["mode"]),
                block_seconds=float(ref_prompt_permutation["block_seconds"]),
                bootstrap=str(ref_prompt_permutation.get("bootstrap", "off")),
            )
            ref_prompt_permutation_stats = stats.as_dict()
        if ref_prompt_slot_codes is not None:
            prompt_references = [source_codes, ref_prompt_slot_codes]
        else:
            prompt_references = [source_codes] if args.timbre_side_only else [source_codes, timbre_prompt_codes]
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
            f"timbre_side_only={bool(args.timbre_side_only)} ref_prompt_slot={bool(ref_prompt_slot_enabled)} "
            f"ref_prompt_permutation_enabled={bool(ref_prompt_permutation['enabled'])} "
            f"ref_prompt_permutation_applied={bool(should_permute_ref_prompt)} "
            f"ref_prompt_permutation_mode={ref_prompt_permutation['mode']} "
            f"ref_prompt_permutation_block_seconds={float(ref_prompt_permutation['block_seconds']):.3f} "
            f"ref_prompt_permutation_stats={ref_prompt_permutation_stats} "
            f"prompt={tuple(inputs['input_ids'].shape)}",
            flush=True,
        )
        if self.use_timbre_memory:
            gen_kwargs["timbre_cfg_scale"] = float(args.timbre_cfg_scale)
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
            if speaker_vec_path:
                gen_kwargs["speaker_vec_path"] = [speaker_vec_path]
            cross_source = str(
                getattr(self.model.timbre_memory_config, "speaker_cross_attn_source", "vector") or "vector"
            ).strip().lower()
            if cross_source == "sequence" and not speaker_seq_path:
                raise ValueError(
                    f"checkpoint expects speaker_cross_attn_source=sequence but row has no speaker_seq_path: "
                    f"{row.get('case_id')}"
                )
            if speaker_seq_path:
                speaker_seq_features, speaker_seq_mask = load_speaker_seq_tensor(speaker_seq_path, self.device)
                gen_kwargs["speaker_seq_features"] = speaker_seq_features
                gen_kwargs["speaker_seq_features_mask"] = speaker_seq_mask
            gen_kwargs["timbre_ref_speaker_audio_path"] = [timbre_ref_audio]
            if int(args.ref_speaker_prompt_attention_capture_frames) > 0:
                gen_kwargs["ref_speaker_prompt_attention_capture_frames"] = int(
                    args.ref_speaker_prompt_attention_capture_frames
                )
                gen_kwargs["ref_speaker_prompt_attention_layers"] = str(args.ref_speaker_prompt_attention_layers)
            if getattr(self.model.timbre_memory_config, "use_role_routing", False):
                prompt_role_ids = self.mod.infer_prompt_role_ids_from_audio_spans(
                    inputs["input_ids"],
                    audio_pad_code=int(self.model.config.audio_pad_code),
                )
                if ref_prompt_slot_enabled:
                    slot_positions = self.mod.ref_speaker_prompt_slot_positions(
                        inputs["input_ids"],
                        audio_start_token_id=int(self.model.config.audio_start_token_id),
                        audio_end_token_id=int(self.model.config.audio_end_token_id),
                        audio_gen_slot_token_id=(
                            int(getattr(self.model.config, "audio_user_slot_token_id", self.model.config.audio_assistant_gen_slot_token_id)),
                            int(self.model.config.audio_assistant_gen_slot_token_id),
                        ),
                        token_count=int(ref_prompt_slot_tokens),
                        occurrence=2,
                    )
                    prompt_role_ids = prompt_role_ids.clone()
                    prompt_role_ids[slot_positions.to(device=prompt_role_ids.device).bool()] = self.mod.REF_CODEC
                    gen_kwargs["ref_speaker_prompt_slot_positions"] = slot_positions.to(self.device)
                    print(
                        "[persistent-infer] ref speaker prompt slot "
                        f"tokens={ref_prompt_slot_tokens} positions={int(slot_positions.sum().item())}",
                        flush=True,
                    )
                gen_kwargs["role_ids"] = prompt_role_ids.to(self.device)
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
                        source_semantic_features = self._extract_source_semantic_features_online_cached(
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
        generation_ids_path = ""
        if args.save_generation_ids_dir:
            generation_ids_dir = Path(args.save_generation_ids_dir).expanduser()
            generation_ids_dir.mkdir(parents=True, exist_ok=True)
            generation_ids_path = str(generation_ids_dir / f"{safe_stem(str(row.get('case_id') or 'case'))}.pt")
            torch.save(
                {
                    "case_id": row.get("case_id"),
                    "mode": row.get("mode"),
                    "cell": row.get("cell"),
                    "timbre_cfg_scale": float(args.timbre_cfg_scale),
                    "output": [
                        (int(start_length), cur_generation_ids.detach().cpu())
                        for start_length, cur_generation_ids in output
                    ],
                },
                generation_ids_path,
            )
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
        result = {
            "generation_max_new_tokens": int(gen_kwargs.get("max_new_tokens", 0)),
            "generation_min_new_tokens": int(min_new_tokens),
            "generation_min_audio_tokens": int(min_audio_tokens),
            "generation_stop_head_budget": bool(self.stop_head_budget),
            "generation_structure": first_stats,
            "timbre_cfg_scale": float(args.timbre_cfg_scale),
            "speaker_vec_path": speaker_vec_path,
            "speaker_seq_path": speaker_seq_path,
            "ref_prompt_codec_permutation": ref_prompt_permutation_stats,
            "ref_prompt_codec_permutation_applied": bool(should_permute_ref_prompt),
        }
        if generation_ids_path:
            result["generation_ids_path"] = generation_ids_path
        ref_slot_attn = getattr(self.model, "last_ref_speaker_prompt_attention_stats", {}) or {}
        if ref_slot_attn:
            result["ref_speaker_prompt_attention_stats"] = ref_slot_attn
        ref_slot_embed = getattr(self.model, "last_ref_speaker_prompt_slot_stats", {}) or {}
        if ref_slot_embed:
            result["ref_speaker_prompt_slot_stats"] = ref_slot_embed
        progress_stop_infer = getattr(self.model, "last_progress_stop_infer_stats", {}) or {}
        if progress_stop_infer:
            result["progress_stop_infer_stats"] = progress_stop_infer
        return result


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
    ap.add_argument(
        "--filter-v2-real-no-text-ref-content-leak",
        action=argparse.BooleanOptionalAction,
        default=env_bool("FILTER_V2_REAL_NO_TEXT_REF_CONTENT_LEAK", True),
        help="Skip no-text rows whose timbre_ref_text normalizes exactly to target_text.",
    )
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
    ap.add_argument("--ref-prompt-codec-permutation", action=argparse.BooleanOptionalAction, default=env_bool_or_none("REF_PROMPT_CODEC_PERMUTATION"))
    ap.add_argument("--ref-prompt-codec-permutation-min-seconds", type=float, default=env_float("REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS", None))
    ap.add_argument("--ref-prompt-codec-permutation-max-seconds", type=float, default=env_float("REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS", None))
    ap.add_argument("--ref-prompt-codec-permutation-frame-rate", type=float, default=env_float("REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE", None))
    ap.add_argument("--ref-prompt-codec-permutation-seed", type=int, default=env_int("REF_PROMPT_CODEC_PERMUTATION_SEED", None))
    ap.add_argument("--ref-prompt-codec-permutation-mode", choices=("shuffle", "contiguous", "block_shuffle"), default=env_str("REF_PROMPT_CODEC_PERMUTATION_MODE", None))
    ap.add_argument("--ref-prompt-codec-permutation-block-seconds", type=float, default=env_float("REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS", None))
    ap.add_argument("--ref-prompt-codec-permutation-bootstrap", choices=("off", "block"), default=env_str("REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP", None))
    ap.add_argument("--ref-speaker-prompt-slot", action=argparse.BooleanOptionalAction, default=env_bool_or_none("REF_SPEAKER_PROMPT_SLOT"))
    ap.add_argument("--ref-speaker-prompt-tokens", type=int, default=env_int("REF_SPEAKER_PROMPT_TOKENS", None))
    ap.add_argument("--ref-speaker-prompt-attention-capture-frames", type=int, default=env_int("REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES", 0))
    ap.add_argument("--ref-speaker-prompt-attention-layers", default=env_str("REF_SPEAKER_PROMPT_ATTENTION_LAYERS", "-4,-3,-2,-1"))
    ap.add_argument(
        "--attn-implementation",
        default=env_str("MOSS_TTS_ATTN_IMPLEMENTATION", ""),
        help="Optional HF attention backend override for diagnostics, e.g. eager for output_attentions.",
    )
    ap.add_argument("--speaker-encoder-type", default=env_str("SPEAKER_ENCODER_TYPE", ""))
    ap.add_argument("--speaker-encoder-path", default=env_str("SPEAKER_ENCODER_PATH", ""))
    ap.add_argument("--speaker-embedding-dim", type=int, default=env_int("SPEAKER_EMBEDDING_DIM", None))
    ap.add_argument("--timbre-ref-speaker-embedding-path", default=env_str("TIMBRE_REF_SPEAKER_EMBEDDING_PATH", ""))
    ap.add_argument("--speaker-vec-path", default=env_str("SPEAKER_VEC_PATH", ""))
    ap.add_argument("--speaker-seq-path", default=env_str("SPEAKER_SEQ_PATH", ""))
    ap.add_argument("--language", default=env_str("LANGUAGE", None))
    ap.add_argument("--instruction", default=env_str("INSTRUCTION", None))
    ap.add_argument("--disable-mode-token", action="store_true", default=env_bool("DISABLE_MODE_TOKEN", False))
    ap.add_argument("--audio-segment-policy", choices=("all", "first", "longest"), default=env_str("AUDIO_SEGMENT_POLICY", "all"))
    ap.add_argument("--disable-source-semantic-memory", action="store_true", default=env_bool("DISABLE_SOURCE_SEMANTIC_MEMORY", False))
    ap.add_argument(
        "--timbre-cfg-scale",
        "--cfg-scale",
        dest="timbre_cfg_scale",
        type=float,
        default=env_float("TIMBRE_CFG_SCALE", 1.0),
        help="Classifier-free guidance scale for S2 speaker conditioning.",
    )
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
    ap.add_argument("--save-generation-ids-dir", default=env_str("SAVE_GENERATION_IDS_DIR", ""))
    args = ap.parse_args()
    if args.num_shards < 1:
        ap.error("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        ap.error("--shard-index must be in [0, --num-shards)")
    if args.timbre_side_only is None:
        args.timbre_side_only = checkpoint_timbre_side_only(args.model_path)
    if not args.speaker_encoder_type and not checkpoint_speaker_side_pathway(args.model_path):
        args.speaker_encoder_type = "speechbrain_ecapa"
    if not args.speaker_encoder_path and args.speaker_encoder_type == "speechbrain_ecapa":
        args.speaker_encoder_path = str(DEFAULT_SPEAKER_ENCODER)
    if args.speaker_embedding_dim is None and args.speaker_encoder_type == "speechbrain_ecapa":
        args.speaker_embedding_dim = 192
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
        source_audio = audio_path_with_meta_fallback(row, "source_audio")
        timbre_ref_audio = audio_path_with_meta_fallback(row, "timbre_ref_audio")
        target_audio = audio_path_with_meta_fallback(row, "target_audio")
        manifest_row = {
            "case_id": case_id,
            "mode": row.get("mode"),
            "cell": row.get("cell"),
            "source_audio": source_audio,
            "timbre_ref_audio": timbre_ref_audio,
            "target_audio": target_audio,
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
