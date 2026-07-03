#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOWNLOAD_ROOT = Path("/inspire/ssd/project/embodied-multimodality/public/xyzhang/download")
DEFAULT_HUBERT_CACHE_DIR = DOWNLOAD_ROOT / "huggingface"


def _first_hf_snapshot(model_cache_dir: Path) -> Path | None:
    snapshots = model_cache_dir / "snapshots"
    if not snapshots.exists():
        return None
    for path in sorted(snapshots.iterdir()):
        if (path / "config.json").exists():
            return path
    return None


DEFAULT_HUBERT_MODEL = (
    DEFAULT_HUBERT_CACHE_DIR
    / "models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a"
)
if not (DEFAULT_HUBERT_MODEL / "config.json").exists():
    DEFAULT_HUBERT_MODEL = Path("facebook/hubert-base-ls960")
DEFAULT_WAVLM_MODEL = _first_hf_snapshot(DEFAULT_HUBERT_CACHE_DIR / "models--microsoft--wavlm-base-plus")
if DEFAULT_WAVLM_MODEL is None:
    DEFAULT_WAVLM_MODEL = Path("microsoft/wavlm-base-plus")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from moss_codecvc.config import deep_get, load_config
from moss_codecvc.models import MossCodecVCTimbreMemoryWrapper
from moss_codecvc.models.moss_codecvc_wrapper import (
    SOURCE_CONTINUOUS_MEMORY_TYPES,
    normalize_source_content_memory_type,
)
from moss_codecvc.moss_codec import ensure_moss_on_path
from moss_codecvc.modes import VC_MODE_NO_TEXT, VC_MODE_TEXT, VC_NO_TEXT_PLACEHOLDER, apply_vc_mode_token
from moss_codecvc.roles import SOURCE_CODEC, count_roles, infer_prompt_role_ids_from_audio_spans


def patch_torchaudio_load_with_soundfile_fallback() -> None:
    original_load = torchaudio.load
    original_save = torchaudio.save

    def safe_load(path, *args, **kwargs):
        try:
            return original_load(path, *args, **kwargs)
        except RuntimeError as exc:
            msg = str(exc)
            if "torchcodec" not in msg and "FFmpeg" not in msg and "libtorchcodec" not in msg:
                raise
            import numpy as np
            import soundfile as sf

            wav, sr = sf.read(path, always_2d=True, dtype="float32")
            wav = torch.from_numpy(np.asarray(wav).T.copy())
            return wav, int(sr)

    def safe_save(path, src, sample_rate, *args, **kwargs):
        try:
            return original_save(path, src, sample_rate, *args, **kwargs)
        except RuntimeError as exc:
            msg = str(exc)
            if "torchcodec" not in msg and "FFmpeg" not in msg and "libtorchcodec" not in msg:
                raise
            import numpy as np
            import soundfile as sf

            wav = src.detach().cpu()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            wav = np.asarray(wav.transpose(0, 1).contiguous(), dtype="float32")
            sf.write(path, wav, int(sample_rate))

    torchaudio.load = safe_load
    torchaudio.save = safe_save


def normalize_device_arg(device_arg: str) -> str:
    if device_arg == "cuda":
        return "cuda:0"
    return device_arg


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def estimate_text_max_new_tokens(
    text: str | None,
    *,
    n_vq: int,
    source_codec_len: int,
    source_audio_seconds: float | None,
    config_ceiling: int,
    cjk_chars_per_second: float = 5.2,
    latin_words_per_second: float = 2.8,
    duration_margin: float = 1.15,
    extra_tokens: int = 48,
    min_tokens: int = 96,
) -> tuple[int, str]:
    """Estimate a text-mode generation budget from target lexical content.

    This is intentionally an upper bound, not a duration target. It prevents the
    model from having a huge 2048-token tail while still scaling with longer text.
    """

    raw = text or ""
    cjk_chars = sum(1 for ch in raw if _is_cjk_char(ch))
    latin_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", raw))
    short_punct = sum(1 for ch in raw if ch in "，、,;；:：")
    sentence_punct = sum(1 for ch in raw if ch in "。！？.!?")

    # Infer codec frame rate from the source prompt if possible. Fall back to
    # MOSS codec's usual low-frame-rate range when metadata is unavailable.
    if source_audio_seconds and source_audio_seconds > 0:
        codec_fps = max(1.0, float(source_codec_len) / float(source_audio_seconds))
    else:
        codec_fps = 25.0

    cjk_seconds = float(cjk_chars) / max(float(cjk_chars_per_second), 1.0e-3)
    latin_seconds = float(latin_words) / max(float(latin_words_per_second), 1.0e-3)
    pause_seconds = 0.12 * float(short_punct) + 0.25 * float(sentence_punct)
    estimated_seconds = max(0.8, cjk_seconds + latin_seconds + pause_seconds + 0.25)
    audio_tokens = estimated_seconds * codec_fps
    budget = int(math.ceil(audio_tokens * float(duration_margin))) + int(n_vq) + int(extra_tokens)
    budget = max(int(min_tokens), budget)
    if config_ceiling > 0:
        budget = min(int(config_ceiling), budget)
    reason = (
        "text_auto("
        f"cjk={cjk_chars},latin_words={latin_words},punct={short_punct + sentence_punct},"
        f"est_sec={estimated_seconds:.2f},codec_fps={codec_fps:.2f},"
        f"margin={duration_margin:.2f},extra={int(extra_tokens)}"
        ")"
    )
    return budget, reason


def audio_duration_seconds(path: str) -> float | None:
    try:
        info = torchaudio.info(path)
        if info.sample_rate and info.num_frames:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def _semantic_torch_dtype(name: str, device: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if str(device).startswith("cuda"):
        return torch.float16
    return torch.float32


def _processor_sample_rate(processor) -> int:
    value = getattr(processor, "sampling_rate", None)
    if value:
        return int(value)
    feature_extractor = getattr(processor, "feature_extractor", None)
    value = getattr(feature_extractor, "sampling_rate", None)
    if value:
        return int(value)
    return 16000


def _read_audio_mono_for_semantic(path: str, target_sr: int) -> torch.Tensor:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(wav).mean(dim=1)
    if int(sr) == int(target_sr):
        return audio
    try:
        import torchaudio.functional as AF

        return AF.resample(audio, orig_freq=int(sr), new_freq=int(target_sr))
    except Exception:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(sr) // gcd
        return torch.from_numpy(resample_poly(audio.numpy(), up, down).astype("float32"))


def load_source_semantic_feature_tensor(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = None
        for key in (
            "source_semantic_features",
            "semantic_features",
            "source_asr_bnf_features",
            "asr_bnf_features",
            "bnf_features",
            "source_wavlm_bnf_features",
            "wavlm_bnf_features",
            "hubert_features",
            "wavlm_features",
            "features",
        ):
            value = payload.get(key)
            if torch.is_tensor(value):
                tensor = value
                break
        if tensor is None:
            raise ValueError(f"no semantic feature tensor found in {path}")
    else:
        raise ValueError(f"unsupported semantic feature payload type in {path}: {type(payload).__name__}")
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"source semantic features must be [S,D], got {tuple(tensor.shape)} from {path}")
    return tensor.detach().float().contiguous()


def is_continuous_source_memory_type(memory_type: str) -> bool:
    return normalize_source_content_memory_type(memory_type) in SOURCE_CONTINUOUS_MEMORY_TYPES


def source_memory_type_prefers_wavlm(memory_type: str) -> bool:
    normalized = normalize_source_content_memory_type(memory_type)
    return normalized in {"wavlm_bnf_continuous", "wavlm_continuous"}


def resolve_source_semantic_model_name_for_memory(memory_type: str, requested_model: str | Path) -> str:
    requested = str(requested_model)
    if source_memory_type_prefers_wavlm(memory_type) and requested == str(DEFAULT_HUBERT_MODEL):
        return str(DEFAULT_WAVLM_MODEL)
    return requested


def add_source_ref_codes_for_memory(
    gen_kwargs: dict,
    *,
    source_codes,
    device: torch.device | str,
) -> torch.Tensor:
    source_codes_for_memory = torch.as_tensor(source_codes, dtype=torch.long, device=device).unsqueeze(0)
    gen_kwargs["source_ref_codes"] = source_codes_for_memory
    gen_kwargs["source_ref_mask"] = torch.ones(
        source_codes_for_memory.shape[:2],
        dtype=torch.bool,
        device=device,
    )
    return source_codes_for_memory


def source_codec_residual_memory_enabled(model) -> bool:
    config = getattr(model, "timbre_memory_config", None)
    if config is None:
        return False
    return float(getattr(config, "source_codec_residual_memory_weight", 0.0) or 0.0) > 0.0


def parse_content_token_ids(value: str) -> torch.Tensor:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty source content token id string")
    if text.startswith("["):
        parsed = json.loads(text)
    else:
        parsed = [item for item in re.split(r"[\s,]+", text) if item]
    ids = torch.as_tensor([int(item) for item in parsed], dtype=torch.long).flatten()
    if ids.numel() == 0:
        raise ValueError("no source content token ids parsed")
    return ids


def load_content_token_ids(path: str) -> torch.Tensor:
    path_obj = Path(path).expanduser()
    if path_obj.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path_obj, map_location="cpu")
        if torch.is_tensor(payload):
            ids = payload
        elif isinstance(payload, dict):
            ids = None
            for key in (
                "content_token_ids",
                "source_text_memory_token_ids",
                "source_semantic_units",
                "source_unit_ids",
                "unit_ids",
                "ids",
            ):
                value = payload.get(key)
                if value is not None:
                    ids = value
                    break
            if ids is None:
                raise ValueError(f"No token id field found in {path_obj}")
        else:
            ids = payload
        ids = torch.as_tensor(ids, dtype=torch.long).flatten()
    else:
        raw = path_obj.read_text(encoding="utf-8").strip()
        if path_obj.suffix.lower() == ".json":
            payload = json.loads(raw)
            if isinstance(payload, dict):
                for key in (
                    "content_token_ids",
                    "source_text_memory_token_ids",
                    "source_semantic_units",
                    "source_unit_ids",
                    "unit_ids",
                    "ids",
                ):
                    if key in payload:
                        payload = payload[key]
                        break
            ids = torch.as_tensor(payload, dtype=torch.long).flatten()
        else:
            ids = parse_content_token_ids(raw)
    if ids.numel() == 0:
        raise ValueError(f"No token ids loaded from {path_obj}")
    return ids.long()


def encode_source_content_text(text: str, spm_model_path: str) -> torch.Tensor:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("sentencepiece is required for --source-content-text inference encoding") from exc
    processor = spm.SentencePieceProcessor()
    if not processor.Load(str(spm_model_path)):
        raise ValueError(f"Failed to load sentencepiece model: {spm_model_path}")
    ids = processor.EncodeAsIds(str(text))
    if not ids:
        raise ValueError("--source-content-text produced no sentencepiece ids")
    return torch.as_tensor([int(item) + 1 for item in ids], dtype=torch.long)


@torch.no_grad()
def extract_source_semantic_features_online(
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
    dtype = _semantic_torch_dtype(dtype_name, device)
    model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype, use_safetensors=False, **common)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False

    sr = _processor_sample_rate(processor)
    audio = _read_audio_mono_for_semantic(audio_path, sr)
    inputs = processor(audio.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    model_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if key == "input_values":
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device=device)
        model_inputs[key] = value
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
    del model
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features.contiguous()


def prepare_peft_adapter_dir_for_inference(model_path: Path) -> Path:
    adapter_file = model_path / "adapter_model.safetensors"
    adapter_config = model_path / "adapter_config.json"
    if not adapter_file.exists() or not adapter_config.exists():
        return model_path

    from safetensors.torch import safe_open, save_file

    with safe_open(adapter_file, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        needs_normalization = any("._fsdp_wrapped_module." in key for key in keys)
        if not needs_normalization:
            return model_path
        normalized_dir = model_path / ".peft_infer_normalized"
        normalized_file = normalized_dir / "adapter_model.safetensors"
        if normalized_file.exists() and normalized_file.stat().st_mtime >= adapter_file.stat().st_mtime:
            print(f"[infer] using normalized FSDP-wrapped PEFT adapter keys from {normalized_dir}")
            return normalized_dir
        state = {
            key.replace("._fsdp_wrapped_module.", "."): handle.get_tensor(key).contiguous()
            for key in keys
        }

    normalized_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(normalized_file))
    shutil.copy2(adapter_config, normalized_dir / "adapter_config.json")
    readme = model_path / "README.md"
    if readme.exists():
        shutil.copy2(readme, normalized_dir / "README.md")
    print(f"[infer] normalized FSDP-wrapped PEFT adapter keys -> {normalized_dir}")
    return normalized_dir


def apply_source_gate_floor_for_inference(model, source_gate_floor: float | None) -> None:
    if source_gate_floor is None:
        return
    floor = float(source_gate_floor)
    if floor <= 0.0:
        return
    if floor >= 1.0:
        floor = 0.999
    role_router = getattr(model, "role_router", None)
    if role_router is None or not hasattr(role_router, "gate_logits"):
        print("[infer] source gate floor requested, but this model has no role_router; skipping.")
        return
    with torch.no_grad():
        logits = role_router.gate_logits
        gates = torch.sigmoid(logits.float()).to(device=logits.device)
        before = gates[SOURCE_CODEC].detach().float().clone()
        gates[SOURCE_CODEC] = torch.clamp(gates[SOURCE_CODEC], min=floor)
        updated_logits = torch.logit(gates.clamp(min=1.0e-4, max=1.0 - 1.0e-4)).to(dtype=logits.dtype)
        logits.copy_(updated_logits)
        after = torch.sigmoid(logits.float())[SOURCE_CODEC].detach().float()
    print(
        "[infer] source role gate floor applied: "
        f"floor={floor:.3f} "
        f"before_mean={before.mean().item():.4f} before_min={before.min().item():.4f} "
        f"after_mean={after.mean().item():.4f} after_min={after.min().item():.4f}"
    )


def apply_source_semantic_debug_overrides(
    model,
    *,
    position_scale: float | None,
    monotonic_bias_strength: float | None,
    monotonic_bias_width: float | None,
) -> None:
    encoder = getattr(model, "source_semantic_memory_encoder", None)
    residual_encoder = getattr(model, "source_semantic_codec_residual_encoder", None)
    adapters = getattr(model, "source_semantic_layer_adapters", None)
    if encoder is None and residual_encoder is None and adapters is None:
        if any(value is not None for value in (position_scale, monotonic_bias_strength, monotonic_bias_width)):
            print("[infer] source semantic override requested, but model has no SourceSemanticMemory; skipping.")
        return
    if position_scale is not None and encoder is not None:
        encoder.position_scale = float(position_scale)
    if position_scale is not None and residual_encoder is not None:
        residual_encoder.position_scale = float(position_scale)
    if position_scale is not None and (encoder is not None or residual_encoder is not None):
        if hasattr(model, "timbre_memory_config"):
            model.timbre_memory_config.source_semantic_position_scale = float(position_scale)
        print(f"[infer] source semantic position_scale override={float(position_scale):.4g}")
    if adapters is None:
        return
    for adapter in adapters.values():
        if monotonic_bias_strength is not None:
            adapter.monotonic_bias_strength = float(monotonic_bias_strength)
        if monotonic_bias_width is not None:
            adapter.monotonic_bias_width = max(1.0e-4, float(monotonic_bias_width))
    if hasattr(model, "timbre_memory_config"):
        if monotonic_bias_strength is not None:
            model.timbre_memory_config.source_semantic_monotonic_bias_strength = float(monotonic_bias_strength)
        if monotonic_bias_width is not None:
            model.timbre_memory_config.source_semantic_monotonic_bias_width = max(1.0e-4, float(monotonic_bias_width))
    if monotonic_bias_strength is not None or monotonic_bias_width is not None:
        first = next(iter(adapters.values()), None)
        if first is not None:
            print(
                "[infer] source semantic monotonic bias override: "
                f"strength={float(first.monotonic_bias_strength):.4g} "
                f"width={float(first.monotonic_bias_width):.4g}"
            )


def set_generation_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    print(f"[infer] generation seed={int(seed)}")


def generation_structure_stats(output, *, processor, config, n_vq: int, min_audio_tokens: int = 0) -> list[dict[str, object]]:
    """Return non-invasive diagnostics for delayed codec generation structure."""
    def _positions(text_codes: torch.Tensor, token_id: int) -> list[int]:
        return torch.nonzero(text_codes == int(token_id), as_tuple=False).reshape(-1).detach().cpu().tolist()

    rows: list[dict[str, object]] = []
    for item_idx, (start_length, generation_ids) in enumerate(output):
        text_codes = generation_ids[:, 0].detach().cpu()
        audio_codes = generation_ids[:, 1:].detach()
        delayed_nonpad = (audio_codes != int(config.audio_pad_code)).any(dim=1)
        try:
            dedelayed = processor.apply_de_delay_pattern(audio_codes)
            dedelayed_nonpad = (dedelayed != int(config.audio_pad_code)).any(dim=1)
            idx = torch.nonzero(dedelayed_nonpad, as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                segment_lengths = []
            else:
                breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
                if breaks.numel() == 0:
                    segment_lengths = [int(idx.numel())]
                else:
                    segment_lengths = [int(seg.numel()) for seg in torch.split(idx, breaks.tolist())]
        except Exception as exc:  # pragma: no cover - debug path only
            segment_lengths = [f"de_delay_failed:{type(exc).__name__}:{exc}"]

        audio_start_positions = _positions(text_codes, int(config.audio_start_token_id))
        audio_end_positions = _positions(text_codes, int(config.audio_end_token_id))
        im_end_positions = _positions(text_codes, int(config.im_end_token_id))
        delay_positions = _positions(text_codes, int(config.audio_assistant_delay_slot_token_id))
        gen_slot_count = int((text_codes == int(config.audio_assistant_gen_slot_token_id)).sum().item())
        delay_slot_count = int((text_codes == int(config.audio_assistant_delay_slot_token_id)).sum().item())
        pad_count = int((text_codes == int(config.pad_token_id)).sum().item())
        delayed_nonpad_count = int(delayed_nonpad.sum().item())
        first_delay_pos = delay_positions[0] if delay_positions else None
        rows.append(
            {
                "item": item_idx,
                "rows": int(generation_ids.shape[0]),
                "start_length": int(start_length),
                "n_vq": int(n_vq),
                "audio_start_positions": audio_start_positions,
                "audio_end_positions": audio_end_positions,
                "im_end_positions": im_end_positions,
                "first_delay_pos": first_delay_pos,
                "delay_minus_min_audio": None if first_delay_pos is None else int(first_delay_pos) - int(min_audio_tokens),
                "gen_slot_count": gen_slot_count,
                "delay_slot_count": delay_slot_count,
                "text_pad_count": pad_count,
                "delayed_audio_nonpad_rows": delayed_nonpad_count,
                "dedelayed_segment_lengths": segment_lengths,
            }
        )
    return rows


def print_generation_structure(output, *, processor, config, n_vq: int, min_audio_tokens: int = 0) -> None:
    """Print non-invasive diagnostics for delayed codec generation structure."""
    for row in generation_structure_stats(
        output,
        processor=processor,
        config=config,
        n_vq=n_vq,
        min_audio_tokens=min_audio_tokens,
    ):
        print(
            "[infer][debug_generation] "
            f"item={row['item']} rows={row['rows']} start_length={row['start_length']} "
            f"n_vq={row['n_vq']} audio_start_pos={row['audio_start_positions']} "
            f"audio_end_pos={row['audio_end_positions']} im_end_pos={row['im_end_positions']} "
            f"first_delay_pos={row['first_delay_pos']} delay_minus_min_audio={row['delay_minus_min_audio']} "
            f"gen_slot_count={row['gen_slot_count']} delay_slot_count={row['delay_slot_count']} "
            f"text_pad_count={row['text_pad_count']} delayed_audio_nonpad_rows={row['delayed_audio_nonpad_rows']} "
            f"dedelayed_segment_lengths={row['dedelayed_segment_lengths']}"
        )


def extract_dedelayed_audio_segments(output, *, processor, config) -> list[torch.Tensor]:
    segments: list[torch.Tensor] = []
    for _start_length, generation_ids in output:
        audio_codes = generation_ids[:, 1:].detach()
        dedelayed = processor.apply_de_delay_pattern(audio_codes)
        nonpad = (dedelayed != int(config.audio_pad_code)).any(dim=1)
        idx = torch.nonzero(nonpad, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
        segment_indices = [idx] if breaks.numel() == 0 else list(torch.split(idx, breaks.tolist()))
        for cur_idx in segment_indices:
            segments.append(dedelayed[cur_idx].detach().cpu().long().contiguous())
    return segments


def select_codec_segments(segments: list[torch.Tensor], *, policy: str) -> torch.Tensor | None:
    if not segments:
        return None
    if len(segments) == 1:
        return segments[0]
    if policy == "first":
        return segments[0]
    if policy == "longest":
        return max(segments, key=lambda item: int(item.shape[0]))
    return torch.cat(segments, dim=0)


def save_source_semantic_attention_debug(model, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = list(getattr(model, "last_source_semantic_attention_maps", []) or [])
    summary: dict[str, object] = {
        "map_count": len(maps),
        "captured_tokens": int(getattr(model, "_source_semantic_attention_captured_tokens", 0)),
        "layers": {},
    }
    torch.save({"maps": maps}, output_dir / "source_semantic_attention_maps.pt")
    if not maps:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"[infer] no source semantic attention maps captured -> {output_dir}")
        return

    by_layer: dict[int, list[torch.Tensor]] = {}
    valid_by_layer: dict[int, list[int]] = {}
    for item in maps:
        layer_idx = int(item["layer_idx"])
        by_layer.setdefault(layer_idx, []).append(item["attention"].float())
        valid_by_layer.setdefault(layer_idx, []).append(int(item.get("source_valid_tokens", item["attention"].shape[-1])))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Image = None
    except Exception as exc:  # pragma: no cover - optional visualization dependency
        plt = None
        try:
            from PIL import Image
        except Exception as pil_exc:  # pragma: no cover - optional visualization dependency
            Image = None
            print(f"[infer] matplotlib/PIL unavailable, only saving tensors/json: {exc}; {pil_exc}")
        else:
            print(f"[infer] matplotlib unavailable, using PIL PNG fallback: {exc}")

    layer_summary: dict[str, object] = {}
    for layer_idx, tensors in sorted(by_layer.items()):
        attention = torch.cat(tensors, dim=0).float()
        source_valid_tokens = max(valid_by_layer.get(layer_idx, [attention.shape[-1]]))
        attention = attention[:, :source_valid_tokens]
        layer_key = f"layer_{layer_idx}"
        torch.save(attention.cpu(), output_dir / f"{layer_key}_attention.pt")

        source_positions = torch.linspace(0.0, 1.0, attention.shape[-1])
        expected_pos = (attention.cpu() * source_positions.view(1, -1)).sum(dim=-1)
        target_pos = torch.linspace(0.0, 1.0, attention.shape[0])
        if target_pos.numel() > 1:
            centered_t = target_pos - target_pos.mean()
            centered_e = expected_pos - expected_pos.mean()
            slope = float((centered_t * centered_e).mean().div(centered_t.pow(2).mean().clamp_min(1.0e-6)).item())
        else:
            slope = 0.0
        peak = attention.max(dim=-1).values
        entropy_probs = attention.clamp_min(1.0e-9)
        entropy = -(entropy_probs * entropy_probs.log()).sum(dim=-1) / math.log(max(2, attention.shape[-1]))
        n_tokens = int(expected_pos.numel())
        first = max(1, n_tokens // 3)
        second = max(first + 1, 2 * n_tokens // 3) if n_tokens > 1 else first
        layer_summary[layer_key] = {
            "shape": list(attention.shape),
            "source_valid_tokens": int(source_valid_tokens),
            "expected_pos_begin": float(expected_pos[:first].mean().item()),
            "expected_pos_mid": float(expected_pos[first:second].mean().item()) if second > first else 0.0,
            "expected_pos_end": float(expected_pos[second:].mean().item()) if n_tokens > second else 0.0,
            "expected_pos_slope": slope,
            "peak_mean": float(peak.mean().item()),
            "entropy_mean": float(entropy.mean().item()),
        }
        if plt is not None:
            fig_w = max(6.0, min(18.0, attention.shape[-1] / 24.0))
            fig_h = max(3.5, min(10.0, attention.shape[0] / 32.0))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
            im = ax.imshow(attention.cpu().numpy(), aspect="auto", origin="lower", interpolation="nearest")
            ax.set_title(f"SourceSemantic attention {layer_key}")
            ax.set_xlabel("source semantic frame")
            ax.set_ylabel("target generated token")
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            fig.tight_layout()
            fig.savefig(output_dir / f"{layer_key}_attention.png")
            plt.close(fig)
        elif Image is not None:
            array = attention.cpu()
            array = array / array.max().clamp_min(1.0e-9)
            image = Image.fromarray((array.numpy() * 255.0).clip(0, 255).astype("uint8"), mode="L")
            scale_x = max(1, min(4, 1024 // max(1, image.size[0])))
            scale_y = max(1, min(4, 512 // max(1, image.size[1])))
            if scale_x > 1 or scale_y > 1:
                image = image.resize((image.size[0] * scale_x, image.size[1] * scale_y), resample=Image.Resampling.NEAREST)
            image.save(output_dir / f"{layer_key}_attention.png")

    summary["layers"] = layer_summary
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"[infer] wrote source semantic attention debug -> {output_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model-path", default=None, help="Required when --model-path points to a LoRA adapter dir without a resolvable base model.")
    ap.add_argument("--source-audio", required=True)
    ap.add_argument("--timbre-ref-audio", required=True)
    ap.add_argument(
        "--timbre-ref-speaker-embedding-path",
        default=None,
        help="Optional precomputed frozen speaker embedding for S2/E_ref used by Ver1.6 TTE conditioning.",
    )
    ap.add_argument(
        "--speaker-encoder-type",
        choices=("embedding_loader", "precomputed_ecapa", "speechbrain_ecapa", "seed_tts_eval_ecapa"),
        default=None,
        help="Optional Ver1.6 inference override. Use speechbrain_ecapa for a lightweight ECAPA sidecar.",
    )
    ap.add_argument("--speaker-encoder-path", default=None, help="Speaker encoder source/local directory for online inference.")
    ap.add_argument("--speaker-embedding-dim", type=int, default=None, help="Must match the dimension used during training.")
    ap.add_argument("--text", default=None, help="Target lexical content for MODE=text/text_prosody.")
    ap.add_argument("--no-text", action="store_true", help="Run VC generation with the no-text placeholder used during training.")
    ap.add_argument("--no-text-placeholder", default=VC_NO_TEXT_PLACEHOLDER)
    ap.add_argument("--language", default=None)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--disable-mode-token", action="store_true")
    ap.add_argument(
        "--disable-timbre-memory",
        action="store_true",
        help="Do not load Ver1.6 timbre memory adapter even if the adapter directory contains it.",
    )
    ap.add_argument(
        "--timbre-side-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not include timbre_ref_audio codec in the AR prompt; use it only through timbre memory/speaker conditioning.",
    )
    ap.add_argument("--output-wav", required=True)
    ap.add_argument(
        "--output-generated-codec",
        default="",
        help="Optional .pt path for dedelayed generated codec tokens [T,32], selected by --audio-segment-policy.",
    )
    ap.add_argument(
        "--output-codec-jsonl",
        default="",
        help="Optional one-row JSONL with source/ref/generated codec tokens for visualization scripts.",
    )
    ap.add_argument(
        "--audio-segment-policy",
        choices=("all", "first", "longest"),
        default="all",
        help="How to handle multiple decoded audio segments. Text mode often wants 'first' to drop accidental tails.",
    )
    ap.add_argument("--n-vq", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=None, help="Optional torch generation seed for reproducible sampling.")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--min-new-tokens", type=int, default=None)
    ap.add_argument(
        "--text-auto-max-new-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For text mode, estimate max_new_tokens from TEXT when --max-new-tokens is omitted.",
    )
    ap.add_argument("--text-cjk-chars-per-second", type=float, default=5.2)
    ap.add_argument("--text-latin-words-per-second", type=float, default=2.8)
    ap.add_argument("--text-duration-margin", type=float, default=1.15)
    ap.add_argument("--text-extra-new-tokens", type=int, default=48)
    ap.add_argument("--text-min-new-tokens-floor", type=int, default=96)
    ap.add_argument(
        "--min-audio-tokens",
        type=int,
        default=None,
        help=(
            "Ver2 wrapper generation guard. In no-text VC, prevent early delay/audio_end "
            "until this many audio gen slots have been produced. Defaults to source codec length."
        ),
    )
    ap.add_argument(
        "--no-text-max-token-margin",
        type=int,
        default=0,
        help=(
            "When --no-text is used and --max-new-tokens is omitted, generate "
            "up to source codec length plus n_vq delay steps plus this tail margin."
        ),
    )
    ap.add_argument(
        "--no-text-duration-budget-ratio",
        type=float,
        default=1.0,
        help=(
            "Scale no_text max_new_tokens from source codec length before adding n_vq and margin. "
            "Use 1.15-1.25 for a looser duration budget, or 1.0 for strict source length."
        ),
    )
    ap.add_argument(
        "--no-text-soft-duration-budget",
        action="store_true",
        help=(
            "Opt-in no-text inference budget relaxation. Keeps the legacy strict budget disabled by default; "
            "when enabled, min_new_tokens is decoupled from max_new_tokens, min_audio_tokens is reduced by "
            "--no-text-soft-min-audio-ratio, and extra tail room is added for delay flush."
        ),
    )
    ap.add_argument("--no-text-soft-min-audio-ratio", type=float, default=0.5)
    ap.add_argument(
        "--no-text-soft-extra-token-margin",
        type=int,
        default=None,
        help="Extra no-text max_new_tokens margin used only with --no-text-soft-duration-budget. Defaults to n_vq.",
    )
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--audio-temperature", type=float, default=None)
    ap.add_argument("--audio-top-p", type=float, default=None)
    ap.add_argument("--audio-top-k", type=int, default=None)
    ap.add_argument("--audio-repetition-penalty", type=float, default=None)
    ap.add_argument(
        "--source-gate-floor",
        type=float,
        default=None,
        help=(
            "Inference-only Ver2 safety knob. Raise SOURCE_CODEC role-routing "
            "gates to at least this value to preserve source content in no-text VC."
        ),
    )
    ap.add_argument(
        "--disable-source-semantic-memory",
        action="store_true",
        help="Do not compute or pass Ver2.5 SourceSemanticMemory features during inference.",
    )
    ap.add_argument(
        "--source-semantic-feature-path",
        default="",
        help=(
            "Optional precomputed HuBERT/WavLM/ASR-BNF feature .pt for source wav. "
            "If omitted, continuous-memory inference extracts the configured SSL model online."
        ),
    )
    ap.add_argument(
        "--source-content-token-ids",
        default="",
        help="Comma/space separated content token ids for Ver2.5 text_tokens or semantic_units memory.",
    )
    ap.add_argument(
        "--source-content-token-ids-path",
        default="",
        help="Path to .txt/.json/.pt token ids for Ver2.5 text_tokens or semantic_units memory.",
    )
    ap.add_argument(
        "--source-content-text",
        default="",
        help="Source transcript to encode with --source-content-spm-model for text_tokens memory.",
    )
    ap.add_argument(
        "--source-content-spm-model",
        default=str(ROOT / "trainset/shared_content_tokenizers/spm_multilingual_byte_fallback_v1.model"),
        help="SentencePiece model used to encode --source-content-text. ids get +1 offset to match training.",
    )
    ap.add_argument(
        "--source-semantic-model-name-or-path",
        default=str(DEFAULT_HUBERT_MODEL),
        help=(
            "Online source semantic extractor. Defaults to HuBERT for hubert_continuous; "
            "wavlm_* memory types auto-switch this default to microsoft/wavlm-base-plus."
        ),
    )
    ap.add_argument("--source-semantic-cache-dir", default=str(DEFAULT_HUBERT_CACHE_DIR))
    ap.add_argument(
        "--source-semantic-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local HuggingFace cache for online source semantic extraction.",
    )
    ap.add_argument("--source-semantic-layer", type=int, default=9)
    ap.add_argument(
        "--source-semantic-device",
        default="same",
        help="Device for online HuBERT extraction. 'same' uses the inference device.",
    )
    ap.add_argument(
        "--source-semantic-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    ap.add_argument("--source-semantic-downsample-stride", type=int, default=1)
    ap.add_argument(
        "--source-semantic-attention-debug-dir",
        default="",
        help="Optional directory to save Ver2.5 SourceSemanticMemory attention tensors and PNG heatmaps.",
    )
    ap.add_argument(
        "--source-semantic-attention-debug-max-tokens",
        type=int,
        default=2048,
        help="Maximum target tokens to keep when exporting source semantic attention debug maps.",
    )
    ap.add_argument(
        "--source-semantic-position-scale",
        type=float,
        default=None,
        help="Inference-only override for SourceSemanticMemory sinusoidal position scale.",
    )
    ap.add_argument(
        "--source-semantic-monotonic-bias-strength",
        type=float,
        default=None,
        help="Inference-only override for SourceSemanticAdapter monotonic attention bias strength.",
    )
    ap.add_argument(
        "--source-semantic-monotonic-bias-width",
        type=float,
        default=None,
        help="Inference-only override for SourceSemanticAdapter monotonic attention bias width.",
    )
    ap.add_argument(
        "--disable-source-semantic-monotonic-bias",
        action="store_true",
        help="Inference-only shorthand for --source-semantic-monotonic-bias-strength 0.0.",
    )
    ap.add_argument(
        "--source-semantic-progress-clock",
        choices=("decode_step", "gen_slot"),
        default="decode_step",
        help="Clock used for source semantic monotonic progress during generation. Default preserves legacy behavior.",
    )
    ap.add_argument(
        "--source-semantic-release-after-progress",
        action="store_true",
        help="Opt-in: fade/release source semantic monotonic bias when progress reaches the end instead of pinning source tail.",
    )
    ap.add_argument("--source-semantic-release-start", type=float, default=1.0)
    ap.add_argument(
        "--debug-generation-structure",
        action="store_true",
        help="Print generated delayed-codec text/audio segment diagnostics before decoding.",
    )
    args = ap.parse_args()
    set_generation_seed(args.seed)

    cfg = load_config(args.config)
    moss_root = deep_get(cfg, "moss.root")
    codec_path = deep_get(cfg, "moss.codec_path")
    n_vq = args.n_vq or int(deep_get(cfg, "moss.default_n_vq", 32))
    ensure_moss_on_path(moss_root)
    patch_torchaudio_load_with_soundfile_fallback()

    from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
    from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor
    from peft import PeftConfig, PeftModel

    device_arg = normalize_device_arg(args.device)
    device = torch.device(device_arg if torch.cuda.is_available() or not device_arg.startswith("cuda") else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_path = Path(args.model_path)
    is_lora_adapter = (model_path / "adapter_config.json").exists()
    peft_model_path = prepare_peft_adapter_dir_for_inference(model_path) if is_lora_adapter else model_path

    if is_lora_adapter:
        peft_cfg = PeftConfig.from_pretrained(str(peft_model_path))
        base_model_path = args.base_model_path or peft_cfg.base_model_name_or_path
        if not base_model_path:
            raise ValueError("--base-model-path is required when adapter_config.json does not specify a base model path.")
        processor_path = base_model_path
        model_load_path = base_model_path
    else:
        processor_path = args.model_path
        model_load_path = args.model_path

    processor = MossTTSDelayProcessor.from_pretrained(
        processor_path,
        codec_path=codec_path,
        trust_remote_code=True,
    )
    processor.audio_tokenizer.to(device)
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
        print(f"[infer] loading PEFT adapter from {peft_model_path}")
        model = PeftModel.from_pretrained(model, str(peft_model_path))
    use_timbre_memory = (
        is_lora_adapter
        and not args.disable_timbre_memory
        and (model_path / "timbre_memory_config.json").exists()
        and (model_path / "timbre_memory_adapter.pt").exists()
    )
    if use_timbre_memory:
        timbre_memory_overrides = {}
        if args.speaker_encoder_type:
            timbre_memory_overrides["speaker_encoder_type"] = args.speaker_encoder_type
        if args.speaker_encoder_path:
            timbre_memory_overrides["speaker_encoder_path"] = args.speaker_encoder_path
        if args.speaker_embedding_dim is not None:
            timbre_memory_overrides["speaker_embedding_dim"] = int(args.speaker_embedding_dim)
        model = MossCodecVCTimbreMemoryWrapper.from_pretrained_timbre_memory(
            model,
            model_path,
            map_location="cpu",
            config_overrides=timbre_memory_overrides or None,
        )
    if args.timbre_side_only and not use_timbre_memory:
        raise ValueError("--timbre-side-only requires an adapter with timbre memory; otherwise timbre conditioning is removed.")
    model = model.to(device).eval()
    apply_source_gate_floor_for_inference(model, args.source_gate_floor)
    monotonic_bias_strength = (
        0.0 if args.disable_source_semantic_monotonic_bias else args.source_semantic_monotonic_bias_strength
    )
    apply_source_semantic_debug_overrides(
        model,
        position_scale=args.source_semantic_position_scale,
        monotonic_bias_strength=monotonic_bias_strength,
        monotonic_bias_width=args.source_semantic_monotonic_bias_width,
    )

    source_codes, timbre_codes = processor.encode_audios_from_path(
        [args.source_audio, args.timbre_ref_audio],
        n_vq=n_vq,
    )
    vc_mode = VC_MODE_NO_TEXT if args.no_text else VC_MODE_TEXT
    if args.instruction:
        instruction = args.instruction
    elif vc_mode == VC_MODE_TEXT:
        instruction = deep_get(cfg, "instruction.text_prosody") or deep_get(cfg, "instruction.default")
    else:
        instruction = (
            deep_get(cfg, "instruction.no_text")
            or deep_get(cfg, "instruction.prosody_no_timbre")
            or deep_get(cfg, "instruction.default")
        )
    instruction = apply_vc_mode_token(instruction, vc_mode, enabled=not args.disable_mode_token)
    prompt_text = args.no_text_placeholder if args.no_text else args.text
    prompt_references = [source_codes] if args.timbre_side_only else [source_codes, timbre_codes]
    user_message = processor.build_user_message(
        text=prompt_text,
        reference=prompt_references,
        instruction=instruction,
        tokens=int(source_codes.shape[0]),
        language=args.language,
        quality="high",
    )
    inputs = processor([[user_message]], mode="generation", n_vq=n_vq)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    text_temperature = args.temperature if args.temperature is not None else float(deep_get(cfg, "inference.temperature", 0.8))
    text_top_p = args.top_p if args.top_p is not None else float(deep_get(cfg, "inference.top_p", 0.9))
    text_top_k = args.top_k if args.top_k is not None else int(deep_get(cfg, "inference.top_k", 50))
    no_text_budget_frames = int(math.ceil(float(source_codes.shape[0]) * max(0.01, float(args.no_text_duration_budget_ratio))))
    no_text_base_new_tokens = no_text_budget_frames + int(n_vq)
    no_text_soft_budget = bool(args.no_text and args.no_text_soft_duration_budget)
    config_max_new_tokens = int(deep_get(cfg, "inference.max_new_tokens", 2048))
    if args.max_new_tokens is not None:
        max_new_tokens = int(args.max_new_tokens)
        max_new_tokens_reason = "explicit"
    elif args.no_text:
        token_margin = max(0, int(args.no_text_max_token_margin))
        if no_text_soft_budget:
            soft_margin = int(args.no_text_soft_extra_token_margin) if args.no_text_soft_extra_token_margin is not None else int(n_vq)
            token_margin = max(token_margin, max(0, soft_margin))
        max_new_tokens = no_text_base_new_tokens + token_margin
        max_new_tokens_reason = f"ceil(source_codec_len({int(source_codes.shape[0])})*ratio({float(args.no_text_duration_budget_ratio):.3f}))+n_vq({int(n_vq)})+margin({token_margin})"
    elif args.text_auto_max_new_tokens:
        max_new_tokens, max_new_tokens_reason = estimate_text_max_new_tokens(
            args.text,
            n_vq=n_vq,
            source_codec_len=int(source_codes.shape[0]),
            source_audio_seconds=audio_duration_seconds(args.source_audio),
            config_ceiling=config_max_new_tokens,
            cjk_chars_per_second=float(args.text_cjk_chars_per_second),
            latin_words_per_second=float(args.text_latin_words_per_second),
            duration_margin=float(args.text_duration_margin),
            extra_tokens=int(args.text_extra_new_tokens),
            min_tokens=int(args.text_min_new_tokens_floor),
        )
    else:
        max_new_tokens = config_max_new_tokens
        max_new_tokens_reason = "config"
    if args.min_new_tokens is not None:
        min_new_tokens = max(0, int(args.min_new_tokens))
        min_new_tokens_reason = "explicit"
    elif no_text_soft_budget:
        min_new_tokens = 0
        min_new_tokens_reason = "soft_duration_budget"
    elif args.no_text and args.max_new_tokens is None:
        min_new_tokens = max_new_tokens
        min_new_tokens_reason = "no_text_source_duration_guard"
    else:
        min_new_tokens = 0
        min_new_tokens_reason = "disabled"
    if args.min_audio_tokens is not None:
        min_audio_tokens = max(0, int(args.min_audio_tokens))
        min_audio_tokens_reason = "explicit"
    elif no_text_soft_budget:
        min_audio_tokens = max(0, int(math.floor(float(source_codes.shape[0]) * max(0.0, float(args.no_text_soft_min_audio_ratio)))))
        min_audio_tokens_reason = f"soft_source_codec_len_x{float(args.no_text_soft_min_audio_ratio):.3f}"
    elif args.no_text:
        min_audio_tokens = int(source_codes.shape[0])
        min_audio_tokens_reason = "source_codec_len"
    else:
        min_audio_tokens = 0
        min_audio_tokens_reason = "disabled"
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "text_temperature": text_temperature,
        "text_top_p": text_top_p,
        "text_top_k": text_top_k,
        "audio_temperature": (
            float(args.audio_temperature)
            if args.audio_temperature is not None
            else float(deep_get(cfg, "inference.audio_temperature", 1.7))
        ),
        "audio_top_p": (
            float(args.audio_top_p)
            if args.audio_top_p is not None
            else float(deep_get(cfg, "inference.audio_top_p", 0.8))
        ),
        "audio_top_k": (
            int(args.audio_top_k)
            if args.audio_top_k is not None
            else int(deep_get(cfg, "inference.audio_top_k", 25))
        ),
        "audio_repetition_penalty": (
            float(args.audio_repetition_penalty)
            if args.audio_repetition_penalty is not None
            else float(deep_get(cfg, "inference.audio_repetition_penalty", 1.0))
        ),
    }
    print(
        "[infer] encoded codec shapes: "
        f"source={tuple(source_codes.shape)} timbre_ref={tuple(timbre_codes.shape)} "
        f"prompt_refs={len(prompt_references)} timbre_side_only={bool(args.timbre_side_only)} n_vq={n_vq}"
    )
    print(f"[infer] prompt input_ids shape={tuple(inputs['input_ids'].shape)}")
    print(f"[infer] effective max_new_tokens={max_new_tokens} ({max_new_tokens_reason})")
    print(f"[infer] effective min_new_tokens={min_new_tokens} ({min_new_tokens_reason})")
    print(f"[infer] effective min_audio_tokens={min_audio_tokens} ({min_audio_tokens_reason})")
    print(
        "[infer] audio sampling: "
        f"temperature={gen_kwargs['audio_temperature']} "
        f"top_p={gen_kwargs['audio_top_p']} "
        f"top_k={gen_kwargs['audio_top_k']} "
        f"repetition_penalty={gen_kwargs['audio_repetition_penalty']}"
    )
    if use_timbre_memory:
        mode_id = int(getattr(model, "MODE_TO_ID", {}).get(vc_mode, 0))
        if mode_id > 0:
            gen_kwargs["vc_mode_id"] = torch.tensor([mode_id], dtype=torch.long, device=device)
        timbre_codes_for_memory = torch.as_tensor(timbre_codes, dtype=torch.long, device=device).unsqueeze(0)
        gen_kwargs["timbre_ref_codes"] = timbre_codes_for_memory
        gen_kwargs["timbre_ref_mask"] = torch.ones(
            timbre_codes_for_memory.shape[:2],
            dtype=torch.bool,
            device=device,
        )
        if args.timbre_ref_speaker_embedding_path:
            gen_kwargs["timbre_ref_speaker_embedding_path"] = [args.timbre_ref_speaker_embedding_path]
        gen_kwargs["timbre_ref_speaker_audio_path"] = [args.timbre_ref_audio]
        if getattr(model.timbre_memory_config, "use_role_routing", False):
            prompt_role_ids = infer_prompt_role_ids_from_audio_spans(
                inputs["input_ids"],
                audio_pad_code=int(model.config.audio_pad_code),
            )
            print(f"[infer] inferred prompt role counts={count_roles(prompt_role_ids).as_dict()}")
            print(
                "[infer] Ver2 role routing enabled: "
                f"T_ref tokens={model.timbre_memory_config.num_memory_tokens} "
                f"P_src tokens={model.timbre_memory_config.prosody_memory_tokens} "
                f"target_head_routing={model.timbre_memory_config.target_head_routing}"
            )
        else:
            print(f"[infer] Ver1.6 timbre memory enabled: T_ref tokens={model.timbre_memory_config.num_memory_tokens}")
        source_semantic_encoder = getattr(model, "source_semantic_memory_encoder", None)
        text_gate = float(getattr(model.timbre_memory_config, "source_semantic_text_gate", 0.0))
        learned_text_gate = bool(getattr(model.timbre_memory_config, "source_semantic_allow_learned_text_gate", False))
        source_content_memory_type = str(
            getattr(model.timbre_memory_config, "source_content_memory_type", "hubert_continuous") or "hubert_continuous"
        ).strip().lower()
        source_content_memory_type = normalize_source_content_memory_type(source_content_memory_type)
        should_use_source_semantic = (
            source_semantic_encoder is not None
            and not args.disable_source_semantic_memory
            and (args.no_text or abs(text_gate) > 1.0e-6 or learned_text_gate)
        )
        if should_use_source_semantic:
            if is_continuous_source_memory_type(source_content_memory_type):
                if args.source_semantic_feature_path:
                    source_semantic_features = load_source_semantic_feature_tensor(args.source_semantic_feature_path)
                    source_semantic_origin = args.source_semantic_feature_path
                else:
                    semantic_device = str(device) if args.source_semantic_device == "same" else normalize_device_arg(args.source_semantic_device)
                    semantic_model_name = resolve_source_semantic_model_name_for_memory(
                        source_content_memory_type,
                        args.source_semantic_model_name_or_path,
                    )
                    source_semantic_features = extract_source_semantic_features_online(
                        audio_path=args.source_audio,
                        model_name_or_path=semantic_model_name,
                        cache_dir=args.source_semantic_cache_dir,
                        local_files_only=bool(args.source_semantic_local_files_only),
                        layer=int(args.source_semantic_layer),
                        device=semantic_device,
                        dtype_name=args.source_semantic_dtype,
                        downsample_stride=int(args.source_semantic_downsample_stride),
                    )
                    source_semantic_origin = f"online:{semantic_model_name}:layer{int(args.source_semantic_layer)}"
                expected_dim = int(getattr(model.timbre_memory_config, "source_semantic_feature_dim", source_semantic_features.shape[-1]))
                if int(source_semantic_features.shape[-1]) != expected_dim:
                    raise ValueError(
                        "source semantic feature dim mismatch: "
                        f"got {source_semantic_features.shape[-1]}, expected {expected_dim}"
                    )
                source_semantic_features = source_semantic_features.unsqueeze(0).to(device=device)
                source_semantic_mask = torch.ones(
                    source_semantic_features.shape[:2],
                    dtype=torch.bool,
                    device=device,
                )
                gen_kwargs["source_semantic_features"] = source_semantic_features
                gen_kwargs["source_semantic_features_mask"] = source_semantic_mask
                if source_codec_residual_memory_enabled(model):
                    source_codes_for_memory = add_source_ref_codes_for_memory(
                        gen_kwargs,
                        source_codes=source_codes,
                        device=device,
                    )
                    print(
                        "[infer] source codec residual memory enabled: "
                        f"codes={tuple(source_codes_for_memory.shape)} "
                        f"weight={float(getattr(model.timbre_memory_config, 'source_codec_residual_memory_weight', 0.0)):.4g}"
                    )
                print(
                    "[infer] Ver2.5 source semantic memory enabled: "
                    f"type={source_content_memory_type} features={tuple(source_semantic_features.shape)} "
                    f"origin={source_semantic_origin} no_text={args.no_text} "
                    f"text_gate={text_gate:.4f} learned_text_gate={learned_text_gate}"
                )
            elif source_content_memory_type in {"text_tokens", "semantic_units"}:
                if args.source_content_token_ids_path:
                    content_ids = load_content_token_ids(args.source_content_token_ids_path)
                    token_origin = args.source_content_token_ids_path
                elif args.source_content_token_ids:
                    content_ids = parse_content_token_ids(args.source_content_token_ids)
                    token_origin = "cli_ids"
                elif source_content_memory_type == "text_tokens" and args.source_content_text:
                    content_ids = encode_source_content_text(args.source_content_text, args.source_content_spm_model)
                    token_origin = f"text:{args.source_content_spm_model}"
                else:
                    raise ValueError(
                        f"Ver2.5 source_content_memory_type={source_content_memory_type} requires "
                        "--source-content-token-ids-path, --source-content-token-ids, or "
                        "--source-content-text for text_tokens."
                    )
                content_ids = content_ids.unsqueeze(0).to(device=device)
                content_mask = torch.ones(content_ids.shape, dtype=torch.bool, device=device)
                if source_content_memory_type == "text_tokens":
                    gen_kwargs["content_token_ids"] = content_ids
                    gen_kwargs["content_token_ids_mask"] = content_mask
                else:
                    gen_kwargs["source_semantic_units"] = content_ids
                    gen_kwargs["source_semantic_units_mask"] = content_mask
                print(
                    "[infer] Ver2.5 source content token memory enabled: "
                    f"type={source_content_memory_type} tokens={int(content_ids.shape[1])} "
                    f"origin={token_origin} no_text={args.no_text} text_gate={text_gate:.4f}"
                )
            elif source_content_memory_type == "codec_bottleneck":
                source_codes_for_memory = add_source_ref_codes_for_memory(
                    gen_kwargs,
                    source_codes=source_codes,
                    device=device,
                )
                print(
                    "[infer] Ver2.5 source codec bottleneck memory enabled: "
                    f"codes={tuple(source_codes_for_memory.shape)} codebooks="
                    f"{getattr(model.timbre_memory_config, 'source_content_codec_codebooks', 'first_4')}"
                )
            else:
                raise ValueError(f"Unsupported source_content_memory_type at inference: {source_content_memory_type}")
        elif source_semantic_encoder is not None:
            reason = "disabled" if args.disable_source_semantic_memory else "text_gate_zero"
            print(f"[infer] Ver2.5 source semantic memory present but not used: {reason} type={source_content_memory_type}")
        gen_kwargs["source_semantic_progress_clock"] = str(args.source_semantic_progress_clock)
        gen_kwargs["source_semantic_release_after_progress"] = bool(args.source_semantic_release_after_progress)
        gen_kwargs["source_semantic_release_start"] = float(args.source_semantic_release_start)
        if min_new_tokens > 0:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        if min_audio_tokens > 0:
            gen_kwargs["min_audio_tokens"] = min_audio_tokens
    if args.source_semantic_attention_debug_dir and hasattr(model, "clear_source_semantic_attention_capture"):
        model.clear_source_semantic_attention_capture()
        model.capture_source_semantic_attention = True
        model.source_semantic_attention_capture_max_tokens = int(args.source_semantic_attention_debug_max_tokens)
    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)
    if args.source_semantic_attention_debug_dir:
        if hasattr(model, "capture_source_semantic_attention"):
            model.capture_source_semantic_attention = False
        save_source_semantic_attention_debug(model, args.source_semantic_attention_debug_dir)
    if args.debug_generation_structure:
        print_generation_structure(output, processor=processor, config=model.config, n_vq=n_vq, min_audio_tokens=min_audio_tokens)
    generated_codec = None
    if args.output_generated_codec or args.output_codec_jsonl:
        codec_segments = extract_dedelayed_audio_segments(output, processor=processor, config=model.config)
        generated_codec = select_codec_segments(codec_segments, policy=args.audio_segment_policy)
        if generated_codec is None:
            print("[infer] no generated codec segment available to save.")
        else:
            if args.output_generated_codec:
                codec_path_out = Path(args.output_generated_codec)
                codec_path_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "generated_codec": generated_codec,
                        "source_codec": torch.as_tensor(source_codes, dtype=torch.long).cpu(),
                        "ref_codec": torch.as_tensor(timbre_codes, dtype=torch.long).cpu(),
                        "n_vq": int(n_vq),
                        "audio_segment_policy": args.audio_segment_policy,
                    },
                    codec_path_out,
                )
                print(f"[infer] wrote generated codec -> {codec_path_out}")
            if args.output_codec_jsonl:
                codec_jsonl_out = Path(args.output_codec_jsonl)
                codec_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "sample_id": Path(args.output_wav).stem,
                    "moss_codecvc_mode": VC_MODE_NO_TEXT if args.no_text else VC_MODE_TEXT,
                    "text": prompt_text,
                    "source_audio": args.source_audio,
                    "timbre_ref_audio": args.timbre_ref_audio,
                    "output_wav": args.output_wav,
                    "reference_audio_codes": [
                        torch.as_tensor(source_codes, dtype=torch.long).cpu().tolist(),
                        torch.as_tensor(timbre_codes, dtype=torch.long).cpu().tolist(),
                    ],
                    "audio_codes": generated_codec.cpu().tolist(),
                    "audio_codes_source": "generated",
                    "n_vq": int(n_vq),
                }
                with codec_jsonl_out.open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[infer] wrote codec visualization jsonl -> {codec_jsonl_out}")
    messages = processor.decode(output)
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
        print(f"[infer] decoded_audio_segments={len(wavs)} sample_lengths={lengths} policy={args.audio_segment_policy}")
        if args.audio_segment_policy == "first":
            wavs = [wavs[0]]
        elif args.audio_segment_policy == "longest":
            wavs = [wavs[max(range(len(wavs)), key=lambda idx: lengths[idx])]]
    wav = torch.cat([w.reshape(-1).cpu() for w in wavs], dim=0)
    out_path = Path(args.output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torchaudio

    torchaudio.save(str(out_path), wav.view(1, -1), int(processor.model_config.sampling_rate))
    print(f"wrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
