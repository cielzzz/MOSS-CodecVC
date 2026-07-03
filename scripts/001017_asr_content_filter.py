#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.modes import VC_NO_TEXT_PLACEHOLDER


TEXT_KEYS = (
    "content_ref_text",
    "source_text",
    "source_transcript",
    "source_asr_text",
    "asr_src_text",
    "transcript",
    "text",
)
TARGET_TEXT_KEYS = (
    "text",
    "target_text",
    "target_transcript",
    "normalized_text",
    "content_ref_text",
)
TARGET_ASR_KEYS = ("asr_tgt_text", "target_asr_text", "target_transcript")
SOURCE_ASR_KEYS = ("asr_src_text", "source_asr_text", "source_transcript", "transcript")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Attach ASR/CER/WER content filtering fields to MOSS-CodecVC JSONL.")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument(
        "--asr-backend",
        choices=("existing_fields", "jsonl_map", "faster_whisper", "whisper", "qwen_asr"),
        default="existing_fields",
        help="existing_fields/jsonl_map only compute metrics from already available ASR text.",
    )
    ap.add_argument("--asr-map-jsonl", default="", help="Optional JSONL with audio/path and text/asr_text fields.")
    ap.add_argument("--faster-whisper-model", default="", help="Model name/path when --asr-backend=faster_whisper.")
    ap.add_argument("--whisper-model", default="small", help="Model name when --asr-backend=whisper.")
    ap.add_argument(
        "--qwen-asr-model",
        default="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b",
        help="Local Qwen3-ASR model path when --asr-backend=qwen_asr. Requires the qwen-asr package.",
    )
    ap.add_argument("--qwen-asr-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--qwen-asr-max-batch-size", type=int, default=16)
    ap.add_argument("--qwen-asr-max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--language", default="", help="Optional ASR language hint, e.g. zh/en.")
    ap.add_argument("--zh-cer-threshold", type=float, default=0.20)
    ap.add_argument("--en-wer-threshold", type=float, default=0.25)
    ap.add_argument("--no-text-zh-cer-threshold", type=float, default=0.25)
    ap.add_argument("--no-text-en-wer-threshold", type=float, default=0.30)
    ap.add_argument("--max-repeat-score", type=float, default=0.30)
    ap.add_argument("--min-asr-chars", type=int, default=2)
    ap.add_argument("--min-duration-ratio", type=float, default=0.50)
    ap.add_argument("--max-duration-ratio", type=float, default=1.80)
    ap.add_argument(
        "--content-reference-mode",
        choices=("auto", "source", "text", "target_text"),
        default="auto",
        help=(
            "Reference text used for target ASR comparison. auto preserves old behavior: "
            "no_text uses source ASR, text modes use explicit text. text/target_text is "
            "the correct setting for text_prosody."
        ),
    )
    ap.add_argument(
        "--skip-source-asr",
        action="store_true",
        help="Do not transcribe source_audio unless source ASR is required by the chosen reference mode.",
    )
    ap.add_argument(
        "--disable-duration-ratio-check",
        action="store_true",
        help="Skip target/source duration-ratio filtering. Useful when source_audio is only a style/prosody reference.",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1, help="Total number of modulo shards for parallel ASR.")
    ap.add_argument("--shard-index", type=int, default=0, help="Current modulo shard index, in [0, num_shards).")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing shard output and skip already written shard rows.",
    )
    return ap.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_jsonl(path: Path, rows, *, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def record_get(record: dict[str, Any], key: str) -> Any:
    if key in record and record[key] not in (None, ""):
        return record[key]
    meta = record.get("moss_codecvc_meta")
    if isinstance(meta, dict) and meta.get(key) not in (None, ""):
        return meta[key]
    return None


def first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record_get(record, key)
        if isinstance(value, str):
            text = value.strip()
            if text and text not in {VC_NO_TEXT_PLACEHOLDER, "None", "none", "null"}:
                return text
    return ""


def mode_is_no_text(record: dict[str, Any]) -> bool:
    mode = str(record_get(record, "moss_codecvc_mode") or record_get(record, "mode") or "").lower()
    if mode == "no_text":
        return True
    text = str(record.get("text") or "").strip()
    return text == VC_NO_TEXT_PLACEHOLDER or not text


def normalize_for_cer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def words_for_wer(text: str) -> list[str]:
    text = text.lower()
    if re.search(r"[\u4e00-\u9fff]", text):
        return list(normalize_for_cer(text))
    text = re.sub(r"[^\w\s]+", " ", text)
    return [item for item in text.split() if item]


def edit_distance(a: list[Any], b: list[Any]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, bv in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if av == bv else 1),
            )
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    ref_chars = list(normalize_for_cer(ref))
    pred_chars = list(normalize_for_cer(pred))
    if not ref_chars:
        return math.inf
    return edit_distance(pred_chars, ref_chars) / max(1, len(ref_chars))


def wer(pred: str, ref: str) -> float:
    ref_words = words_for_wer(ref)
    pred_words = words_for_wer(pred)
    if not ref_words:
        return math.inf
    return edit_distance(pred_words, ref_words) / max(1, len(ref_words))


def is_zh(text: str, language: str = "") -> bool:
    if language.lower().startswith("zh"):
        return True
    if language.lower().startswith("en"):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def script_counts(text: str) -> tuple[int, int]:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    return cjk, latin


def asr_language_mismatch(text: str, language: str = "") -> bool:
    text = (text or "").strip()
    if not text:
        return False
    lang = language.lower()
    cjk, latin = script_counts(text)
    if lang.startswith("zh"):
        # Allow occasional English acronyms in Chinese text, but reject clear English hallucinations.
        return cjk == 0 and latin >= 8
    if lang.startswith("en"):
        # Allow a few CJK names, but reject clear Chinese transcripts for English rows.
        return cjk >= 4 and cjk > latin
    return False


def qwen_asr_language_hint(language: str | None) -> str | None:
    """Translate common dataset language codes to Qwen-ASR language names."""

    lang = str(language or "").strip().lower().replace("_", "-")
    if not lang:
        return None
    aliases = {
        "zh": "Chinese",
        "zho": "Chinese",
        "chi": "Chinese",
        "cmn": "Chinese",
        "cn": "Chinese",
        "zh-cn": "Chinese",
        "zh-hans": "Chinese",
        "chinese": "Chinese",
        "mandarin": "Chinese",
        "yue": "Cantonese",
        "zh-hk": "Cantonese",
        "zh-yue": "Cantonese",
        "cantonese": "Cantonese",
        "en": "English",
        "eng": "English",
        "en-us": "English",
        "en-gb": "English",
        "english": "English",
    }
    return aliases.get(lang, str(language).strip())


def asr_model_label(args: argparse.Namespace) -> str:
    backend = str(args.asr_backend or "")
    if backend == "qwen_asr":
        return str(args.qwen_asr_model or "")
    if backend == "faster_whisper":
        return str(args.faster_whisper_model or args.whisper_model or "")
    if backend == "whisper":
        return str(args.whisper_model or "")
    if backend == "jsonl_map":
        return str(args.asr_map_jsonl or "")
    return ""


def repeated_ngram_ratio(text: str, max_n: int = 4) -> float:
    units = words_for_wer(text)
    if len(units) < 4:
        return 0.0
    best = 0.0
    for n in range(2, max_n + 1):
        grams = [tuple(units[i : i + n]) for i in range(0, len(units) - n + 1)]
        if not grams:
            continue
        counts = Counter(grams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        best = max(best, repeated / max(1, len(grams)))
    return float(best)


def duration_seconds(path: str | None) -> float | None:
    if not path:
        return None
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def load_asr_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    mapping: dict[str, str] = {}
    for row in iter_jsonl(Path(path)):
        text = first_text(row, ("asr_text", "text", "transcript", "normalized_text"))
        for key in ("audio", "wav", "path", "source_audio", "target_audio"):
            value = record_get(row, key)
            if value and text:
                mapping[str(value)] = text
                mapping[str(Path(str(value)).resolve(strict=False))] = text
    return mapping


def patch_transformers_for_qwen_asr() -> None:
    """Patch optional/compatibility pieces needed by the local Qwen-ASR env."""

    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils

        def _false() -> bool:
            return False

        if hasattr(import_utils.is_sklearn_available, "cache_clear"):
            import_utils.is_sklearn_available.cache_clear()
        import_utils.is_sklearn_available = _false
        transformers_utils.is_sklearn_available = _false
    except Exception:
        pass
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if "default" not in ROPE_INIT_FUNCTIONS:

            def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
                _ = seq_len
                if config is None:
                    raise ValueError("config is required for default RoPE parameters")
                if hasattr(config, "standardize_rope_params"):
                    config.standardize_rope_params()
                rope_parameters = getattr(config, "rope_parameters", None)
                if isinstance(rope_parameters, dict):
                    params = rope_parameters.get(layer_type, rope_parameters) if layer_type is not None else rope_parameters
                    base = params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
                    partial_rotary_factor = params.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
                else:
                    base = getattr(config, "rope_theta", 10000.0)
                    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                dim = int(head_dim * partial_rotary_factor)
                inv_freq = 1.0 / (
                    base
                    ** (
                        torch.arange(0, dim, 2, dtype=torch.int64)
                        .to(device=device, dtype=torch.float)
                        / dim
                    )
                )
                return inv_freq, 1.0

            import torch

            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    except Exception:
        pass
    try:
        from qwen_asr.core.transformers_backend import configuration_qwen3_asr as qwen3_asr_config

        def _has_value(obj, name: str) -> bool:
            try:
                return getattr(obj, name) is not None
            except AttributeError:
                return False

        def _set_default_token_ids(config) -> None:
            if not _has_value(config, "pad_token_id"):
                config.pad_token_id = 151643
            if not _has_value(config, "eos_token_id"):
                config.eos_token_id = [151643, 151645]
            if not _has_value(config, "bos_token_id"):
                config.bos_token_id = None

        def _patch_config_class(cls) -> None:
            if getattr(cls, "_moss_codecvc_token_patch", False):
                return
            original_init = cls.__init__

            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                _set_default_token_ids(self)
                for child_name in ("thinker_config", "text_config"):
                    child = getattr(self, child_name, None)
                    if child is not None:
                        _set_default_token_ids(child)

            cls.__init__ = patched_init
            cls._moss_codecvc_token_patch = True

        for class_name in ("Qwen3ASRConfig", "Qwen3ASRThinkerConfig", "Qwen3ASRTextConfig"):
            cls = getattr(qwen3_asr_config, class_name, None)
            if cls is not None:
                _patch_config_class(cls)
    except Exception:
        pass
    try:
        from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen3_asr_modeling
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        cls = getattr(qwen3_asr_modeling, "Qwen3ASRThinkerTextRotaryEmbedding", None)
        if cls is not None and not hasattr(cls, "compute_default_rope_parameters"):

            def compute_default_rope_parameters(self, config=None):
                return ROPE_INIT_FUNCTIONS["default"](config if config is not None else self.config)

            cls.compute_default_rope_parameters = compute_default_rope_parameters
    except Exception:
        pass
    try:
        from transformers import AutoProcessor

        original_from_pretrained = AutoProcessor.from_pretrained
        if not getattr(original_from_pretrained, "_moss_codecvc_fix_mistral_patch", False):

            def patched_from_pretrained(*args, **kwargs):
                kwargs.pop("fix_mistral_regex", None)
                return original_from_pretrained(*args, **kwargs)

            patched_from_pretrained._moss_codecvc_fix_mistral_patch = True
            AutoProcessor.from_pretrained = patched_from_pretrained
    except Exception:
        pass


class ASRBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.map = load_asr_map(args.asr_map_jsonl)
        self.model = None
        if args.asr_backend == "faster_whisper":
            from faster_whisper import WhisperModel

            model_name = args.faster_whisper_model or args.whisper_model
            self.model = WhisperModel(model_name, device=args.device)
        elif args.asr_backend == "whisper":
            import whisper

            self.model = whisper.load_model(args.whisper_model, device=args.device)
        elif args.asr_backend == "qwen_asr":
            try:
                patch_transformers_for_qwen_asr()
                import torch
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise ImportError(
                    "--asr-backend=qwen_asr requires the `qwen-asr` package. "
                    "Install it in the data-processing environment, or use --asr-backend=jsonl_map/existing_fields."
                ) from exc
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[str(args.qwen_asr_dtype)]
            device_map = args.device
            if device_map == "cuda":
                device_map = "cuda:0"
            self.model = Qwen3ASRModel.from_pretrained(
                args.qwen_asr_model,
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=int(args.qwen_asr_max_batch_size),
                max_new_tokens=int(args.qwen_asr_max_new_tokens),
            )

    def transcribe(self, audio_path: str | None, *, fallback: str = "", language: str = "") -> str:
        if not audio_path:
            return fallback
        language_hint = language or self.args.language or None
        if self.args.asr_backend == "jsonl_map":
            return self.map.get(str(audio_path)) or self.map.get(str(Path(audio_path).resolve(strict=False))) or fallback
        if self.args.asr_backend == "faster_whisper":
            segments, _info = self.model.transcribe(audio_path, language=language_hint)
            return "".join(segment.text for segment in segments).strip()
        if self.args.asr_backend == "whisper":
            result = self.model.transcribe(audio_path, language=language_hint)
            return str(result.get("text") or "").strip()
        if self.args.asr_backend == "qwen_asr":
            language_hint = qwen_asr_language_hint(language_hint)
            results = self.model.transcribe(audio=audio_path, language=language_hint)
            if isinstance(results, (list, tuple)) and results:
                item = results[0]
            else:
                item = results
            if isinstance(item, dict):
                return str(item.get("text") or "").strip()
            return str(getattr(item, "text", "") or "").strip()
        return fallback

    def transcribe_many(
        self,
        audio_paths: list[str | None],
        *,
        fallbacks: list[str] | None = None,
        languages: list[str] | None = None,
    ) -> list[str]:
        fallbacks = fallbacks or [""] * len(audio_paths)
        languages = languages or [""] * len(audio_paths)
        if len(fallbacks) != len(audio_paths) or len(languages) != len(audio_paths):
            raise ValueError("audio_paths/fallbacks/languages length mismatch")
        if self.args.asr_backend != "qwen_asr":
            return [
                self.transcribe(audio_path, fallback=fallback, language=language)
                for audio_path, fallback, language in zip(audio_paths, fallbacks, languages)
            ]

        out = list(fallbacks)
        batch_audio: list[str] = []
        batch_language: list[str | None] = []
        batch_indices: list[int] = []
        for idx, (audio_path, language) in enumerate(zip(audio_paths, languages)):
            if not audio_path:
                continue
            batch_audio.append(str(audio_path))
            batch_language.append(qwen_asr_language_hint(language or self.args.language or ""))
            batch_indices.append(idx)
        if not batch_audio:
            return out

        results = self.model.transcribe(audio=batch_audio, language=batch_language)
        if not isinstance(results, (list, tuple)):
            results = [results]
        if len(results) != len(batch_indices):
            raise RuntimeError(f"Qwen-ASR batch result mismatch: expected={len(batch_indices)} got={len(results)}")
        for idx, item in zip(batch_indices, results):
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
            else:
                text = str(getattr(item, "text", "") or "").strip()
            out[idx] = text
        return out


def content_decision(
    *,
    no_text: bool,
    language: str,
    ref_text: str,
    asr_tgt_text: str,
    repeat_score: float,
    duration_ratio: float | None,
    source_asr_lang_mismatch: bool,
    target_asr_lang_mismatch: bool,
    args: argparse.Namespace,
) -> tuple[bool, str, float, float]:
    reasons = []
    metric_cer = cer(asr_tgt_text, ref_text) if ref_text and asr_tgt_text else math.inf
    metric_wer = wer(asr_tgt_text, ref_text) if ref_text and asr_tgt_text else math.inf
    if not ref_text:
        reasons.append("missing_source_asr" if no_text else "missing_ref_text")
    if source_asr_lang_mismatch:
        reasons.append("source_asr_lang_mismatch")
    if target_asr_lang_mismatch:
        reasons.append("target_asr_lang_mismatch")
    if len(normalize_for_cer(asr_tgt_text)) < int(args.min_asr_chars):
        reasons.append("empty_or_too_short_target_asr")
    if repeat_score > float(args.max_repeat_score):
        reasons.append("repeat_score")
    if duration_ratio is not None:
        if duration_ratio < float(args.min_duration_ratio):
            reasons.append("target_too_short")
        if duration_ratio > float(args.max_duration_ratio):
            reasons.append("target_too_long")
    zh = is_zh(ref_text, language)
    if zh:
        threshold = args.no_text_zh_cer_threshold if no_text else args.zh_cer_threshold
        if metric_cer > float(threshold):
            reasons.append("cer")
    else:
        threshold = args.no_text_en_wer_threshold if no_text else args.en_wer_threshold
        if metric_wer > float(threshold):
            reasons.append("wer")
    return (not reasons), ",".join(reasons) if reasons else "keep", float(metric_cer), float(metric_wer)


def prepare_record_context(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    no_text = mode_is_no_text(record)
    language = str(record_get(record, "language") or args.language or "")
    source_audio = record_get(record, "source_audio")
    target_audio = record_get(record, "target_audio")
    asr_src_existing = first_text(record, SOURCE_ASR_KEYS)
    asr_tgt_existing = first_text(record, TARGET_ASR_KEYS)
    reference_mode = str(args.content_reference_mode or "auto").strip().lower()
    asr_language = language or str(args.language or "")
    needs_source_asr = (
        not bool(args.skip_source_asr)
        or reference_mode == "source"
        or (reference_mode == "auto" and no_text)
    )
    return {
        "no_text": no_text,
        "language": language,
        "source_audio": source_audio,
        "target_audio": target_audio,
        "asr_src_existing": asr_src_existing,
        "asr_tgt_existing": asr_tgt_existing,
        "reference_mode": reference_mode,
        "asr_language": asr_language,
        "needs_source_asr": needs_source_asr,
    }


def finish_record_with_asr_text(
    record: dict[str, Any],
    *,
    context: dict[str, Any],
    asr_src_text: str,
    asr_tgt_text: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out = dict(record)
    no_text = bool(context["no_text"])
    language = str(context["language"])
    source_audio = context["source_audio"]
    target_audio = context["target_audio"]
    reference_mode = str(context["reference_mode"])
    asr_language = str(context["asr_language"])
    needs_source_asr = bool(context["needs_source_asr"])
    explicit_text = first_text(record, TARGET_TEXT_KEYS if reference_mode in {"text", "target_text"} else TEXT_KEYS)
    if reference_mode == "source":
        content_ref_text = asr_src_text
        content_ref_text_source = "source_asr"
    elif reference_mode in {"text", "target_text"}:
        content_ref_text = explicit_text
        content_ref_text_source = "target_text"
    else:
        content_ref_text = asr_src_text if no_text else explicit_text
        content_ref_text_source = "source_asr" if no_text else "target_text"
    if not content_ref_text and not no_text and reference_mode == "auto":
        content_ref_text = asr_src_text
        content_ref_text_source = "source_asr"

    source_dur = duration_seconds(source_audio) if not args.disable_duration_ratio_check else None
    target_dur = duration_seconds(target_audio) if not args.disable_duration_ratio_check else None
    duration_ratio = None
    if source_dur and source_dur > 0 and target_dur and target_dur > 0:
        duration_ratio = float(target_dur) / float(source_dur)
    repeat_score = repeated_ngram_ratio(asr_tgt_text)
    source_asr_lang_mismatch = asr_language_mismatch(asr_src_text, language)
    target_asr_lang_mismatch = asr_language_mismatch(asr_tgt_text, language)
    keep, reason, metric_cer, metric_wer = content_decision(
        no_text=no_text,
        language=language,
        ref_text=content_ref_text,
        asr_tgt_text=asr_tgt_text,
        repeat_score=repeat_score,
        duration_ratio=duration_ratio,
        source_asr_lang_mismatch=source_asr_lang_mismatch,
        target_asr_lang_mismatch=target_asr_lang_mismatch,
        args=args,
    )
    out.update(
        {
            "asr_src_text": asr_src_text,
            "asr_tgt_text": asr_tgt_text,
            "content_ref_text": content_ref_text,
            "content_ref_text_source": content_ref_text_source,
            "content_asr_backend": str(args.asr_backend) if content_ref_text_source == "source_asr" else "",
            "content_asr_model": asr_model_label(args) if content_ref_text_source == "source_asr" else "",
            "source_asr_backend": str(args.asr_backend) if needs_source_asr else "existing_fields",
            "source_asr_model": asr_model_label(args) if needs_source_asr else "",
            "target_asr_backend": str(args.asr_backend),
            "target_asr_model": asr_model_label(args),
            "cer_tgt": metric_cer,
            "wer_tgt": metric_wer,
            "repeat_score": repeat_score,
            "duration_ratio_tgt_src": duration_ratio,
            "asr_language_hint": asr_language,
            "source_asr_lang_mismatch": bool(source_asr_lang_mismatch),
            "target_asr_lang_mismatch": bool(target_asr_lang_mismatch),
            "content_keep": bool(keep),
            "content_filter_reason": reason,
        }
    )
    return out


def process_record(record: dict[str, Any], asr: ASRBackend, args: argparse.Namespace) -> dict[str, Any]:
    context = prepare_record_context(record, args)
    asr_src_text = (
        asr.transcribe(
            context["source_audio"],
            fallback=str(context["asr_src_existing"]),
            language=str(context["asr_language"]),
        )
        if context["needs_source_asr"]
        else str(context["asr_src_existing"])
    )
    asr_tgt_text = asr.transcribe(
        context["target_audio"],
        fallback=str(context["asr_tgt_existing"]),
        language=str(context["asr_language"]),
    )
    return finish_record_with_asr_text(
        record,
        context=context,
        asr_src_text=asr_src_text,
        asr_tgt_text=asr_tgt_text,
        args=args,
    )


def process_record_batch(records: list[dict[str, Any]], asr: ASRBackend, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not records:
        return []
    if args.asr_backend != "qwen_asr" or int(args.qwen_asr_max_batch_size) <= 1:
        return [process_record(record, asr, args) for record in records]

    try:
        contexts = [prepare_record_context(record, args) for record in records]
        source_texts = [str(context["asr_src_existing"]) for context in contexts]
        source_indices = [idx for idx, context in enumerate(contexts) if context["needs_source_asr"]]
        if source_indices:
            source_batch = asr.transcribe_many(
                [contexts[idx]["source_audio"] for idx in source_indices],
                fallbacks=[str(contexts[idx]["asr_src_existing"]) for idx in source_indices],
                languages=[str(contexts[idx]["asr_language"]) for idx in source_indices],
            )
            for idx, text in zip(source_indices, source_batch):
                source_texts[idx] = text

        target_texts = asr.transcribe_many(
            [context["target_audio"] for context in contexts],
            fallbacks=[str(context["asr_tgt_existing"]) for context in contexts],
            languages=[str(context["asr_language"]) for context in contexts],
        )
        return [
            finish_record_with_asr_text(
                record,
                context=context,
                asr_src_text=asr_src_text,
                asr_tgt_text=asr_tgt_text,
                args=args,
            )
            for record, context, asr_src_text, asr_tgt_text in zip(records, contexts, source_texts, target_texts)
        ]
    except Exception as exc:
        print(
            f"[asr-filter] qwen batch failed; falling back to single-record ASR for batch_size={len(records)}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return [process_record(record, asr, args) for record in records]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    if int(args.num_shards) <= 0:
        raise ValueError("--num-shards must be positive")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    resume_existing_rows = 0
    append_output = False
    if output_path.exists() and not args.overwrite:
        if not args.resume:
            raise FileExistsError(f"output exists, pass --overwrite or --resume: {output_path}")
        resume_existing_rows = count_jsonl_lines(output_path)
        append_output = True
        print(
            f"[asr-filter] resume output={output_path} existing_shard_rows={resume_existing_rows}",
            flush=True,
        )
    elif output_path.exists() and args.overwrite:
        append_output = False
    elif args.resume:
        print(f"[asr-filter] resume requested but output does not exist yet: {output_path}", flush=True)
    if output_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"output exists, pass --overwrite: {output_path}")
    asr = ASRBackend(args)
    stats = Counter()
    stats["resume_existing_rows"] = int(resume_existing_rows)
    record_batch_size = 1
    if args.asr_backend == "qwen_asr":
        record_batch_size = max(1, int(args.qwen_asr_max_batch_size))
        if record_batch_size > 1:
            print(f"[asr-filter] qwen_record_batch_size={record_batch_size}", flush=True)

    def rows():
        shard_rows_seen = 0
        pending: list[dict[str, Any]] = []

        def emit_pending():
            if not pending:
                return
            outputs = process_record_batch(pending, asr, args)
            pending.clear()
            for out in outputs:
                stats["rows"] += 1
                stats["kept" if out["content_keep"] else "filtered"] += 1
                for reason in str(out["content_filter_reason"]).split(","):
                    if reason and reason != "keep":
                        stats[f"reason:{reason}"] += 1
                if stats["rows"] % max(1, int(args.progress_every)) == 0:
                    print(f"[asr-filter] rows={stats['rows']} kept={stats['kept']} filtered={stats['filtered']}", flush=True)
                yield out

        for idx, row in enumerate(iter_jsonl(input_path)):
            if args.limit > 0 and idx >= args.limit:
                break
            stats["input_rows_seen"] += 1
            if int(args.num_shards) > 1 and (idx % int(args.num_shards)) != int(args.shard_index):
                continue
            if shard_rows_seen < resume_existing_rows:
                shard_rows_seen += 1
                stats["resume_skipped_rows"] += 1
                continue
            shard_rows_seen += 1
            pending.append(row)
            if len(pending) >= record_batch_size:
                yield from emit_pending()
        yield from emit_pending()

    count = write_jsonl(output_path, rows(), append=append_output)
    total_rows = resume_existing_rows + count
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": total_rows,
                "new_rows": count,
                "resume_existing_rows": resume_existing_rows,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "stats": dict(stats),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[asr-filter] wrote new_rows={count} total_rows={total_rows} output={output_path}")
    print(f"[asr-filter] summary={summary_path} stats={dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
