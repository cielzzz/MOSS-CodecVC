#!/usr/bin/env python
"""Unified, mergeable VC evaluation for Batch-42.

The legacy SeedTTS-320 scripts remain the source of truth for historical
results.  This entry point adds a stable schema around them and provides lazy
adapters for the paper-facing speaker and ASR backends.  Each backend can be
run in an isolated process/environment, then merged without recomputing the
other metrics.

The module intentionally imports only the Python standard library at import
time.  Heavy dependencies and model weights are loaded only when a concrete
backend is executed; ``--schema-only`` and the unit tests therefore need no
models or GPUs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import string
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "moss_codecvc.unified_vc_eval.v1"

SPEAKER_BACKENDS = ("wavlm_large_sv", "eres2net", "speechbrain_ecapa")
ASR_BACKENDS = ("paraformer_zh", "whisper_large_v3", "qwen_asr")
INPUT_PROFILES = ("auto", "legacy_internal", "official_seedtts_vc")
METRIC_PROFILES = ("seedtts_official", "legacy_internal")
RESULT_STATUSES = {
    "ok",
    "precomputed",
    "pending",
    "skipped_language",
    "missing_audio",
    "missing_reference",
    "backend_unavailable",
    "error",
}

# Exact ``zhon.hanzi.punctuation`` value used by Seed-TTS-Eval/run_wer.py.
# Keeping the constant here makes schema/unit-test operations dependency-free;
# the official ASR model adapter still requires zhconv for Mandarin hypotheses.
SEEDTTS_ZHON_HANZI_PUNCTUATION = (
    "\uFF02\uFF03\uFF04\uFF05\uFF06\uFF07\uFF08\uFF09\uFF0A\uFF0B\uFF0C\uFF0D"
    "\uFF0F\uFF1A\uFF1B\uFF1C\uFF1D\uFF1E\uFF20\uFF3B\uFF3C\uFF3D\uFF3E\uFF3F"
    "\uFF40\uFF5B\uFF5C\uFF5D\uFF5E\uFF5F\uFF60\uFF62\uFF63\uFF64\u3000\u3001"
    "\u3003\u3008\u3009\u300A\u300B\u300C\u300D\u300E\u300F\u3010\u3011\u3014"
    "\u3015\u3016\u3017\u3018\u3019\u301A\u301B\u301C\u301D\u301E\u301F\u3030"
    "\u303E\u303F\u2013\u2014\u2018\u2019\u201B\u201C\u201D\u201E\u201F\u2026"
    "\u2027\uFE4F\uFE51\uFE54\u00B7\uFF0E\uFF01\uFF1F\uFF61\u3002"
)

DEFAULT_SPEAKER_SIM_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction"
)
DEFAULT_SEEDTTS_MODEL_ROOT = Path(
    "/inspire/hdd/project/embodied-multimodality/public/kxhuang/vcdata_construction/models"
)
DEFAULT_SPEECHBRAIN_MODEL = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/"
    "speechbrain/spkrec-ecapa-voxceleb"
)
DEFAULT_QWEN_ASR_MODEL = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b"
)

SPEAKER_ALIASES = {
    "wavlm": "wavlm_large_sv",
    "wavlm_sv": "wavlm_large_sv",
    "wavlm_large": "wavlm_large_sv",
    "seedtts": "wavlm_large_sv",
    "seedtts_wavlm_ecapa": "wavlm_large_sv",
    "seedtts_wavlm_large_ecapa": "wavlm_large_sv",
    "ecapa": "speechbrain_ecapa",
    "spb_ecapa": "speechbrain_ecapa",
    "speechbrain": "speechbrain_ecapa",
    "eres2": "eres2net",
}

ASR_ALIASES = {
    "paraformer": "paraformer_zh",
    "paraformer-zh": "paraformer_zh",
    "whisper": "whisper_large_v3",
    "whisper-large-v3": "whisper_large_v3",
    "qwen": "qwen_asr",
    "qwen3_asr": "qwen_asr",
}


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def first_value(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def first_named_value(row: dict[str, Any], keys: Sequence[str]) -> tuple[str, Any]:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return key, value
    return "", None


def first_text(row: dict[str, Any], keys: Sequence[str]) -> str:
    value = first_value(row, keys)
    return str(value).strip() if value not in (None, "") else ""


def normalize_language(value: Any, reference_text: str = "") -> str:
    lang = str(value or "").strip().lower().replace("_", "-")
    if lang.startswith(("zh", "cmn", "zho", "chi")) or lang in {"cn", "chinese", "mandarin"}:
        return "zh"
    if lang.startswith(("en", "eng")) or lang == "english":
        return "en"
    if re.search(r"[\u4e00-\u9fff]", reference_text):
        return "zh"
    if re.search(r"[A-Za-z]", reference_text):
        return "en"
    return lang or "unknown"


def normalize_for_cer(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def words_for_wer(text: str) -> list[str]:
    text = str(text or "").lower()
    if re.search(r"[\u4e00-\u9fff]", text):
        return list(normalize_for_cer(text))
    text = re.sub(r"[^\w\s']+", " ", text)
    return [item for item in text.split() if item]


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        short, long = left, right
    else:
        short, long = right, left
    previous = list(range(len(short) + 1))
    for row_index, long_item in enumerate(long, start=1):
        current = [row_index]
        for col_index, short_item in enumerate(short, start=1):
            current.append(
                min(
                    previous[col_index] + 1,
                    current[-1] + 1,
                    previous[col_index - 1] + (0 if long_item == short_item else 1),
                )
            )
        previous = current
    return previous[-1]


def cer(hypothesis: str, reference: str) -> float | None:
    ref = list(normalize_for_cer(reference))
    if not ref:
        return None
    return edit_distance(list(normalize_for_cer(hypothesis)), ref) / len(ref)


def wer(hypothesis: str, reference: str) -> float | None:
    ref = words_for_wer(reference)
    if not ref:
        return None
    return edit_distance(words_for_wer(hypothesis), ref) / len(ref)


def seedtts_official_units(text: str, language: str) -> list[str]:
    """Reproduce Seed-TTS-Eval ``run_wer.py::process_one`` tokenization.

    The official script removes zhon + ASCII punctuation except the ASCII
    apostrophe, converts Mandarin strings to character tokens, lowercases
    English, and lets jiwer collapse whitespace. ``str.split`` has the same
    whitespace behavior for these already-normalized strings.
    """

    normalized = str(text or "")
    for mark in SEEDTTS_ZHON_HANZI_PUNCTUATION + string.punctuation:
        if mark != "'":
            normalized = normalized.replace(mark, "")
    normalized = normalized.replace("  ", " ")
    if language == "zh":
        return " ".join(list(normalized)).split()
    if language == "en":
        return normalized.lower().split()
    return []


def seedtts_official_error_rate(
    hypothesis: str, reference: str, language: str
) -> float | None:
    reference_units = seedtts_official_units(reference, language)
    if not reference_units:
        return None
    hypothesis_units = seedtts_official_units(hypothesis, language)
    return edit_distance(hypothesis_units, reference_units) / len(reference_units)


def metric_profile_id(metric_profile: str) -> str:
    if metric_profile == "seedtts_official":
        return "seedtts_eval_run_wer.py"
    if metric_profile == "legacy_internal":
        return "moss_codecvc_001017_legacy"
    raise ValueError(f"unsupported metric profile: {metric_profile!r}")


def error_rate_payload(
    hypothesis: str, reference: str, language: str, metric_profile: str
) -> dict[str, Any]:
    if metric_profile == "seedtts_official":
        primary = seedtts_official_error_rate(hypothesis, reference, language)
        metric = "cer" if language == "zh" else "wer" if language == "en" else "unknown"
        return {
            "metric_profile": metric_profile_id(metric_profile),
            "metric": metric,
            "primary_error": primary,
            "cer": primary if metric == "cer" else None,
            "wer": primary if metric == "wer" else None,
        }
    if metric_profile != "legacy_internal":
        raise ValueError(f"unsupported metric profile: {metric_profile!r}")
    cer_value = cer(hypothesis, reference)
    wer_value = wer(hypothesis, reference)
    metric = "cer" if language == "zh" else "wer" if language == "en" else "unknown"
    return {
        "metric_profile": metric_profile_id(metric_profile),
        "metric": metric,
        "primary_error": cer_value if metric == "cer" else wer_value if metric == "wer" else None,
        "cer": cer_value,
        "wer": wer_value,
    }


def canonical_speaker_backend(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    key = SPEAKER_ALIASES.get(key, key)
    if key not in SPEAKER_BACKENDS:
        raise ValueError(f"unsupported speaker scorer: {value!r}; choices={SPEAKER_BACKENDS}")
    return key


def canonical_asr_backend(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    key = ASR_ALIASES.get(key, key)
    if key not in ASR_BACKENDS:
        raise ValueError(f"unsupported ASR backend: {value!r}; choices={ASR_BACKENDS}")
    return key


def parse_backend_list(values: Sequence[str] | None, *, kind: str) -> list[str]:
    raw: list[str] = []
    for value in values or []:
        raw.extend(item.strip() for item in str(value).split(",") if item.strip())
    if not raw:
        return []
    all_values = SPEAKER_BACKENDS if kind == "speaker" else ASR_BACKENDS
    if any(item.lower() == "all" for item in raw):
        return list(all_values)
    convert = canonical_speaker_backend if kind == "speaker" else canonical_asr_backend
    out: list[str] = []
    for item in raw:
        backend = convert(item)
        if backend not in out:
            out.append(backend)
    return out


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def extract_legacy_speaker_results(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    nested = row.get("speaker_similarity")
    if isinstance(nested, dict):
        for raw_backend, raw_result in nested.items():
            if not isinstance(raw_result, dict):
                continue
            try:
                backend = canonical_speaker_backend(str(raw_backend))
            except ValueError:
                continue
            result = dict(raw_result)
            result.setdefault("status", "precomputed")
            result.setdefault("backend", backend)
            results[backend] = result

    wavlm_ref = finite(
        first_value(
            row,
            (
                "seedtts_wavlm_ecapa_sim_ref",
                "wavlm_sim_ref",
                "sim_gen_ref",
            ),
        )
    )
    wavlm_src = finite(
        first_value(
            row,
            (
                "seedtts_wavlm_ecapa_sim_src",
                "wavlm_sim_src",
                "sim_gen_source",
                "sim_gen_src",
            ),
        )
    )
    if wavlm_ref is not None or wavlm_src is not None:
        results.setdefault(
            "wavlm_large_sv",
            {
                "status": "precomputed",
                "backend": "wavlm_large_sv",
                "implementation": "legacy_seedtts_wavlm_large_ecapa",
                "model_id": "legacy:Seed-TTS-Eval-WavLM-Large+ECAPA",
                "sim_ref": wavlm_ref,
                "sim_src": wavlm_src,
            },
        )

    ecapa_ref = finite(
        first_value(
            row,
            (
                "speechbrain_ecapa_sim_ref",
                "speechbrain_sim_ref",
                "ecapa_sim_gen_ref",
                "ecapa_sim_ref",
            ),
        )
    )
    ecapa_src = finite(
        first_value(
            row,
            (
                "speechbrain_ecapa_sim_src",
                "speechbrain_sim_src",
                "ecapa_sim_gen_source",
                "ecapa_sim_src",
            ),
        )
    )
    if ecapa_ref is not None or ecapa_src is not None:
        results.setdefault(
            "speechbrain_ecapa",
            {
                "status": "precomputed",
                "backend": "speechbrain_ecapa",
                "implementation": "legacy_speechbrain",
                "model_id": "legacy:speechbrain/spkrec-ecapa-voxceleb",
                "sim_ref": ecapa_ref,
                "sim_src": ecapa_src,
            },
        )
    return results


def infer_legacy_asr_backend(row: dict[str, Any], default_backend: str) -> str:
    label = first_text(row, ("content_asr_backend", "asr_backend", "asr_model"))
    lowered = label.lower()
    if "paraformer" in lowered:
        return "paraformer_zh"
    if "whisper" in lowered:
        return "whisper_large_v3"
    if "qwen" in lowered:
        return "qwen_asr"
    return canonical_asr_backend(default_backend)


def extract_legacy_asr_results(
    row: dict[str, Any], *, language: str, reference_text: str, default_backend: str
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    nested = row.get("content_asr")
    if isinstance(nested, dict):
        for raw_backend, raw_result in nested.items():
            if not isinstance(raw_result, dict):
                continue
            try:
                backend = canonical_asr_backend(str(raw_backend))
            except ValueError:
                continue
            result = dict(raw_result)
            result.setdefault("status", "precomputed")
            result.setdefault("backend", backend)
            result.setdefault("language", language)
            result.setdefault("reference_text", reference_text)
            result.setdefault("metric_profile", "legacy_precomputed_unspecified")
            results[backend] = result

    hypothesis = first_text(row, ("asr_tgt_text", "hypothesis", "transcript"))
    legacy_cer = finite(first_value(row, ("cer_tgt", "cer")))
    legacy_wer = finite(first_value(row, ("wer_tgt", "wer")))
    if hypothesis or legacy_cer is not None or legacy_wer is not None:
        backend = infer_legacy_asr_backend(row, default_backend)
        primary_metric = "cer" if language == "zh" else "wer" if language == "en" else "unknown"
        primary_error = legacy_cer if primary_metric == "cer" else legacy_wer if primary_metric == "wer" else None
        results.setdefault(
            backend,
            {
                "status": "precomputed",
                "backend": backend,
                "model_id": first_text(row, ("content_asr_model", "asr_model")) or f"legacy:{backend}",
                "language": language,
                "hypothesis": hypothesis,
                "reference_text": reference_text,
                "metric_profile": "legacy_precomputed_unspecified",
                "metric": primary_metric,
                "primary_error": primary_error,
                "cer": legacy_cer,
                "wer": legacy_wer,
            },
        )
    return results


def validate_metric_sections(row: dict[str, Any], *, context: str) -> None:
    for section, allowed_backends in (
        ("speaker_similarity", set(SPEAKER_BACKENDS)),
        ("content_asr", set(ASR_BACKENDS)),
    ):
        payload = row.get(section, {})
        if not isinstance(payload, dict):
            raise ValueError(f"{context}: {section} must be an object")
        for backend, result in payload.items():
            if backend not in allowed_backends:
                raise ValueError(f"{context}: unsupported {section} backend {backend!r}")
            if not isinstance(result, dict):
                raise ValueError(f"{context}: {section}.{backend} must be an object")
            if result.get("backend") != backend:
                raise ValueError(
                    f"{context}: {section}.{backend}.backend must equal {backend!r}"
                )
            status = str(result.get("status") or "")
            if status not in RESULT_STATUSES:
                raise ValueError(
                    f"{context}: {section}.{backend} has unsupported status {status!r}"
                )


def normalize_case(
    row: dict[str, Any],
    *,
    input_index: int,
    input_path: Path,
    run_id: str,
    system_override: str,
    test_set_override: str,
    legacy_asr_backend: str,
    input_profile: str = "auto",
) -> dict[str, Any]:
    if input_profile not in INPUT_PROFILES:
        raise ValueError(f"unsupported input profile: {input_profile!r}")
    if row.get("schema_version") == SCHEMA_VERSION and row.get("record_type") == "vc_eval_case":
        # Allow a schema-only output to be fed back to one concrete backend.
        # A JSON round-trip makes a defensive deep copy without importing any
        # non-standard dependency.
        case = json.loads(json.dumps(row, ensure_ascii=False))
        case["run_id"] = run_id or case.get("run_id")
        if system_override:
            case["system_id"] = system_override
        if test_set_override:
            case["test_set_id"] = test_set_override
        case.setdefault("speaker_similarity", {})
        case.setdefault("content_asr", {})
        validate_metric_sections(case, context=f"{input_path}:{input_index}")
        case.setdefault("provenance", {}).update(
            {
                "input_path": str(input_path),
                "input_index": input_index,
                "input_profile": input_profile,
            }
        )
        if input_profile == "official_seedtts_vc":
            audio = case.get("audio") if isinstance(case.get("audio"), dict) else {}
            required = {
                "system_id": case.get("system_id"),
                "test_set_id": case.get("test_set_id"),
                "case_id": case.get("case_id"),
                "language": case.get("language"),
                "audio.generated": audio.get("generated"),
                "audio.reference": audio.get("reference"),
                "audio.source": audio.get("source"),
                "reference_text": case.get("reference_text"),
            }
            missing = [name for name, value in required.items() if value in (None, "", "unknown")]
            generated_origin = (
                case.get("provenance", {})
                .get("audio_field_mapping", {})
                .get("generated")
            )
            if generated_origin == "target_audio":
                raise ValueError(
                    "official_seedtts_vc rejects canonical input whose generated audio "
                    "originated from legacy target_audio"
                )
            if missing:
                raise ValueError(
                    "official_seedtts_vc canonical input is incomplete; missing "
                    + ", ".join(missing)
                )
        return case

    reference_text = first_text(
        row,
        (
            "reference_text",
            "target_text",
            "content_ref_text",
            "normalized_text",
            "text",
            "source_text",
        ),
    )
    language = normalize_language(
        first_value(row, ("language", "target_lang", "source_lang", "lang")), reference_text
    )
    case_id = first_text(row, ("case_id", "sample_id", "utt_id", "utt", "id")) or f"case_{input_index:08d}"
    row_system_id = first_text(row, ("system_id", "system", "run", "run_id"))
    system_id = system_override or row_system_id or run_id
    test_set_id = test_set_override or first_text(row, ("test_set_id", "test_set", "dataset", "split")) or "unknown"
    generated_field, generated_value = first_named_value(
        row, ("generated_audio", "target_audio", "output_wav", "gen_audio", "hyp_audio")
    )
    reference_field, reference_value = first_named_value(
        row,
        ("reference_audio", "timbre_ref_audio", "ref_audio", "prompt_audio", "speaker_ref_audio"),
    )
    source_field, source_value = first_named_value(
        row, ("source_audio", "content_audio", "src_audio", "original_audio")
    )
    generated_audio = str(generated_value).strip() if generated_value not in (None, "") else ""
    reference_audio = str(reference_value).strip() if reference_value not in (None, "") else ""
    source_audio = str(source_value).strip() if source_value not in (None, "") else ""

    if input_profile == "official_seedtts_vc":
        # In Seed-TTS VC meta, infer_wav/"target" audio is the content source,
        # not the generated waveform. Requiring explicit canonical names stops
        # a ground-truth/source file from being silently scored as system output.
        required_explicit = {
            "case_id": row.get("case_id"),
            "language": row.get("language"),
            "generated_audio": row.get("generated_audio"),
            "reference_audio": row.get("reference_audio"),
            "source_audio": row.get("source_audio"),
            "reference_text": row.get("reference_text"),
            "system_id (row or --system-id)": system_override or row_system_id,
            "test_set_id (row or --test-set-id)": None if test_set_id == "unknown" else test_set_id,
        }
        missing = [name for name, value in required_explicit.items() if value in (None, "")]
        if missing:
            raise ValueError(
                "official_seedtts_vc requires explicit converted VC fields; missing "
                + ", ".join(missing)
                + ". Do not map infer_wav/target_audio to generated_audio."
            )
        generated_field = "generated_audio"
        reference_field = "reference_audio"
        source_field = "source_audio"

    speaker = extract_legacy_speaker_results(row)
    content_asr = extract_legacy_asr_results(
        row,
        language=language,
        reference_text=reference_text,
        default_backend=legacy_asr_backend,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "vc_eval_case",
        "run_id": run_id,
        "system_id": system_id,
        "test_set_id": test_set_id,
        "case_id": case_id,
        "language": language,
        "audio": {
            "generated": generated_audio,
            "reference": reference_audio,
            "source": source_audio,
        },
        "reference_text": reference_text,
        "metadata": {
            "mode": row.get("mode") or row.get("moss_codecvc_mode"),
            "cell": row.get("cell"),
            "source_lang": row.get("source_lang"),
            "ref_lang": row.get("ref_lang"),
        },
        "speaker_similarity": speaker,
        "content_asr": content_asr,
        "provenance": {
            "input_path": str(input_path),
            "input_index": input_index,
            "input_profile": input_profile,
            "audio_field_mapping": {
                "generated": generated_field,
                "reference": reference_field,
                "source": source_field,
            },
        },
    }


def backend_status(status: str, backend: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "backend": backend}
    result.update(extra)
    return result


def asr_is_applicable(backend: str, language: str) -> bool:
    if backend == "paraformer_zh":
        return language == "zh"
    if backend == "whisper_large_v3":
        return language == "en"
    return language in {"zh", "en", "unknown"}


class EmbeddingSpeakerScorer:
    backend_name = ""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}

    def embed(self, path: str) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def cosine(self, left: Any, right: Any) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def similarity(self, left_path: str, right_path: str) -> float:
        if left_path not in self.cache:
            self.cache[left_path] = self.embed(left_path)
        if right_path not in self.cache:
            self.cache[right_path] = self.embed(right_path)
        return float(self.cosine(self.cache[left_path], self.cache[right_path]))

    def metadata(self) -> dict[str, Any]:
        return {}


class SeedTTSWavLMLargeScorer(EmbeddingSpeakerScorer):
    backend_name = "wavlm_large_sv"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        root = Path(args.speaker_sim_root).expanduser()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from speaker_similarity import SpeakerSimilarity

        kwargs: dict[str, Any] = {"device": args.speaker_device}
        if args.wavlm_checkpoint:
            kwargs["checkpoint"] = args.wavlm_checkpoint
        if args.seedtts_eval_root:
            kwargs["seed_tts_eval_root"] = args.seedtts_eval_root
        if args.wavlm_model_dir:
            kwargs["wavlm_dir"] = args.wavlm_model_dir
        self.model = SpeakerSimilarity(**kwargs)

    def embed(self, path: str) -> Any:
        return self.model.embed_from_file(path)

    def cosine(self, left: Any, right: Any) -> float:
        return float(self.model.compute_similarity(left, right))

    def metadata(self) -> dict[str, Any]:
        return {
            "implementation": "seedtts_official_wavlm_large_ecapa",
            "model_id": "Seed-TTS-Eval:WavLM-Large+ECAPA-TDNN",
        }


class HFWavLMXVectorScorer(EmbeddingSpeakerScorer):
    backend_name = "wavlm_large_sv"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch = torch
        self.device = torch.device(args.speaker_device if torch.cuda.is_available() else "cpu")
        self.model_id = args.wavlm_hf_model
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        self.model = WavLMForXVector.from_pretrained(self.model_id).to(self.device).eval()

    def _load_audio(self, path: str) -> tuple[Any, int]:
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = self.torch.from_numpy(audio.mean(axis=1))
        target_rate = int(getattr(self.extractor, "sampling_rate", 16000))
        if int(sample_rate) != target_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform, int(sample_rate), target_rate)
            sample_rate = target_rate
        return waveform, int(sample_rate)

    def embed(self, path: str) -> Any:
        waveform, sample_rate = self._load_audio(path)
        inputs = self.extractor(
            waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            embedding = self.model(**inputs).embeddings.squeeze(0)
        return self.torch.nn.functional.normalize(embedding.float().cpu(), dim=0)

    def cosine(self, left: Any, right: Any) -> float:
        return float(self.torch.nn.functional.cosine_similarity(left[None], right[None]).item())

    def metadata(self) -> dict[str, Any]:
        return {"implementation": "huggingface_wavlm_xvector", "model_id": self.model_id}


class SpeechBrainEcapaScorer(EmbeddingSpeakerScorer):
    backend_name = "speechbrain_ecapa"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        import torch

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from moss_codecvc.third_party import add_download_python_deps

        add_download_python_deps()
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        self.torch = torch
        self.device = torch.device(args.speaker_device if torch.cuda.is_available() else "cpu")
        source = Path(args.speechbrain_model).expanduser()
        kwargs: dict[str, Any] = {"source": str(source)}
        if source.exists():
            local_source = str(source.resolve())
            kwargs.update(
                source=local_source,
                savedir=local_source,
                overrides={"pretrained_path": local_source},
            )
        try:
            kwargs["run_opts"] = {"device": str(self.device)}
            self.model = EncoderClassifier.from_hparams(**kwargs)
        except TypeError:
            kwargs.pop("run_opts", None)
            self.model = EncoderClassifier.from_hparams(**kwargs).to(self.device)
        self.model.eval()
        self.model_id = str(source)

    def embed(self, path: str) -> Any:
        with self.torch.inference_mode():
            if hasattr(self.model, "encode_file"):
                embedding = self.model.encode_file(path).squeeze()
            else:
                signal = self.model.load_audio(path).to(self.device)
                embedding = self.model.encode_batch(signal.unsqueeze(0)).squeeze()
        embedding = self.torch.as_tensor(embedding, dtype=self.torch.float32, device="cpu").flatten()
        return self.torch.nn.functional.normalize(embedding, dim=0)

    def cosine(self, left: Any, right: Any) -> float:
        return float(self.torch.nn.functional.cosine_similarity(left[None], right[None]).item())

    def metadata(self) -> dict[str, Any]:
        return {"implementation": "speechbrain_encoder_classifier", "model_id": self.model_id}


class ModelScopeERes2NetScorer:
    backend_name = "eres2net"

    def __init__(self, args: argparse.Namespace) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.model_id = args.eres2net_model
        device = args.speaker_device
        if device.startswith("cuda"):
            device = "gpu" + (":" + device.split(":", 1)[1] if ":" in device else "")
        self.pipeline = pipeline(task=Tasks.speaker_verification, model=self.model_id, device=device)
        self.cache: dict[tuple[str, str], float] = {}

    @staticmethod
    def _parse_score(value: Any) -> float:
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, dict):
            for key in ("score", "similarity", "cosine", "value"):
                score = finite(value.get(key))
                if score is not None:
                    return score
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return ModelScopeERes2NetScorer._parse_score(value[0])
        raise ValueError(f"cannot parse ERes2Net speaker-verification output: {value!r}")

    def similarity(self, left_path: str, right_path: str) -> float:
        key = (left_path, right_path)
        reverse = (right_path, left_path)
        if key in self.cache:
            return self.cache[key]
        if reverse in self.cache:
            return self.cache[reverse]
        result = self.pipeline([left_path, right_path])
        score = self._parse_score(result)
        self.cache[key] = score
        return score

    def metadata(self) -> dict[str, Any]:
        return {"implementation": "modelscope_speaker_verification", "model_id": self.model_id}


def build_speaker_scorer(backend: str, args: argparse.Namespace) -> Any:
    if backend == "wavlm_large_sv":
        if args.wavlm_implementation == "hf_xvector":
            return HFWavLMXVectorScorer(args)
        return SeedTTSWavLMLargeScorer(args)
    if backend == "speechbrain_ecapa":
        return SpeechBrainEcapaScorer(args)
    if backend == "eres2net":
        return ModelScopeERes2NetScorer(args)
    raise ValueError(backend)


class ParaformerASR:
    backend_name = "paraformer_zh"

    def __init__(self, args: argparse.Namespace) -> None:
        from funasr import AutoModel
        try:
            from zhconv import convert as zhconv_convert
        except ImportError as exc:
            raise ImportError(
                "Paraformer official Seed-TTS scoring requires zhconv; install the "
                "seed-tts-eval requirements before running this backend"
            ) from exc

        self.model_id = args.paraformer_model
        self.zhconv_convert = zhconv_convert
        self.model = AutoModel(
            model=self.model_id,
            device=args.asr_device,
            disable_update=True,
        )

    def transcribe(self, path: str, language: str) -> str:
        result = self.model.generate(input=path, batch_size_s=300)
        item = result[0] if isinstance(result, list) and result else result
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("sentence") or "")
        else:
            text = str(item or "")
        text = re.sub(r"<[^>]+>", "", text).strip()
        return str(self.zhconv_convert(text, "zh-cn"))

    def metadata(self) -> dict[str, Any]:
        return {"implementation": "funasr_auto_model", "model_id": self.model_id}


class WhisperLargeV3ASR:
    backend_name = "whisper_large_v3"

    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        import transformers.utils.import_utils as transformers_import_utils

        # This adapter always decodes through soundfile below.  Disable the
        # optional torchcodec branch before importing the pipeline: current
        # local torchcodec wheels can be discoverable yet fail at import time
        # because their libtorchcodec/FFmpeg ABI does not match torch 2.6.
        # The private flag is the source consulted by
        # transformers.is_torchcodec_available().
        transformers_import_utils._torchcodec_available = False
        from transformers import pipeline

        self.model_id = args.whisper_model
        device: int | str = args.asr_device
        if device.startswith("cuda:"):
            device = int(device.split(":", 1)[1])
        elif device == "cuda":
            device = 0
        elif device == "cpu":
            device = -1
        dtype = torch.float16 if isinstance(device, int) and device >= 0 else torch.float32
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=device,
            torch_dtype=dtype,
        )

    def _load_audio(self, path: str) -> dict[str, Any]:
        """Decode locally so Transformers never invokes torchcodec/FFmpeg.

        Some otherwise-valid Batch-42 environments have a torchcodec build
        whose shared libraries do not match the installed torch/FFmpeg stack.
        Passing a filename makes the ASR pipeline import that decoder.  A
        mono float32 array is the same pipeline input semantically and keeps
        audio decoding deterministic through soundfile instead.
        """
        import soundfile as sf
        import torch
        import torchaudio

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.mean(axis=1))
        target_rate = int(self.pipeline.feature_extractor.sampling_rate)
        if int(sample_rate) != target_rate:
            waveform = torchaudio.functional.resample(
                waveform, int(sample_rate), target_rate
            )
        return {
            "array": waveform.cpu().numpy(),
            "sampling_rate": target_rate,
        }

    def transcribe(self, path: str, language: str) -> str:
        kwargs: dict[str, Any] = {"task": "transcribe"}
        if language == "en":
            kwargs["language"] = "english"
        elif language == "zh":
            kwargs["language"] = "chinese"
        result = self.pipeline(self._load_audio(path), generate_kwargs=kwargs)
        if isinstance(result, dict):
            return str(result.get("text") or "").strip()
        return str(result or "").strip()

    def metadata(self) -> dict[str, Any]:
        return {
            "implementation": "transformers_asr_pipeline_soundfile_decode",
            "model_id": self.model_id,
        }


class LegacyQwenASR:
    backend_name = "qwen_asr"

    def __init__(self, args: argparse.Namespace) -> None:
        legacy_path = ROOT / "scripts/001017_asr_content_filter.py"
        spec = importlib.util.spec_from_file_location("moss_codecvc_legacy_asr_filter", legacy_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load legacy Qwen-ASR adapter from {legacy_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        legacy_args = SimpleNamespace(
            asr_backend="qwen_asr",
            asr_map_jsonl="",
            faster_whisper_model="",
            whisper_model="",
            qwen_asr_model=args.qwen_asr_model,
            qwen_asr_dtype=args.qwen_asr_dtype,
            qwen_asr_max_batch_size=args.qwen_asr_max_batch_size,
            qwen_asr_max_new_tokens=args.qwen_asr_max_new_tokens,
            device=args.asr_device,
            language="",
        )
        self.model = module.ASRBackend(legacy_args)
        self.model_id = args.qwen_asr_model

    def transcribe(self, path: str, language: str) -> str:
        return str(self.model.transcribe(path, language=language) or "").strip()

    def metadata(self) -> dict[str, Any]:
        return {"implementation": "legacy_001017_qwen3_asr", "model_id": self.model_id}


def build_asr_backend(backend: str, args: argparse.Namespace) -> Any:
    if backend == "paraformer_zh":
        return ParaformerASR(args)
    if backend == "whisper_large_v3":
        return WhisperLargeV3ASR(args)
    if backend == "qwen_asr":
        return LegacyQwenASR(args)
    raise ValueError(backend)


def score_speaker_backend(
    cases: list[dict[str, Any]], backend: str, args: argparse.Namespace
) -> None:
    def has_reusable(case: dict[str, Any]) -> bool:
        result = case.get("speaker_similarity", {}).get(backend)
        return isinstance(result, dict) and str(result.get("status")) in {"ok", "precomputed"}

    pending = [
        case
        for case in cases
        if not (args.reuse_existing and has_reusable(case))
    ]
    if not pending:
        return
    if args.schema_only:
        for case in pending:
            case["speaker_similarity"][backend] = backend_status("pending", backend)
        return
    try:
        scorer = build_speaker_scorer(backend, args)
    except Exception as exc:
        if not args.continue_on_error:
            raise
        error = f"{type(exc).__name__}: {exc}"
        for case in pending:
            case["speaker_similarity"][backend] = backend_status(
                "backend_unavailable", backend, error=error
            )
        return
    metadata = scorer.metadata()
    for case in pending:
        audio = case["audio"]
        generated = str(audio.get("generated") or "")
        reference = str(audio.get("reference") or "")
        source = str(audio.get("source") or "")
        if not generated or not reference:
            case["speaker_similarity"][backend] = backend_status(
                "missing_audio",
                backend,
                **metadata,
                sim_ref=None,
                sim_src=None,
                error="generated and reference audio are required",
            )
            continue
        try:
            sim_ref = finite(scorer.similarity(generated, reference))
            sim_src = finite(scorer.similarity(generated, source)) if source else None
            if sim_ref is None or (source and sim_src is None):
                raise ValueError("speaker scorer returned a non-finite value")
            case["speaker_similarity"][backend] = backend_status(
                "ok", backend, **metadata, sim_ref=sim_ref, sim_src=sim_src
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            case["speaker_similarity"][backend] = backend_status(
                "error",
                backend,
                **metadata,
                sim_ref=None,
                sim_src=None,
                error=f"{type(exc).__name__}: {exc}",
            )


def score_asr_backend(cases: list[dict[str, Any]], backend: str, args: argparse.Namespace) -> None:
    def has_reusable(case: dict[str, Any]) -> bool:
        result = case.get("content_asr", {}).get(backend)
        return isinstance(result, dict) and str(result.get("status")) in {"ok", "precomputed"}

    pending = [case for case in cases if not (args.reuse_existing and has_reusable(case))]
    if not pending:
        return
    applicable: list[dict[str, Any]] = []
    for case in pending:
        if not asr_is_applicable(backend, str(case.get("language") or "unknown")):
            case["content_asr"][backend] = backend_status(
                "skipped_language",
                backend,
                language=case.get("language"),
                metric_profile=metric_profile_id(args.metric_profile),
            )
        else:
            applicable.append(case)
    if args.schema_only:
        for case in applicable:
            case["content_asr"][backend] = backend_status(
                "pending",
                backend,
                language=case.get("language"),
                metric_profile=metric_profile_id(args.metric_profile),
            )
        return
    try:
        model = build_asr_backend(backend, args)
    except Exception as exc:
        if not args.continue_on_error:
            raise
        error = f"{type(exc).__name__}: {exc}"
        for case in applicable:
            case["content_asr"][backend] = backend_status(
                "backend_unavailable",
                backend,
                language=case.get("language"),
                metric_profile=metric_profile_id(args.metric_profile),
                error=error,
            )
        return
    metadata = model.metadata()
    for case in applicable:
        generated = str(case["audio"].get("generated") or "")
        reference_text = str(case.get("reference_text") or "")
        language = str(case.get("language") or "unknown")
        if not generated:
            case["content_asr"][backend] = backend_status(
                "missing_audio",
                backend,
                **metadata,
                language=language,
                metric_profile=metric_profile_id(args.metric_profile),
                error="generated audio is required",
            )
            continue
        if not reference_text:
            case["content_asr"][backend] = backend_status(
                "missing_reference",
                backend,
                **metadata,
                language=language,
                metric_profile=metric_profile_id(args.metric_profile),
                error="reference text is required",
            )
            continue
        try:
            hypothesis = model.transcribe(generated, language)
            error_payload = error_rate_payload(
                hypothesis, reference_text, language, args.metric_profile
            )
            case["content_asr"][backend] = backend_status(
                "ok",
                backend,
                **metadata,
                language=language,
                hypothesis=hypothesis,
                reference_text=reference_text,
                **error_payload,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            case["content_asr"][backend] = backend_status(
                "error",
                backend,
                **metadata,
                language=language,
                metric_profile=metric_profile_id(args.metric_profile),
                error=f"{type(exc).__name__}: {exc}",
            )


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    clean = [value for value in (finite(item) for item in values) if value is not None]
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean) if clean else None,
        "std": statistics.pstdev(clean) if len(clean) >= 2 else 0.0 if clean else None,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        speaker: dict[str, Any] = {}
        for backend in SPEAKER_BACKENDS:
            raw_results = [row.get("speaker_similarity", {}).get(backend) for row in rows]
            results = [result for result in raw_results if isinstance(result, dict)]
            speaker[backend] = {
                "status_counts": dict(
                    Counter(
                        str(result.get("status") or "unknown")
                        if isinstance(result, dict)
                        else "not_present"
                        for result in raw_results
                    )
                ),
                "implementation_counts": dict(
                    Counter(
                        str(result.get("implementation"))
                        for result in results
                        if result.get("implementation")
                    )
                ),
                "model_id_counts": dict(
                    Counter(
                        str(result.get("model_id"))
                        for result in results
                        if result.get("model_id")
                    )
                ),
                "sim_ref": numeric_summary(result.get("sim_ref") for result in results),
                "sim_src": numeric_summary(result.get("sim_src") for result in results),
            }
        content: dict[str, Any] = {}
        for backend in ASR_BACKENDS:
            raw_results = [row.get("content_asr", {}).get(backend) for row in rows]
            results = [result for result in raw_results if isinstance(result, dict)]
            content[backend] = {
                "status_counts": dict(
                    Counter(
                        str(result.get("status") or "unknown")
                        if isinstance(result, dict)
                        else "not_present"
                        for result in raw_results
                    )
                ),
                "metric_profile_counts": dict(
                    Counter(
                        str(result.get("metric_profile"))
                        for result in results
                        if result.get("metric_profile")
                    )
                ),
                "implementation_counts": dict(
                    Counter(
                        str(result.get("implementation"))
                        for result in results
                        if result.get("implementation")
                    )
                ),
                "model_id_counts": dict(
                    Counter(
                        str(result.get("model_id"))
                        for result in results
                        if result.get("model_id")
                    )
                ),
                "primary_error": numeric_summary(result.get("primary_error") for result in results),
                "cer": numeric_summary(result.get("cer") for result in results),
                "wer": numeric_summary(result.get("wer") for result in results),
            }
        return {"n_cases": len(rows), "speaker_similarity": speaker, "content_asr": content}

    groups: dict[str, list[dict[str, Any]]] = {"all": list(cases)}
    by_system_test: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_system_test_lang: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        system = str(case.get("system_id") or "unknown")
        test_set = str(case.get("test_set_id") or "unknown")
        language = str(case.get("language") or "unknown")
        by_system_test[(system, test_set)].append(case)
        by_system_test_lang[(system, test_set, language)].append(case)
    for (system, test_set), rows in sorted(by_system_test.items()):
        groups[f"system_test_set:{system}:{test_set}"] = rows
    for (system, test_set, language), rows in sorted(by_system_test_lang.items()):
        groups[f"system_test_set_language:{system}:{test_set}:{language}"] = rows
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "vc_eval_summary",
        "groups": {name: summarize_group(rows) for name, rows in groups.items()},
    }


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "" if number is None else f"{number:.{digits}f}"


def render_summary_md(summary: dict[str, Any]) -> str:
    groups = summary["groups"]
    selected = [
        (name, payload)
        for name, payload in groups.items()
        if name.startswith("system_test_set:")
    ]
    lines = [
        "# Unified VC Evaluation",
        "",
        f"schema: `{SCHEMA_VERSION}`",
        "",
        "## Speaker similarity",
        "",
        "| system | test set | scorer | n | SIM(ref) | SIM(src) | implementation | statuses |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for name, payload in selected:
        _, system, test_set = name.split(":", 2)
        for backend in SPEAKER_BACKENDS:
            item = payload["speaker_similarity"][backend]
            lines.append(
                f"| {system} | {test_set} | {backend} | {item['sim_ref']['n']} | "
                f"{fmt(item['sim_ref']['mean'])} | {fmt(item['sim_src']['mean'])} | "
                f"`{json.dumps(item['implementation_counts'], ensure_ascii=False, sort_keys=True)}` | "
                f"`{json.dumps(item['status_counts'], ensure_ascii=False, sort_keys=True)}` |"
            )
    lines.extend(
        [
            "",
            "## Content ASR",
            "",
            "| system | test set | ASR | n | primary error | CER | WER | metric profile | statuses |",
            "|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for name, payload in selected:
        _, system, test_set = name.split(":", 2)
        for backend in ASR_BACKENDS:
            item = payload["content_asr"][backend]
            lines.append(
                f"| {system} | {test_set} | {backend} | {item['primary_error']['n']} | "
                f"{fmt(item['primary_error']['mean'])} | {fmt(item['cer']['mean'])} | "
                f"{fmt(item['wer']['mean'])} | "
                f"`{json.dumps(item['metric_profile_counts'], ensure_ascii=False, sort_keys=True)}` | "
                f"`{json.dumps(item['status_counts'], ensure_ascii=False, sort_keys=True)}` |"
            )
    lines.append("")
    return "\n".join(lines)


def flatten_case(case: dict[str, Any]) -> dict[str, Any]:
    audio = case.get("audio") or {}
    row: dict[str, Any] = {
        "schema_version": case.get("schema_version"),
        "run_id": case.get("run_id"),
        "system_id": case.get("system_id"),
        "test_set_id": case.get("test_set_id"),
        "case_id": case.get("case_id"),
        "language": case.get("language"),
        "generated_audio": audio.get("generated"),
        "reference_audio": audio.get("reference"),
        "source_audio": audio.get("source"),
        "reference_text": case.get("reference_text"),
    }
    for backend in SPEAKER_BACKENDS:
        result = case.get("speaker_similarity", {}).get(backend, {})
        prefix = f"speaker__{backend}__"
        for key in ("status", "implementation", "model_id", "sim_ref", "sim_src", "error"):
            row[prefix + key] = result.get(key)
    for backend in ASR_BACKENDS:
        result = case.get("content_asr", {}).get(backend, {})
        prefix = f"asr__{backend}__"
        for key in (
            "status",
            "implementation",
            "model_id",
            "hypothesis",
            "metric_profile",
            "metric",
            "primary_error",
            "cer",
            "wer",
            "error",
        ):
            row[prefix + key] = result.get(key)
    return row


def write_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    rows = [flatten_case(case) for case in cases]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else list(flatten_case({}).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(args.output_dir).expanduser()
    stem = args.output_stem or args.run_id or "unified_vc_eval"
    if not args.output_stem and getattr(args, "num_shards", 1) > 1:
        stem += (
            f".shard-{int(args.shard_index):05d}-of-"
            f"{int(args.num_shards):05d}"
        )
    return (
        output_dir / f"{stem}.unified_eval.jsonl",
        output_dir / f"{stem}.unified_eval.csv",
        output_dir / f"{stem}.summary.json",
        output_dir / f"{stem}.summary.md",
    )


def write_all_outputs(cases: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    jsonl_path, csv_path, summary_json, summary_md = resolve_output_paths(args)
    write_jsonl(jsonl_path, cases)
    write_csv(csv_path, cases)
    summary = summarize_cases(cases)
    summary["run_id"] = args.run_id
    summary["per_case_jsonl"] = str(jsonl_path)
    summary["per_case_csv"] = str(csv_path)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary_md.write_text(render_summary_md(summary), encoding="utf-8")
    return {
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }


def merge_metric_result(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return dict(incoming)
    if existing == incoming:
        return existing
    weak = {"pending", "backend_unavailable", "error", "missing_audio", "missing_reference"}
    existing_status = str(existing.get("status") or "")
    incoming_status = str(incoming.get("status") or "")
    if existing_status in weak and incoming_status not in weak:
        return dict(incoming)
    if incoming_status in weak and existing_status not in weak:
        return existing
    raise ValueError(
        "conflicting concrete metric results: "
        f"existing={json.dumps(existing, ensure_ascii=False, sort_keys=True)} "
        f"incoming={json.dumps(incoming, ensure_ascii=False, sort_keys=True)}"
    )


def merge_cases(paths: Sequence[Path], *, run_id: str) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        for row in iter_records(path):
            if row.get("schema_version") != SCHEMA_VERSION or row.get("record_type") != "vc_eval_case":
                raise ValueError(f"{path}: not a {SCHEMA_VERSION} vc_eval_case record")
            validate_metric_sections(row, context=str(path))
            key = (
                str(row.get("system_id") or ""),
                str(row.get("test_set_id") or ""),
                str(row.get("case_id") or ""),
            )
            if key not in merged:
                merged[key] = dict(row)
                merged[key]["run_id"] = run_id or row.get("run_id")
                merged[key].setdefault("provenance", {})["merged_from"] = [str(path)]
                continue
            target = merged[key]
            identity_fields = ("language", "reference_text", "audio")
            for field in identity_fields:
                if target.get(field) != row.get(field):
                    raise ValueError(f"{path}: conflicting {field} for case key {key}")
            for section in ("speaker_similarity", "content_asr"):
                target.setdefault(section, {})
                incoming_section = row.get(section) or {}
                for backend, result in incoming_section.items():
                    target[section][backend] = merge_metric_result(target[section].get(backend), result)
            target.setdefault("provenance", {}).setdefault("merged_from", []).append(str(path))
    return [merged[key] for key in sorted(merged)]


def dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    def module_available(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    speechbrain_deps = Path(
        "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/python_deps/speechbrain_py312"
    )
    report = {
        "python": sys.executable,
        "schema_version": SCHEMA_VERSION,
        "backends": {
            "wavlm_large_sv": {
                "implementation": args.wavlm_implementation,
                "python_modules": {
                    "torch": module_available("torch"),
                    "transformers": module_available("transformers"),
                    "librosa": module_available("librosa"),
                },
                "speaker_similarity_py": str(Path(args.speaker_sim_root) / "speaker_similarity.py"),
                "speaker_similarity_py_exists": (Path(args.speaker_sim_root) / "speaker_similarity.py").exists(),
                "seedtts_checkpoint_exists": Path(args.wavlm_checkpoint).exists() if args.wavlm_checkpoint else None,
                "seedtts_eval_root_exists": Path(args.seedtts_eval_root).exists() if args.seedtts_eval_root else None,
                "wavlm_model_dir_exists": Path(args.wavlm_model_dir).exists() if args.wavlm_model_dir else None,
                "hf_model_id": args.wavlm_hf_model,
            },
            "eres2net": {
                "python_modules": {
                    "modelscope": module_available("modelscope"),
                    "addict": module_available("addict"),
                },
                "model_id": args.eres2net_model,
                "note": (
                    "modelscope.pipelines also requires addict; remote/cache model availability is checked "
                    "only when ModelScope loads the adapter"
                ),
            },
            "speechbrain_ecapa": {
                "python_modules": {"speechbrain": module_available("speechbrain")},
                "managed_dependency_dir": str(speechbrain_deps),
                "managed_dependency_dir_exists": speechbrain_deps.exists(),
                "available_via_managed_dependency_dir": (speechbrain_deps / "speechbrain").is_dir(),
                "model": args.speechbrain_model,
                "model_exists": Path(args.speechbrain_model).exists(),
            },
            "paraformer_zh": {
                "python_modules": {
                    "funasr": module_available("funasr"),
                    "zhconv": module_available("zhconv"),
                },
                "model_id": args.paraformer_model,
            },
            "whisper_large_v3": {
                "python_modules": {
                    "torch": module_available("torch"),
                    "transformers": module_available("transformers"),
                },
                "model_id": args.whisper_model,
            },
            "qwen_asr": {
                "python_modules": {"qwen_asr": module_available("qwen_asr")},
                "model": args.qwen_asr_model,
                "model_exists": Path(args.qwen_asr_model).exists(),
                "legacy_adapter": str(ROOT / "scripts/001017_asr_content_filter.py"),
            },
        },
    }
    return report


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--speaker-device", default="cuda:0")
    parser.add_argument("--asr-device", default="cuda:0")
    parser.add_argument("--speaker-sim-root", default=str(DEFAULT_SPEAKER_SIM_ROOT))
    parser.add_argument(
        "--wavlm-implementation",
        choices=("seedtts_official", "hf_xvector"),
        default="seedtts_official",
        help="Official Seed-TTS WavLM-Large+ECAPA is the paper-facing default; HF is a fallback scorer.",
    )
    parser.add_argument(
        "--wavlm-checkpoint", default=str(DEFAULT_SEEDTTS_MODEL_ROOT / "wavlm_large_finetune.pth")
    )
    parser.add_argument(
        "--seedtts-eval-root", default=str(DEFAULT_SEEDTTS_MODEL_ROOT / "seed-tts-eval")
    )
    parser.add_argument("--wavlm-model-dir", default=str(DEFAULT_SEEDTTS_MODEL_ROOT / "wavlm-large"))
    parser.add_argument("--wavlm-hf-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument(
        "--eres2net-model", default="iic/speech_eres2net_sv_zh-cn_16k-common"
    )
    parser.add_argument("--speechbrain-model", default=str(DEFAULT_SPEECHBRAIN_MODEL))
    parser.add_argument(
        "--paraformer-model",
        default="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )
    parser.add_argument("--whisper-model", default="openai/whisper-large-v3")
    parser.add_argument("--qwen-asr-model", default=str(DEFAULT_QWEN_ASR_MODEL))
    parser.add_argument(
        "--qwen-asr-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--qwen-asr-max-batch-size", type=int, default=1)
    parser.add_argument("--qwen-asr-max-new-tokens", type=int, default=256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or merge the Batch-42 unified VC speaker/content evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Normalize cases and optionally execute selected backends.")
    evaluate.add_argument("--input", required=True, help="Input JSONL or CSV with case/audio/text fields.")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--output-stem", default="")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--system-id", default="", help="Override per-row system ID.")
    evaluate.add_argument("--test-set-id", default="", help="Override per-row test-set ID.")
    evaluate.add_argument(
        "--input-profile",
        choices=INPUT_PROFILES,
        default="auto",
        help=(
            "Use official_seedtts_vc for paper-facing VC runs; it requires explicit "
            "generated/reference/source audio fields and rejects ambiguous target_audio mapping."
        ),
    )
    evaluate.add_argument(
        "--metric-profile",
        choices=METRIC_PROFILES,
        default="seedtts_official",
        help="ASR text normalization/error profile; paper-facing default matches Seed-TTS-Eval run_wer.py.",
    )
    evaluate.add_argument(
        "--speaker-scorer",
        action="append",
        default=[],
        help="Repeat or comma-separate; choices are wavlm_large_sv, eres2net, speechbrain_ecapa, all.",
    )
    evaluate.add_argument(
        "--asr-backend",
        action="append",
        default=[],
        help="Repeat or comma-separate; choices are paraformer_zh, whisper_large_v3, qwen_asr, all.",
    )
    evaluate.add_argument(
        "--legacy-asr-backend",
        default="qwen_asr",
        help="Backend assigned to legacy asr_tgt_text/cer_tgt/wer_tgt fields when no label is present.",
    )
    evaluate.add_argument(
        "--schema-only",
        action="store_true",
        help="Never import/load models; mark missing metrics pending.",
    )
    evaluate.add_argument("--continue-on-error", action="store_true")
    evaluate.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false")
    evaluate.set_defaults(reuse_existing=True)
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--num-shards", type=int, default=1)
    evaluate.add_argument("--shard-index", type=int, default=0)
    add_backend_args(evaluate)

    merge = subparsers.add_parser("merge", help="Merge independently scored canonical partial JSONLs.")
    merge.add_argument("--partial", action="append", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--output-stem", default="")
    merge.add_argument("--run-id", required=True)

    check = subparsers.add_parser("check", help="Report dependency/model-path readiness without loading a model.")
    check.add_argument("--output-json", default="")
    add_backend_args(check)
    return parser


def command_evaluate(args: argparse.Namespace) -> int:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards>=1 and 0<=shard_index<num_shards")
    input_path = Path(args.input).expanduser()
    raw_rows = list(iter_records(input_path))
    selected = [
        (index, row)
        for index, row in enumerate(raw_rows)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit > 0:
        selected = selected[: args.limit]
    cases = [
        normalize_case(
            row,
            input_index=index,
            input_path=input_path,
            run_id=args.run_id,
            system_override=args.system_id,
            test_set_override=args.test_set_id,
            legacy_asr_backend=args.legacy_asr_backend,
            input_profile=args.input_profile,
        )
        for index, row in selected
    ]
    identities = [
        (
            str(case.get("system_id") or ""),
            str(case.get("test_set_id") or ""),
            str(case.get("case_id") or ""),
        )
        for case in cases
    ]
    duplicate_identities = [key for key, count in Counter(identities).items() if count > 1]
    if duplicate_identities:
        raise ValueError(
            "duplicate (system_id, test_set_id, case_id) identities in selected input: "
            + repr(duplicate_identities[:5])
        )
    target_audio_alias_rows = sum(
        case.get("provenance", {}).get("audio_field_mapping", {}).get("generated")
        == "target_audio"
        for case in cases
    )
    if target_audio_alias_rows and args.input_profile == "auto":
        print(
            "[unified-vc-eval] WARNING: "
            f"{target_audio_alias_rows} rows map legacy target_audio to generated audio. "
            "This is valid for historical internal outputs but forbidden for official "
            "Seed-TTS VC; use --input-profile official_seedtts_vc there.",
            file=sys.stderr,
        )
    speaker_backends = parse_backend_list(args.speaker_scorer, kind="speaker")
    asr_backends = parse_backend_list(args.asr_backend, kind="asr")
    for backend in speaker_backends:
        score_speaker_backend(cases, backend, args)
    for backend in asr_backends:
        score_asr_backend(cases, backend, args)
    outputs = write_all_outputs(cases, args)
    print(
        json.dumps(
            {
                "command": "evaluate",
                "input_rows": len(raw_rows),
                "selected_rows": len(cases),
                "schema_only": args.schema_only,
                "input_profile": args.input_profile,
                "metric_profile": args.metric_profile,
                "legacy_target_audio_alias_rows": target_audio_alias_rows,
                "speaker_backends": speaker_backends,
                "asr_backends": asr_backends,
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_merge(args: argparse.Namespace) -> int:
    paths = [Path(value).expanduser() for value in args.partial]
    cases = merge_cases(paths, run_id=args.run_id)
    outputs = write_all_outputs(cases, args)
    print(
        json.dumps(
            {"command": "merge", "partials": [str(path) for path in paths], "cases": len(cases), "outputs": outputs},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_check(args: argparse.Namespace) -> int:
    report = dependency_report(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        output = Path(args.output_json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        return command_evaluate(args)
    if args.command == "merge":
        return command_merge(args)
    if args.command == "check":
        return command_check(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
