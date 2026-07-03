#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import stable_id


DEFAULT_NO_TEXT_INSTRUCTION = (
    "Voice conversion task. [S1] is the source speech carrying content, pauses, duration and prosody. "
    "[S2] is the target timbre reference. Generate the same content as S1 with S2 timbre while preserving "
    "S1 timing and prosody."
)

DEFAULT_TEXT_PROSODY_INSTRUCTION = (
    "Text-guided voice conversion task. Use the provided text as lexical content. Use S1 only as a "
    "prosody/style reference for rhythm, pauses, speaking rate, stress and duration hints. Use S2 as the "
    "target timbre reference. Do not copy S1 speaker identity or S1 words."
)

DEFAULT_AUDIO_KEYS = (
    "audio_path",
    "local_path",
    "audio",
    "wav",
    "wav_path",
    "ref_audio",
    "ref_audio_path",
    "original_audio",
    "original_audio_path",
)

DEFAULT_TEXT_KEYS = (
    "mtd_transcript",
    "text",
    "transcript",
    "mtd_asr_text",
    "ref_text",
    "original_text",
    "caption",
)

DEFAULT_SPEAKER_KEYS = (
    "speaker_id",
    "spk_id",
    "speaker",
    "speaker_name",
    "utt_speaker",
)

DEFAULT_GENDER_KEYS = (
    "gender",
    "speaker_gender",
    "sex",
)

PAIR_NO_TEXT = "no_text"
PAIR_TEXT_PROSODY = "text_prosody"
PAIR_TEXT_ALIGNED_DIAGNOSTIC = "text_aligned_diagnostic"


@dataclass(slots=True)
class Utterance:
    uid: str
    audio: str
    text: str
    language: str
    speaker_id: str
    gender: str
    duration: float | None
    dataset: str
    input_jsonl: str
    input_line: int
    raw_id: str
    source_jsonl: str
    source_line: int | None
    row_source: str
    speaker_source: str
    gender_source: str


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def pick_first(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\[(?:S\d+|\d+(?:\.\d+)?)\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compare_text_key(text: str, *, strip_punctuation: bool) -> str:
    text = normalize_text(text).lower()
    if strip_punctuation:
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", "", text)


def normalize_language(value: Any, text: str) -> str:
    language = str(value or "").strip().lower()
    if language in {"zh", "zho", "chinese", "cmn", "mandarin"}:
        return "zh"
    if language in {"en", "eng", "english"}:
        return "en"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    return language or "unknown"


def normalize_gender(value: Any) -> str:
    if value is None:
        return "unknown"
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    if not text:
        return "unknown"
    if text in {"f", "female", "woman", "girl", "feminine", "女", "女性", "女生"}:
        return "female"
    if text in {"m", "male", "man", "boy", "masculine", "男", "男性", "男生"}:
        return "male"
    if "female" in text or "woman" in text or "girl" in text or "女性" in text or "女" in text:
        return "female"
    if "male" in text or "man" in text or "boy" in text or "男性" in text or "男" in text:
        return "male"
    return "unknown"


def parse_caption_result(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def extract_gender(row: dict[str, Any], gender_keys: Iterable[str]) -> tuple[str, str]:
    direct = pick_first(row, gender_keys)
    gender = normalize_gender(direct)
    if gender != "unknown":
        return gender, "field"
    caption = parse_caption_result(row.get("caption_result"))
    gender = normalize_gender(pick_first(caption, gender_keys) or caption.get("gender"))
    if gender != "unknown":
        return gender, "caption_result"
    thought = row.get("thought") or row.get("zh_summary") or row.get("zh_summary_bak")
    gender = normalize_gender(thought)
    if gender != "unknown":
        return gender, "caption_text"
    return "unknown", "missing"


def parse_duration(row: dict[str, Any]) -> float | None:
    value = pick_first(row, ("duration_sec", "duration", "actual_duration_seconds", "audio_duration", "seconds"))
    if value in (None, ""):
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration >= 0 else None


def resolve_audio_path(value: Any, audio_root: Path | None) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (audio_root or Path.cwd()) / path
    return str(path.resolve(strict=False))


def infer_filename_hash(row: dict[str, Any], audio: str) -> str:
    candidates = [audio, str(row.get("audio_path") or ""), str(row.get("local_path") or ""), str(row.get("object_name") or "")]
    for value in candidates:
        if not value:
            continue
        stem = Path(value).stem
        match = re.search(r"-([0-9a-fA-F]{24,40})-\d+$", stem)
        if match:
            return match.group(1).lower()
        object_match = re.search(r"/([0-9a-fA-F]{24,40})/[^/]+$", value)
        if object_match:
            return object_match.group(1).lower()
    return ""


def infer_speaker_id(row: dict[str, Any], audio: str, language: str, mode: str, speaker_keys: list[str]) -> tuple[str, str]:
    explicit = pick_first(row, speaker_keys)
    if explicit not in (None, ""):
        return str(explicit), "field"
    filename_hash = infer_filename_hash(row, audio)
    if mode in {"auto", "filename_hash"} and filename_hash:
        source = str(row.get("source") or row.get("dataset") or "unknown")
        return f"filename_hash:{language}:{source}:{filename_hash}", "filename_hash"
    if mode == "none":
        return "", "missing"
    if mode == "id":
        raw_id = row.get("id")
        if raw_id not in (None, ""):
            return f"id:{raw_id}", "id"
        return "", "missing"
    path = Path(audio)
    if mode in {"auto", "path_parent"}:
        parent = path.parent.name
        source = str(row.get("source") or row.get("dataset") or "unknown")
        return f"path_parent:{language}:{source}:{parent}", "path_parent"
    if mode == "path_grandparent_parent":
        parent = path.parent.name
        grandparent = path.parent.parent.name if path.parent.parent else "unknown"
        source = str(row.get("source") or row.get("dataset") or "unknown")
        return f"path_group:{language}:{source}:{grandparent}:{parent}", "path_grandparent_parent"
    return "", "missing"


def expand_input_paths(values: list[str], languages: set[str], include_top_level_jsonl: bool) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for raw_value in values:
        for item in split_csv(raw_value):
            expanded = Path(item).expanduser()
            matches: list[Path]
            if any(ch in item for ch in "*?[]"):
                matches = [Path(p) for p in glob.glob(item, recursive=True)]
            elif expanded.is_dir():
                language_dirs = [expanded / lang for lang in sorted(languages) if (expanded / lang).is_dir()]
                if language_dirs:
                    matches = []
                    for lang_dir in language_dirs:
                        matches.extend(sorted(lang_dir.glob("*.jsonl")))
                    if include_top_level_jsonl:
                        matches.extend(sorted(expanded.glob("*.jsonl")))
                else:
                    matches = sorted(expanded.rglob("*.jsonl"))
            else:
                matches = [expanded]
            for match in matches:
                resolved = match.resolve(strict=False)
                key = str(resolved)
                if key not in seen:
                    seen.add(key)
                    paths.append(resolved)
    return sorted(paths)


def iter_jsonl_with_line(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def normalize_row(
    row: dict[str, Any],
    *,
    input_path: Path,
    input_line: int,
    global_index: int,
    audio_root: Path | None,
    audio_keys: list[str],
    text_keys: list[str],
    speaker_keys: list[str],
    gender_keys: list[str],
    infer_speaker_from: str,
    allow_missing_audio: bool,
    allow_missing_speaker: bool,
    min_duration: float,
    max_duration: float,
    stats: Counter,
) -> Utterance | None:
    audio_value = pick_first(row, audio_keys)
    if audio_value in (None, ""):
        stats["missing_audio"] += 1
        return None
    audio = resolve_audio_path(audio_value, audio_root)
    if not allow_missing_audio and not Path(audio).exists():
        stats["missing_audio"] += 1
        return None

    duration = parse_duration(row)
    if duration is not None and min_duration > 0 and duration < min_duration:
        stats["too_short"] += 1
        return None
    if duration is not None and max_duration > 0 and duration > max_duration:
        stats["too_long"] += 1
        return None

    text = normalize_text(pick_first(row, text_keys))
    if not text:
        stats["missing_text"] += 1

    language = normalize_language(row.get("language"), text)
    speaker_id, speaker_source = infer_speaker_id(row, audio, language, infer_speaker_from, speaker_keys)
    if not speaker_id:
        stats["missing_speaker"] += 1
        if not allow_missing_speaker:
            return None
        speaker_id = f"unknown:{stable_id(audio, length=12)}"
        speaker_source = "synthetic_unknown"

    gender, gender_source = extract_gender(row, gender_keys)
    raw_id = str(row.get("id") or row.get("utt_id") or stable_id(audio, input_path, input_line, length=16))
    uid = stable_id(input_path, input_line, raw_id, audio, length=24)
    source_line_value = row.get("source_line")
    try:
        source_line = int(source_line_value) if source_line_value not in (None, "") else None
    except (TypeError, ValueError):
        source_line = None

    return Utterance(
        uid=uid,
        audio=audio,
        text=text,
        language=language,
        speaker_id=speaker_id,
        gender=gender,
        duration=duration,
        dataset=str(row.get("dataset") or row.get("source") or ""),
        input_jsonl=str(input_path),
        input_line=input_line,
        raw_id=raw_id,
        source_jsonl=str(row.get("source_jsonl") or ""),
        source_line=source_line,
        row_source=str(row.get("source") or row.get("dataset") or ""),
        speaker_source=speaker_source,
        gender_source=gender_source,
    )


def augment_from_source_jsonl(utterances: list[Utterance], gender_keys: list[str], text_keys: list[str]) -> dict[str, Any]:
    requests_by_file: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, utt in enumerate(utterances):
        if not utt.source_jsonl or utt.source_line is None:
            continue
        if utt.gender != "unknown" and utt.text:
            continue
        for line_no in {utt.source_line, utt.source_line - 1 if utt.source_line > 0 else utt.source_line}:
            if line_no >= 0:
                requests_by_file[utt.source_jsonl][line_no].append(idx)

    summary = {
        "requested_files": len(requests_by_file),
        "requested_rows": sum(len(lines) for lines in requests_by_file.values()),
        "gender_filled": 0,
        "text_filled": 0,
        "failed_files": 0,
    }
    for source_jsonl, line_map in requests_by_file.items():
        path = Path(source_jsonl).expanduser()
        candidates_by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
        try:
            max_line = max(line_map)
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle):
                    if line_no > max_line:
                        break
                    if line_no not in line_map:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for idx in line_map[line_no]:
                        candidates_by_idx[idx].append(row)
        except OSError:
            summary["failed_files"] += 1
            continue

        for idx, candidates in candidates_by_idx.items():
            utt = utterances[idx]
            row = choose_augmented_row(utt, candidates)
            if not row:
                continue
            if utt.gender == "unknown":
                gender, gender_source = extract_gender(row, gender_keys)
                if gender != "unknown":
                    utt.gender = gender
                    utt.gender_source = f"source_jsonl:{gender_source}"
                    summary["gender_filled"] += 1
            if not utt.text:
                text = normalize_text(pick_first(row, text_keys))
                if text:
                    utt.text = text
                    summary["text_filled"] += 1
    return summary


def choose_augmented_row(utt: Utterance, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    audio_name = Path(utt.audio).name
    for row in candidates:
        if str(row.get("id") or "") == utt.raw_id:
            return row
    for row in candidates:
        for key in ("local_path", "audio_path"):
            value = row.get(key)
            if value and Path(str(value)).name == audio_name:
                return row
        object_name = row.get("object_name")
        if object_name and Path(str(object_name)).name == audio_name:
            return row
    return candidates[0]


def build_indexes(utterances: list[Utterance]) -> dict[str, Any]:
    by_language: dict[str, list[int]] = defaultdict(list)
    by_speaker: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_gender: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_input_jsonl: dict[str, list[int]] = defaultdict(list)
    for idx, utt in enumerate(utterances):
        by_language[utt.language].append(idx)
        by_speaker[(utt.language, utt.speaker_id)].append(idx)
        by_gender[(utt.language, utt.gender)].append(idx)
        by_input_jsonl[utt.input_jsonl].append(idx)
    return {
        "by_language": by_language,
        "by_speaker": by_speaker,
        "by_gender": by_gender,
        "by_input_jsonl": by_input_jsonl,
    }


def pick_candidate(
    pool: list[int],
    rng: random.Random,
    reject,
    *,
    max_random_tries: int = 64,
) -> int | None:
    if not pool:
        return None
    for _ in range(min(max_random_tries, len(pool) * 2)):
        idx = pool[rng.randrange(len(pool))]
        if not reject(idx):
            return idx
    for idx in pool:
        if not reject(idx):
            return idx
    return None


def pick_timbre_ref(
    target_idx: int,
    utterances: list[Utterance],
    indexes: dict[str, Any],
    rng: random.Random,
    allow_cross_speaker_fallback: bool,
) -> tuple[int | None, str]:
    target = utterances[target_idx]
    same_speaker = indexes["by_speaker"].get((target.language, target.speaker_id), [])
    picked = pick_candidate(same_speaker, rng, lambda idx: idx == target_idx or utterances[idx].audio == target.audio)
    if picked is not None:
        return picked, "same_speaker"
    if not allow_cross_speaker_fallback:
        return None, "missing_same_speaker_ref"
    same_lang = indexes["by_language"].get(target.language, [])
    picked = pick_candidate(
        same_lang,
        rng,
        lambda idx: idx == target_idx or utterances[idx].audio == target.audio or utterances[idx].speaker_id == target.speaker_id,
    )
    if picked is not None:
        return picked, "cross_speaker_fallback"
    return None, "missing_cross_speaker_ref"


def pick_timbre_ref_same_jsonl(
    source_idx: int,
    utterances: list[Utterance],
    indexes: dict[str, Any],
    rng: random.Random,
) -> tuple[int | None, str]:
    source = utterances[source_idx]
    same_file = indexes["by_input_jsonl"].get(source.input_jsonl, [])
    picked = pick_candidate(same_file, rng, lambda idx: idx == source_idx or utterances[idx].audio == source.audio)
    if picked is not None:
        relation = "same_speaker" if utterances[picked].speaker_id == source.speaker_id else "same_jsonl_random"
        return picked, relation
    return None, "missing_same_jsonl_ref"


def pick_text_source(
    target_idx: int,
    timbre_idx: int,
    utterances: list[Utterance],
    indexes: dict[str, Any],
    rng: random.Random,
    *,
    policy: str,
    strict_policy: bool,
    strip_punctuation_for_compare: bool,
) -> tuple[int | None, str, bool]:
    target = utterances[target_idx]
    timbre = utterances[timbre_idx]
    same_lang = indexes["by_language"].get(target.language, [])
    target_text_key = compare_text_key(target.text, strip_punctuation=strip_punctuation_for_compare)

    def reject_base(idx: int) -> bool:
        cand = utterances[idx]
        return idx in {target_idx, timbre_idx} or cand.audio in {target.audio, timbre.audio}

    def rejects_for_policy(idx: int, *, require_text_mismatch: bool) -> bool:
        if reject_base(idx):
            return True
        cand = utterances[idx]
        if policy == "different_speaker" and cand.speaker_id == target.speaker_id:
            return True
        if policy == "different_gender":
            if target.gender == "unknown" or cand.gender == "unknown" or cand.gender == target.gender:
                return True
        if require_text_mismatch and target_text_key and compare_text_key(cand.text, strip_punctuation=strip_punctuation_for_compare) == target_text_key:
            return True
        return False

    picked = pick_candidate(same_lang, rng, lambda idx: rejects_for_policy(idx, require_text_mismatch=True))
    if picked is not None:
        return picked, policy, False
    picked = pick_candidate(same_lang, rng, lambda idx: rejects_for_policy(idx, require_text_mismatch=False))
    if picked is not None:
        return picked, f"{policy}_text_aligned_fallback", True
    if strict_policy:
        return None, f"missing_{policy}", False
    picked = pick_candidate(same_lang, rng, lambda idx: reject_base(idx) or utterances[idx].speaker_id == target.speaker_id)
    if picked is not None:
        aligned = bool(target_text_key and compare_text_key(utterances[picked].text, strip_punctuation=strip_punctuation_for_compare) == target_text_key)
        return picked, "different_speaker_fallback", aligned
    picked = pick_candidate(same_lang, rng, reject_base)
    if picked is not None:
        aligned = bool(target_text_key and compare_text_key(utterances[picked].text, strip_punctuation=strip_punctuation_for_compare) == target_text_key)
        return picked, "same_language_any_fallback", aligned
    return None, "missing_text_source", False


def make_sample_id(run_name: str, pair_type: str, source: Utterance, timbre: Utterance, target: Utterance, ordinal: int) -> str:
    digest = stable_id(pair_type, source.audio, timbre.audio, target.audio, length=12)
    return f"{run_name}:{pair_type}:{ordinal:08d}:{digest}"


def make_sample_id_from_audio(
    run_name: str,
    pair_type: str,
    source_audio: str,
    timbre_audio: str,
    target_audio: str,
    ordinal: int,
) -> str:
    digest = stable_id(pair_type, source_audio, timbre_audio, target_audio, length=12)
    return f"{run_name}:{pair_type}:{ordinal:08d}:{digest}"


def seedvc_relative_audio_path(source: Utterance, ordinal: int) -> Path:
    input_digest = stable_id(source.input_jsonl, length=10)
    input_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source.input_jsonl).stem)[:80] or "input"
    return Path(input_stem + "_" + input_digest) / f"{ordinal:08d}.wav"


def build_seedvc_job(
    *,
    run_name: str,
    source: Utterance,
    timbre: Utterance,
    output_audio: str,
    job_ordinal: int,
    seed: int,
    timbre_rule: str,
) -> dict[str, Any]:
    job_id = f"{run_name}:seedvc_v1:{job_ordinal:08d}:{stable_id(source.audio, timbre.audio, output_audio, length=12)}"
    return {
        "job_id": job_id,
        "pair_type": "ProsodyNoTimbre",
        "prosody_ref_audio": source.audio,
        "prosody_ref_text": source.text,
        "source_audio": source.audio,
        "source_text": source.text,
        "timbre_ref_audio": timbre.audio,
        "timbre_ref_text": timbre.text,
        "target_audio": output_audio,
        "output_audio": output_audio,
        "target_text": source.text,
        "language": source.language,
        "instruction": DEFAULT_NO_TEXT_INSTRUCTION,
        "source_edit_tag": "seedvc_v1_batch_timbre",
        "source_edit": "seedvc_v1_batch_timbre",
        "target_voice_profile": "batch_timbre_ref",
        "metadata": {
            "route": "prosody_no_timbre_seedvc_v1",
            "run_name": run_name,
            "source_input_jsonl": source.input_jsonl,
            "source_input_line": source.input_line,
            "timbre_ref_input_jsonl": timbre.input_jsonl,
            "timbre_ref_input_line": timbre.input_line,
            "source_speaker_id": source.speaker_id,
            "timbre_ref_speaker_id": timbre.speaker_id,
            "source_gender": source.gender,
            "timbre_ref_gender": timbre.gender,
            "timbre_rule": timbre_rule,
            "backend": "seed_vc_v1_zero_shot_voice_conversion",
            "seed": seed,
            "note": "Source audio supplies lexical content, timing, pauses, rhythm and prosody; timbre_ref_audio supplies target timbre.",
        },
    }


def build_row(
    *,
    run_name: str,
    pair_type: str,
    source: Utterance,
    timbre: Utterance,
    target: Utterance,
    ordinal: int,
    construction_rule: str,
    text_alignment: str,
    target_audio_override: str | None = None,
    target_text_override: str | None = None,
    target_speaker: Utterance | None = None,
    target_duration_override: float | None = None,
    seedvc_job_id: str | None = None,
) -> dict[str, Any]:
    target_audio = target_audio_override or target.audio
    target_text = target_text_override if target_text_override is not None else target.text
    target_identity = target_speaker or target
    return {
        "sample_id": make_sample_id_from_audio(run_name, pair_type, source.audio, timbre.audio, target_audio, ordinal),
        "source_audio": source.audio,
        "source_text": source.text,
        "timbre_ref_audio": timbre.audio,
        "timbre_ref_text": timbre.text,
        "target_audio": target_audio,
        "target_text": target_text,
        "language": source.language,
        "source_speaker_id": source.speaker_id,
        "timbre_ref_speaker_id": timbre.speaker_id,
        "target_speaker_id": target_identity.speaker_id,
        "source_gender": source.gender,
        "timbre_ref_gender": timbre.gender,
        "target_gender": target_identity.gender,
        "pair_type": pair_type,
        "instruction": DEFAULT_NO_TEXT_INSTRUCTION,
        "text_prosody_instruction": DEFAULT_TEXT_PROSODY_INSTRUCTION,
        "preferred_emit_mode": "no_text" if pair_type == PAIR_NO_TEXT else "text",
        "meta": {
            "source_dataset": source.dataset,
            "timbre_ref_dataset": timbre.dataset,
            "target_dataset": target.dataset,
            "construction_rule": construction_rule,
            "text_alignment": text_alignment,
            "source_duration": source.duration,
            "timbre_ref_duration": timbre.duration,
            "target_duration": target_duration_override if target_duration_override is not None else target.duration,
            "source_input_jsonl": source.input_jsonl,
            "source_input_line": source.input_line,
            "timbre_ref_input_jsonl": timbre.input_jsonl,
            "timbre_ref_input_line": timbre.input_line,
            "target_input_jsonl": target.input_jsonl,
            "target_input_line": target.input_line,
            "source_speaker_source": source.speaker_source,
            "timbre_ref_speaker_source": timbre.speaker_source,
            "target_speaker_source": target_identity.speaker_source,
            "source_gender_source": source.gender_source,
            "timbre_ref_gender_source": timbre.gender_source,
            "target_gender_source": target_identity.gender_source,
            "target_audio_backend": "seed_vc_v1_zero_shot_voice_conversion" if target_audio_override else "raw_audio",
            "seedvc_job_id": seedvc_job_id,
        },
    }


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a large mixed-mode VC wav-triple manifest from raw utterance JSONL files."
    )
    parser.add_argument("--input-jsonl", action="append", required=True, help="Input JSONL path, directory, glob, or comma list.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--audio-root", default="", help="Root used when raw audio paths are relative.")
    parser.add_argument("--run-name", default="large_mixed_mode")
    parser.add_argument("--languages", default="zh,en", help="Comma-separated language filter. Empty disables filtering.")
    parser.add_argument("--include-top-level-jsonl", type=str_to_bool, default=False)
    parser.add_argument("--audio-keys", default=",".join(DEFAULT_AUDIO_KEYS))
    parser.add_argument("--text-keys", default=",".join(DEFAULT_TEXT_KEYS))
    parser.add_argument("--speaker-keys", default=",".join(DEFAULT_SPEAKER_KEYS))
    parser.add_argument("--gender-keys", default=",".join(DEFAULT_GENDER_KEYS))
    parser.add_argument("--emit-pair-types", default="no_text,text_prosody")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--allow-missing-audio", type=str_to_bool, default=False)
    parser.add_argument("--allow-missing-speaker", type=str_to_bool, default=False)
    parser.add_argument(
        "--infer-speaker-from",
        choices=("auto", "filename_hash", "path_parent", "path_grandparent_parent", "id", "none"),
        default="auto",
    )
    parser.add_argument("--augment-source-jsonl", type=str_to_bool, default=True)
    parser.add_argument("--strip-punctuation-for-compare", type=str_to_bool, default=False)
    parser.add_argument(
        "--text-source-policy",
        choices=("different_speaker", "different_gender", "same_language_any"),
        default="different_speaker",
    )
    parser.add_argument("--strict-text-source-policy", type=str_to_bool, default=False)
    parser.add_argument("--allow-cross-speaker-timbre-ref", type=str_to_bool, default=False)
    parser.add_argument(
        "--no-text-source-mode",
        choices=("seedvc",),
        default="seedvc",
        help="seedvc uses the raw utterance as source/prosody and writes target_audio as the planned Seed-VC output.",
    )
    parser.add_argument(
        "--target-audio-mode",
        choices=("seedvc",),
        default="seedvc",
        help="How target_audio is obtained. seedvc means target_audio must be generated by Seed-VC from source_audio+timbre_ref_audio.",
    )
    parser.add_argument(
        "--seedvc-output-root",
        default="",
        help="Root directory for planned Seed-VC target wavs. Defaults to <output-jsonl-dir>/seedvc_targets/<run-name>.",
    )
    parser.add_argument("--seedvc-audio-subdir", default="audio/seedvc_v1")
    parser.add_argument("--seedvc-jobs-jsonl", default="", help="Optional output JSONL of Seed-VC conversion jobs.")
    parser.add_argument(
        "--require-target-audio",
        type=str_to_bool,
        default=False,
        help="If true, only write rows whose planned Seed-VC target_audio already exists.",
    )
    parser.add_argument(
        "--min-target-audio-bytes",
        type=int,
        default=1,
        help="Minimum target_audio file size when --require-target-audio is true. Use 4096 to skip interrupted/empty wavs.",
    )
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    languages = set(split_csv(args.languages))
    audio_keys = split_csv(args.audio_keys)
    text_keys = split_csv(args.text_keys)
    speaker_keys = split_csv(args.speaker_keys)
    gender_keys = split_csv(args.gender_keys)
    emit_pair_types = split_csv(args.emit_pair_types)
    unsupported = sorted(set(emit_pair_types) - {PAIR_NO_TEXT, PAIR_TEXT_PROSODY, PAIR_TEXT_ALIGNED_DIAGNOSTIC})
    if unsupported:
        raise SystemExit(f"unsupported --emit-pair-types values: {unsupported}")

    input_paths = expand_input_paths(args.input_jsonl, languages or {"zh", "en"}, args.include_top_level_jsonl)
    if args.max_files > 0:
        input_paths = input_paths[: args.max_files]
    if not input_paths:
        raise SystemExit("no input JSONL files found")

    audio_root = Path(args.audio_root).expanduser().resolve(strict=False) if args.audio_root else None
    stats: Counter = Counter(
        {
            "read_rows": 0,
            "valid_utterances": 0,
            "missing_audio": 0,
            "missing_text": 0,
            "too_short": 0,
            "too_long": 0,
            "missing_speaker": 0,
            "pairs_no_text": 0,
            "pairs_text_prosody": 0,
            "pairs_text_aligned_diagnostic": 0,
            "seedvc_jobs": 0,
            "duplicates": 0,
            "written": 0,
        }
    )
    extra_stats: Counter = Counter()
    utterances: list[Utterance] = []
    stop_reading = False
    for input_path in input_paths:
        if stop_reading:
            break
        try:
            iterator = iter_jsonl_with_line(input_path)
            for line_no, row in iterator:
                if args.max_rows > 0 and stats["read_rows"] >= args.max_rows:
                    stop_reading = True
                    break
                stats["read_rows"] += 1
                utt = normalize_row(
                    row,
                    input_path=input_path,
                    input_line=line_no,
                    global_index=stats["read_rows"] - 1,
                    audio_root=audio_root,
                    audio_keys=audio_keys,
                    text_keys=text_keys,
                    speaker_keys=speaker_keys,
                    gender_keys=gender_keys,
                    infer_speaker_from=args.infer_speaker_from,
                    allow_missing_audio=args.allow_missing_audio,
                    allow_missing_speaker=args.allow_missing_speaker,
                    min_duration=args.min_duration_sec,
                    max_duration=args.max_duration_sec,
                    stats=stats,
                )
                if utt is None:
                    continue
                if languages and utt.language not in languages:
                    extra_stats["language_filtered"] += 1
                    continue
                utterances.append(utt)
                if args.progress_every > 0 and stats["read_rows"] % args.progress_every == 0:
                    print(f"[read] rows={stats['read_rows']} valid={len(utterances)}", flush=True)
        except OSError as exc:
            extra_stats["failed_input_files"] += 1
            print(f"[warn] failed to read {input_path}: {type(exc).__name__}: {exc}", flush=True)

    stats["valid_utterances"] = len(utterances)
    augment_summary: dict[str, Any] = {}
    if args.augment_source_jsonl:
        augment_summary = augment_from_source_jsonl(utterances, gender_keys, text_keys)
        print(
            "[augment] "
            f"files={augment_summary.get('requested_files', 0)} "
            f"gender_filled={augment_summary.get('gender_filled', 0)} "
            f"text_filled={augment_summary.get('text_filled', 0)}",
            flush=True,
        )

    indexes = build_indexes(utterances)
    target_indices = list(range(len(utterances)))
    rng.shuffle(target_indices)

    out_path = Path(args.output_jsonl).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seedvc_output_root = (
        Path(args.seedvc_output_root).expanduser().resolve(strict=False)
        if args.seedvc_output_root
        else (out_path.parent / "seedvc_targets" / args.run_name).resolve(strict=False)
    )
    seedvc_audio_root = seedvc_output_root / args.seedvc_audio_subdir
    seedvc_jobs_path = Path(args.seedvc_jobs_jsonl).expanduser() if args.seedvc_jobs_jsonl else None
    seedvc_jobs_handle = None
    if seedvc_jobs_path is not None:
        seedvc_jobs_path.parent.mkdir(parents=True, exist_ok=True)
        seedvc_jobs_handle = seedvc_jobs_path.open("w", encoding="utf-8")
    seen_pairs: set[tuple[str, str, str, str]] = set()
    seen_jobs: set[tuple[str, str, str]] = set()

    try:
        with out_path.open("w", encoding="utf-8") as handle:
            for plan_ordinal, target_idx in enumerate(target_indices):
                if args.max_pairs > 0 and stats["written"] >= args.max_pairs:
                    break
                source = utterances[target_idx]
                target = source
                timbre_idx, timbre_rule = pick_timbre_ref_same_jsonl(
                    target_idx,
                    utterances,
                    indexes,
                    rng,
                )
                if timbre_idx is None:
                    extra_stats["missing_timbre_ref"] += 1
                    continue
                timbre = utterances[timbre_idx]
                target_audio = str((seedvc_audio_root / seedvc_relative_audio_path(source, plan_ordinal)).resolve(strict=False))
                seedvc_job = build_seedvc_job(
                    run_name=args.run_name,
                    source=source,
                    timbre=timbre,
                    output_audio=target_audio,
                    job_ordinal=plan_ordinal,
                    seed=args.seed,
                    timbre_rule=timbre_rule,
                )
                seedvc_job_id = seedvc_job["job_id"]
                if args.require_target_audio:
                    target_path = Path(target_audio)
                    if not target_path.exists():
                        extra_stats["missing_target_audio"] += 1
                        continue
                    try:
                        target_size = target_path.stat().st_size
                    except OSError:
                        extra_stats["missing_target_audio"] += 1
                        continue
                    if target_size < args.min_target_audio_bytes:
                        extra_stats["too_small_target_audio"] += 1
                        continue

                job_key = (source.audio, timbre.audio, target_audio)
                if job_key not in seen_jobs:
                    seen_jobs.add(job_key)
                    if seedvc_jobs_handle is not None:
                        seedvc_jobs_handle.write(json.dumps(seedvc_job, ensure_ascii=False) + "\n")
                    stats["seedvc_jobs"] += 1

                if PAIR_NO_TEXT in emit_pair_types and args.no_text_source_mode == "seedvc":
                    pair_type = PAIR_NO_TEXT
                    key = (source.audio, timbre.audio, target_audio, pair_type)
                    if key in seen_pairs:
                        stats["duplicates"] += 1
                    else:
                        seen_pairs.add(key)
                        row = build_row(
                            run_name=args.run_name,
                            pair_type=pair_type,
                            source=source,
                            timbre=timbre,
                            target=target,
                            ordinal=stats["written"],
                            construction_rule=f"seedvc_no_text__timbre_{timbre_rule}",
                            text_alignment="aligned",
                            target_audio_override=target_audio,
                            target_text_override=source.text,
                            target_speaker=timbre,
                            target_duration_override=source.duration,
                            seedvc_job_id=seedvc_job_id,
                        )
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats["pairs_no_text"] += 1
                        stats["written"] += 1
                        if args.max_pairs > 0 and stats["written"] >= args.max_pairs:
                            break

                text_pair_requested = PAIR_TEXT_PROSODY in emit_pair_types or PAIR_TEXT_ALIGNED_DIAGNOSTIC in emit_pair_types
                if not text_pair_requested:
                    continue
                if not source.text:
                    extra_stats["text_prosody_missing_target_text"] += 1
                    continue
                pair_type = PAIR_TEXT_PROSODY
                key = (source.audio, timbre.audio, target_audio, pair_type)
                if key in seen_pairs:
                    stats["duplicates"] += 1
                    continue
                seen_pairs.add(key)
                row = build_row(
                    run_name=args.run_name,
                    pair_type=pair_type,
                    source=source,
                    timbre=timbre,
                    target=target,
                    ordinal=stats["written"],
                    construction_rule=f"seedvc_text_prosody__timbre_{timbre_rule}",
                    text_alignment="aligned",
                    target_audio_override=target_audio,
                    target_text_override=source.text,
                    target_speaker=timbre,
                    target_duration_override=source.duration,
                    seedvc_job_id=seedvc_job_id,
                )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["pairs_text_prosody"] += 1
                stats["written"] += 1
                if args.progress_every > 0 and stats["written"] % args.progress_every == 0:
                    print(f"[write] pairs={stats['written']}", flush=True)
    finally:
        if seedvc_jobs_handle is not None:
            seedvc_jobs_handle.close()

    language_counts = Counter(utt.language for utt in utterances)
    gender_counts = Counter(utt.gender for utt in utterances)
    speaker_counts_by_language = Counter()
    for language in language_counts:
        speaker_counts_by_language[language] = len(
            {utt.speaker_id for utt in utterances if utt.language == language}
        )

    summary = {
        "stats": dict(stats),
        "extra_stats": dict(extra_stats),
        "input_jsonl_count": len(input_paths),
        "input_jsonl": [str(path) for path in input_paths],
        "output_jsonl": str(out_path.resolve(strict=False)),
        "config": {
            "run_name": args.run_name,
            "emit_pair_types": emit_pair_types,
            "languages": sorted(languages),
            "min_duration_sec": args.min_duration_sec,
            "max_duration_sec": args.max_duration_sec,
            "allow_missing_audio": args.allow_missing_audio,
            "allow_missing_speaker": args.allow_missing_speaker,
            "infer_speaker_from": args.infer_speaker_from,
            "augment_source_jsonl": args.augment_source_jsonl,
            "text_source_policy": args.text_source_policy,
            "strict_text_source_policy": args.strict_text_source_policy,
            "allow_cross_speaker_timbre_ref": args.allow_cross_speaker_timbre_ref,
            "no_text_source_mode": args.no_text_source_mode,
            "target_audio_mode": args.target_audio_mode,
            "seedvc_output_root": str(seedvc_output_root),
            "seedvc_audio_subdir": args.seedvc_audio_subdir,
            "seedvc_jobs_jsonl": str(seedvc_jobs_path.resolve(strict=False)) if seedvc_jobs_path else "",
            "require_target_audio": args.require_target_audio,
            "min_target_audio_bytes": args.min_target_audio_bytes,
            "seed": args.seed,
        },
        "utterance_counts": {
            "by_language": dict(language_counts),
            "by_gender": dict(gender_counts),
            "speakers_by_language": dict(speaker_counts_by_language),
        },
        "augment_source_jsonl_summary": augment_summary,
        "schema": {
            "required_audio_fields": ["source_audio", "timbre_ref_audio", "target_audio"],
            "pair_types": {
                PAIR_NO_TEXT: "target_audio is generated by Seed-VC from source_audio content/prosody and timbre_ref_audio timbre; downstream SFT should use <NO_TEXT>",
                PAIR_TEXT_PROSODY: "target_audio is the same Seed-VC output and target_text equals source_text, because Seed-VC preserves source lexical content",
            },
        },
    }
    summary_path = Path(args.summary_json).expanduser() if args.summary_json else out_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(f"wrote {stats['written']} manifest rows -> {out_path.resolve(strict=False)}")
    print(f"summary -> {summary_path.resolve(strict=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
