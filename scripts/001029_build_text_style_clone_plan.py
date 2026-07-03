#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "vc_data_temp/mtd_pass_nonmulti_primary_le_0p3_split_10k"
)


TEXT_KEYS = (
    "mtd_transcript",
    "mtd_asr_text",
    "text",
    "transcript",
    "source_text",
    "target_text",
)
AUDIO_KEYS = ("audio_path", "local_path", "wav", "audio", "path")
DURATION_KEYS = ("duration_sec", "duration", "audio_duration")


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    audio: str
    text: str
    language: str
    duration: float | None
    source_dataset: str
    source_jsonl: str
    source_line: int


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def pick_first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def resolve_audio_path(value: Any, audio_root: Path | None) -> str:
    if value is None:
        return ""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        if audio_root is None:
            path = path.absolute()
        else:
            path = audio_root / path
    return str(path.absolute())


def parse_duration(record: dict[str, Any]) -> float | None:
    value = pick_first(record, DURATION_KEYS)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_jsonl_files(input_root: Path, language: str) -> list[Path]:
    lang_dir = input_root / language
    if lang_dir.exists():
        files = sorted(lang_dir.glob("*.jsonl"))
    else:
        files = sorted(input_root.glob(f"{language}*.jsonl"))
    return files


def load_language_candidates(
    *,
    input_root: Path,
    language: str,
    audio_root: Path | None,
    max_rows: int,
    min_duration: float,
    max_duration: float,
    require_audio_exists: bool,
) -> tuple[list[Utterance], dict[str, int]]:
    stats = {
        "read_rows": 0,
        "valid": 0,
        "missing_audio": 0,
        "missing_text": 0,
        "missing_file": 0,
        "too_short": 0,
        "too_long": 0,
    }
    candidates: list[Utterance] = []
    for jsonl_path in iter_jsonl_files(input_root, language):
        jsonl_abs = str(jsonl_path.absolute())
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if max_rows > 0 and stats["read_rows"] >= max_rows:
                    return candidates, stats
                stats["read_rows"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_lang = normalize_text(record.get("language") or language).lower()
                if rec_lang and rec_lang != language:
                    continue
                audio = resolve_audio_path(pick_first(record, AUDIO_KEYS), audio_root)
                if not audio:
                    stats["missing_audio"] += 1
                    continue
                if require_audio_exists and not Path(audio).exists():
                    stats["missing_file"] += 1
                    continue
                text = normalize_text(pick_first(record, TEXT_KEYS))
                if not text:
                    stats["missing_text"] += 1
                    continue
                duration = parse_duration(record)
                if duration is not None:
                    if duration < min_duration:
                        stats["too_short"] += 1
                        continue
                    if duration > max_duration:
                        stats["too_long"] += 1
                        continue
                utt_id = normalize_text(record.get("id") or f"{jsonl_path.stem}_{line_no}")
                candidates.append(
                    Utterance(
                        utt_id=utt_id,
                        audio=audio,
                        text=text,
                        language=language,
                        duration=duration,
                        source_dataset=normalize_text(record.get("source") or record.get("dataset") or ""),
                        source_jsonl=jsonl_abs,
                        source_line=line_no,
                    )
                )
                stats["valid"] += 1
    return candidates, stats


def text_length(text: str, language: str) -> int:
    if language == "zh":
        return max(1, len(re.sub(r"\s+", "", text)))
    return max(1, len(re.findall(r"[A-Za-z0-9']+", text)))


def pick_donor(
    rng: random.Random,
    source: Utterance,
    candidates: list[Utterance],
    *,
    language: str,
    min_text_ratio: float,
    max_text_ratio: float,
) -> Utterance:
    src_len = text_length(source.text, language)
    pool = [
        item
        for item in candidates
        if item.utt_id != source.utt_id
        and item.audio != source.audio
        and item.text != source.text
        and min_text_ratio <= text_length(item.text, language) / src_len <= max_text_ratio
    ]
    if not pool:
        pool = [
            item
            for item in candidates
            if item.utt_id != source.utt_id and item.audio != source.audio and item.text != source.text
        ]
    if not pool:
        raise ValueError(f"not enough candidates to pick text donor for {source.utt_id}")
    return rng.choice(pool)


def pick_timbre_ref(rng: random.Random, source: Utterance, donor: Utterance, candidates: list[Utterance]) -> Utterance:
    pool = [
        item
        for item in candidates
        if item.audio not in {source.audio, donor.audio}
        and item.utt_id not in {source.utt_id, donor.utt_id}
    ]
    if not pool:
        pool = [item for item in candidates if item.audio != source.audio and item.utt_id != source.utt_id]
    if not pool:
        raise ValueError(f"not enough candidates to pick timbre ref for {source.utt_id}")
    return rng.choice(pool)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input_root).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve() if args.audio_root else None
    backends = [item.strip() for item in args.tts_backends.split(",") if item.strip()]
    languages = [item.strip().lower() for item in args.languages.split(",") if item.strip()]
    rng = random.Random(args.seed)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "input_root": str(input_root),
        "output_jsonl": str(output_jsonl),
        "output_root": str(output_root),
        "seed": args.seed,
        "tts_backends": backends,
        "languages": {},
        "written": 0,
    }
    seen_ids: set[str] = set()
    with output_jsonl.open("w", encoding="utf-8") as out:
        for language in languages:
            candidates, stats = load_language_candidates(
                input_root=input_root,
                language=language,
                audio_root=audio_root,
                max_rows=args.max_rows_per_language,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                require_audio_exists=not args.allow_missing_audio,
            )
            rng.shuffle(candidates)
            selected = candidates[: min(args.samples_per_language, len(candidates))]
            lang_written = 0
            for idx, source in enumerate(selected):
                donor = pick_donor(
                    rng,
                    source,
                    candidates,
                    language=language,
                    min_text_ratio=args.min_text_ratio,
                    max_text_ratio=args.max_text_ratio,
                )
                timbre_ref = pick_timbre_ref(rng, source, donor, candidates)
                base_id = f"{language}_{idx:04d}_{source.utt_id[:10]}_{donor.utt_id[:10]}"
                for backend in backends:
                    sample_id = f"{backend}_{base_id}"
                    if sample_id in seen_ids:
                        continue
                    seen_ids.add(sample_id)
                    style_carrier = output_root / "style_carrier" / backend / language / f"{sample_id}.wav"
                    teacher_target = output_root / "teacher_target_seedvc" / backend / language / f"{sample_id}.wav"
                    row = {
                        "sample_id": sample_id,
                        "utt_id": base_id,
                        "mode": "text",
                        "pair_type": "text_prosody",
                        "language": language,
                        "tts_backend": backend,
                        "vc_backend": "seed_vc",
                        "source_style_wav": source.audio,
                        "source_audio": source.audio,
                        "source_style_text": source.text,
                        "source_text": source.text,
                        "input_text": donor.text,
                        "target_text": donor.text,
                        "text_donor_audio": donor.audio,
                        "text_donor_text": donor.text,
                        "timbre_ref_wav": timbre_ref.audio,
                        "timbre_ref_audio": timbre_ref.audio,
                        "timbre_ref_text": timbre_ref.text,
                        "style_carrier_wav": str(style_carrier),
                        "teacher_target_wav": str(teacher_target),
                        "target_audio": str(teacher_target),
                        "source_text_same_as_target": source.text == donor.text,
                        "construction_rule": "text_style_clone_then_seedvc",
                        "instruction": (
                            "Text teacher construction. Use source_style_wav only for speaking style, "
                            "rhythm, pauses and duration hints. Use input_text as lexical content. "
                            "Use timbre_ref_wav as the target timbre in the Seed-VC stage."
                        ),
                        "meta": {
                            "source_dataset": source.source_dataset,
                            "source_jsonl": source.source_jsonl,
                            "source_line": source.source_line,
                            "text_donor_jsonl": donor.source_jsonl,
                            "text_donor_line": donor.source_line,
                            "timbre_ref_jsonl": timbre_ref.source_jsonl,
                            "timbre_ref_line": timbre_ref.source_line,
                            "source_duration": source.duration,
                            "input_text_duration_hint": donor.duration,
                            "timbre_ref_duration": timbre_ref.duration,
                            "source_text_len": text_length(source.text, language),
                            "input_text_len": text_length(donor.text, language),
                        },
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    lang_written += 1
                    summary["written"] += 1
            stats["selected_base_pairs"] = len(selected)
            stats["written_backend_rows"] = lang_written
            summary["languages"][language] = stats
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a small text-mode style-clone plan: source style wav + different input text -> style carrier."
    )
    ap.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--summary-json", default=None)
    ap.add_argument("--languages", default="zh,en")
    ap.add_argument("--samples-per-language", type=int, default=10)
    ap.add_argument("--tts-backends", default="moss_tts")
    ap.add_argument("--max-rows-per-language", type=int, default=20000)
    ap.add_argument("--min-duration", type=float, default=2.0)
    ap.add_argument("--max-duration", type=float, default=12.0)
    ap.add_argument("--min-text-ratio", type=float, default=0.7)
    ap.add_argument("--max-text-ratio", type=float, default=1.4)
    ap.add_argument("--audio-root", default=None)
    ap.add_argument("--allow-missing-audio", action="store_true")
    ap.add_argument("--seed", type=int, default=20260626)
    args = ap.parse_args()

    summary = build_plan(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
