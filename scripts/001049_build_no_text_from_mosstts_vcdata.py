#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import stable_id


PAIR_TYPE = "no_text"

DEFAULT_NO_TEXT_INSTRUCTION = (
    "Voice conversion task. [S1] is the source speech carrying content, pauses, duration and prosody. "
    "[S2] is the target timbre reference. Generate the same content as S1 with S2 timbre while preserving "
    "S1 timing and prosody."
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_key(value: Any) -> str:
    text = normalize_text(value).lower()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", "", text)


def resolve_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve(strict=False))


def path_exists(value: str) -> bool:
    try:
        return Path(value).exists()
    except OSError:
        return False


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_language(path: Path, row: dict[str, Any]) -> str:
    lang = str(row.get("language") or "").strip().lower()
    if lang in {"zh", "en"}:
        return lang
    for part in reversed(path.parts):
        if part in {"zh", "en"}:
            return part
    text = normalize_text(row.get("original_text") or row.get("ref_text"))
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    return "en"


def split_name_from_path(path: Path) -> str:
    return path.parent.name or path.stem or "split"


def discover_vcdata_jsonls(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for raw in values:
        for item in [part.strip() for part in raw.split(",") if part.strip()]:
            p = Path(item).expanduser()
            if p.is_dir():
                matches = sorted(p.glob("**/merged.stepaudio_input.all.jsonl"))
                if not matches:
                    matches = sorted(p.glob("**/manifest_merged.jsonl"))
            else:
                matches = sorted(Path(x) for x in p.parent.glob(p.name)) if any(ch in item for ch in "*?[]") else [p]
            for match in matches:
                resolved = match.resolve(strict=False)
                key = str(resolved)
                if key not in seen:
                    seen.add(key)
                    paths.append(resolved)
    return paths


def resolve_clone_audio(input_jsonl: Path, row: dict[str, Any]) -> tuple[str, bool]:
    field_audio = resolve_path(row.get("ref_audio_path"))
    if path_exists(field_audio):
        return field_audio, False
    original_idx = as_int(row.get("original_idx"))
    if original_idx is None:
        return field_audio, False
    archived_audio = input_jsonl.parent / "ref_audio" / f"{original_idx:06d}_ref.wav"
    archived_audio = archived_audio.resolve(strict=False)
    if archived_audio.exists():
        return str(archived_audio), True
    return field_audio, False


def speaker_id(prefix: str, language: str, split: str, row: dict[str, Any], audio: str) -> str:
    idx = row.get("original_idx")
    if idx not in (None, ""):
        return f"{prefix}:{language}:{split}:{idx}"
    return f"{prefix}:{language}:{stable_id(audio, length=16)}"


def load_rows(path: Path, stats: Counter) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        rows.append(row)
        if as_int(row.get("original_idx")) is None:
            stats["missing_original_idx"] += 1
    return rows


def valid_original_candidate(row: dict[str, Any], *, require_existing_audio: bool) -> bool:
    if as_int(row.get("original_idx")) is None:
        return False
    if row.get("original_audio_path") in (None, ""):
        return False
    if not normalize_text(row.get("original_text")):
        return False
    if require_existing_audio and not path_exists(resolve_path(row.get("original_audio_path"))):
        return False
    return True


def valid_clone_candidate(input_jsonl: Path, row: dict[str, Any], *, require_existing_audio: bool) -> bool:
    if as_int(row.get("original_idx")) is None:
        return False
    if row.get("ref_audio_path") in (None, ""):
        return False
    if require_existing_audio and not path_exists(resolve_clone_audio(input_jsonl, row)[0]):
        return False
    return True


def pick_original_timbre_donor(
    *,
    input_jsonl: Path,
    source_row: dict[str, Any],
    candidates: list[dict[str, Any]],
    seed: int,
    require_different_text: bool,
) -> dict[str, Any] | None:
    source_idx = as_int(source_row.get("original_idx"))
    source_audio = resolve_path(source_row.get("original_audio_path"))
    source_text_key = text_key(source_row.get("original_text"))
    if not candidates:
        return None
    choice_seed = int(stable_id(seed, input_jsonl, source_idx, source_audio, length=16), 16)
    rng = random.Random(choice_seed)

    def eligible(candidate: dict[str, Any]) -> bool:
        candidate_idx = as_int(candidate.get("original_idx"))
        if candidate_idx is None or candidate_idx == source_idx:
            return False
        candidate_audio = resolve_path(candidate.get("original_audio_path"))
        if candidate_audio == source_audio:
            return False
        if require_different_text and text_key(candidate.get("original_text")) == source_text_key:
            return False
        return True

    for _ in range(min(len(candidates) * 2, 256)):
        candidate = candidates[rng.randrange(len(candidates))]
        if eligible(candidate):
            return candidate
    start = rng.randrange(len(candidates))
    for offset in range(len(candidates)):
        candidate = candidates[(start + offset) % len(candidates)]
        if eligible(candidate):
            return candidate
    return None


def should_skip_for_quality(row: dict[str, Any], args: argparse.Namespace, stats: Counter) -> bool:
    if args.skip_flags:
        flags = {item.strip() for item in args.skip_flags.split(",") if item.strip()}
        if str(row.get("flag") or "") in flags:
            stats["skip_flag"] += 1
            return True
    best_similarity = as_float(row.get("best_similarity"))
    if args.min_best_similarity > 0 and (best_similarity is None or best_similarity < args.min_best_similarity):
        stats["skip_low_similarity"] += 1
        return True
    dnsmos = as_float(row.get("dnsmos"))
    if args.min_dnsmos > 0 and (dnsmos is None or dnsmos < args.min_dnsmos):
        stats["skip_low_dnsmos"] += 1
        return True
    return False


def make_output_audio(root: Path, language: str, split: str, original_idx: int, source_audio: str, timbre_audio: str) -> str:
    digest = stable_id(language, split, original_idx, source_audio, timbre_audio, length=12)
    return str((root / language / split / f"{original_idx:06d}_{digest}.wav").resolve(strict=False))


def make_job(
    *,
    run_name: str,
    input_jsonl: Path,
    language: str,
    split: str,
    source_row: dict[str, Any],
    timbre_row: dict[str, Any],
    timbre_audio: str,
    timbre_text: str,
    output_audio: str,
    timbre_ref_policy: str,
) -> dict[str, Any]:
    original_idx = as_int(source_row.get("original_idx"))
    if original_idx is None:
        raise ValueError("missing original_idx")
    source_audio = resolve_path(source_row["original_audio_path"])
    source_text = normalize_text(source_row.get("original_text"))
    timbre_idx = as_int(timbre_row.get("original_idx"))
    digest = stable_id(input_jsonl, original_idx, source_audio, timbre_audio, source_text, length=12)
    source_spk = speaker_id("vcdata_original", language, split, source_row, source_audio)
    timbre_spk_prefix = "vcdata_clone" if timbre_ref_policy == "clone_same_source" else "vcdata_original"
    timbre_spk = speaker_id(timbre_spk_prefix, language, split, timbre_row, timbre_audio)
    return {
        "job_id": f"{run_name}:no_text_seedvc:{language}:{split}:{original_idx:06d}:{digest}",
        "pair_type": PAIR_TYPE,
        "prosody_ref_audio": source_audio,
        "prosody_ref_text": source_text,
        "source_audio": source_audio,
        "source_text": source_text,
        "timbre_ref_audio": timbre_audio,
        "timbre_ref_text": timbre_text,
        "target_audio": output_audio,
        "output_audio": output_audio,
        "target_text": source_text,
        "language": language,
        "instruction": DEFAULT_NO_TEXT_INSTRUCTION,
        "source_speaker_id": source_spk,
        "timbre_ref_speaker_id": timbre_spk,
        "target_speaker_id": timbre_spk,
        "source_gender": "unknown",
        "timbre_ref_gender": "unknown",
        "target_gender": "unknown",
        "metadata": {
            "route": "mosstts_vcdata_then_seedvc_no_text",
            "run_name": run_name,
            "input_jsonl": str(input_jsonl),
            "split": split,
            "language": language,
            "original_idx": original_idx,
            "timbre_ref_original_idx": timbre_idx,
            "timbre_ref_policy": timbre_ref_policy,
            "style_clone_backend": "moss_tts",
            "style_clone_best_seed": source_row.get("best_seed"),
            "style_clone_best_similarity": source_row.get("best_similarity"),
            "style_clone_flag": source_row.get("flag"),
            "style_clone_dnsmos": source_row.get("dnsmos"),
            "seedvc_backend": "seed_vc_v1_zero_shot_voice_conversion",
            "construction_rule": (
                "source_original_to_independent_original_timbre"
                if timbre_ref_policy == "random_original_different_row"
                else "source_original_to_same_source_mosstts_clone_timbre"
            ),
            "source_audio_role": "S1 source/content/prosody carrier.",
            "timbre_ref_audio_role": "S2 target timbre reference.",
        },
    }


def prepare(args: argparse.Namespace) -> int:
    inputs = discover_vcdata_jsonls(args.vcdata_jsonl)
    if not inputs:
        raise SystemExit("no vcdata JSONL files found")

    jobs_path = Path(args.jobs_jsonl).expanduser().resolve(strict=False)
    target_audio_root = Path(args.target_audio_root).expanduser().resolve(strict=False)
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and jobs_path.exists():
        jobs_path.unlink()
    if jobs_path.exists() and not args.overwrite:
        raise SystemExit(f"jobs_jsonl already exists, pass --overwrite to replace: {jobs_path}")

    languages = {item.strip() for item in args.languages.split(",") if item.strip()}
    stats: Counter = Counter()
    seen_jobs: set[str] = set()
    written = 0
    with jobs_path.open("w", encoding="utf-8") as out:
        for input_path in inputs:
            rows = load_rows(input_path, stats)
            source_rows = rows[: args.max_rows_per_input] if args.max_rows_per_input > 0 else rows
            original_candidates = [row for row in rows if valid_original_candidate(row, require_existing_audio=args.require_existing_audio)]
            clone_candidates = [row for row in rows if valid_clone_candidate(input_path, row, require_existing_audio=args.require_existing_audio)]
            split = split_name_from_path(input_path)
            stats["input_files"] += 1
            stats["loaded_rows"] += len(rows)
            stats["candidate_rows"] += len(source_rows)
            stats["original_timbre_candidate_rows"] += len(original_candidates)
            stats["clone_timbre_candidate_rows"] += len(clone_candidates)
            print(
                "[prepare-no-text-vcdata] "
                f"input={input_path} rows={len(rows)} candidate_rows={len(source_rows)} "
                f"original_timbre_candidates={len(original_candidates)} clone_timbre_candidates={len(clone_candidates)}",
                flush=True,
            )

            for row in source_rows:
                if args.max_jobs > 0 and written >= args.max_jobs:
                    break
                language = infer_language(input_path, row)
                if languages and language not in languages:
                    stats["skip_language"] += 1
                    continue
                if args.max_jobs_per_language > 0 and stats[f"jobs_{language}"] >= args.max_jobs_per_language:
                    stats["skip_language_quota_filled"] += 1
                    continue
                if should_skip_for_quality(row, args, stats):
                    continue
                if not valid_original_candidate(row, require_existing_audio=args.require_existing_audio):
                    stats["skip_invalid_source"] += 1
                    continue
                source_text = normalize_text(row.get("original_text"))
                if not source_text:
                    stats["skip_empty_source_text"] += 1
                    continue
                original_idx = as_int(row.get("original_idx"))
                if original_idx is None:
                    stats["skip_bad_original_idx"] += 1
                    continue

                if args.timbre_ref_policy == "random_original_different_row":
                    donor = pick_original_timbre_donor(
                        input_jsonl=input_path,
                        source_row=row,
                        candidates=original_candidates,
                        seed=args.timbre_ref_seed,
                        require_different_text=args.require_different_timbre_text,
                    )
                    if donor is None:
                        stats["skip_no_original_timbre_donor"] += 1
                        continue
                    timbre_audio = resolve_path(donor.get("original_audio_path"))
                    timbre_text = normalize_text(donor.get("original_text"))
                elif args.timbre_ref_policy == "clone_same_source":
                    if not valid_clone_candidate(input_path, row, require_existing_audio=args.require_existing_audio):
                        stats["skip_invalid_clone_timbre"] += 1
                        continue
                    donor = row
                    timbre_audio, repaired = resolve_clone_audio(input_path, row)
                    if repaired:
                        stats["repaired_clone_audio_path"] += 1
                    timbre_text = normalize_text(row.get("ref_text"))
                else:
                    raise ValueError(f"unsupported timbre_ref_policy: {args.timbre_ref_policy}")

                if args.require_existing_audio and not path_exists(timbre_audio):
                    stats["skip_missing_timbre_audio"] += 1
                    continue
                source_audio = resolve_path(row.get("original_audio_path"))
                output_audio = make_output_audio(target_audio_root, language, split, original_idx, source_audio, timbre_audio)
                try:
                    job = make_job(
                        run_name=args.run_name,
                        input_jsonl=input_path,
                        language=language,
                        split=split,
                        source_row=row,
                        timbre_row=donor,
                        timbre_audio=timbre_audio,
                        timbre_text=timbre_text,
                        output_audio=output_audio,
                        timbre_ref_policy=args.timbre_ref_policy,
                    )
                except (KeyError, ValueError):
                    stats["skip_build_error"] += 1
                    continue

                job_key = stable_id(job["prosody_ref_audio"], job["timbre_ref_audio"], job["output_audio"], PAIR_TYPE, length=24)
                if job_key in seen_jobs:
                    stats["duplicates"] += 1
                    continue
                seen_jobs.add(job_key)
                out.write(json.dumps(job, ensure_ascii=False) + "\n")
                written += 1
                stats[f"jobs_{language}"] += 1
                if args.progress_every > 0 and written % args.progress_every == 0:
                    print(f"[prepare-no-text-vcdata] written={written} language={language} split={split}", flush=True)
            if args.max_jobs > 0 and written >= args.max_jobs:
                break

    stats["written_jobs"] = written
    summary = {
        "stage": "prepare",
        "pair_type": PAIR_TYPE,
        "run_name": args.run_name,
        "inputs": [str(path) for path in inputs],
        "jobs_jsonl": str(jobs_path),
        "target_audio_root": str(target_audio_root),
        "timbre_ref_policy": args.timbre_ref_policy,
        "timbre_ref_seed": args.timbre_ref_seed,
        "stats": dict(stats),
        "schema_note": "Seed-VC uses prosody_ref_audio=source original audio. Final SFT no_text text is filled by downstream placeholder mode.",
    }
    summary_path = Path(args.summary_json).expanduser().resolve(strict=False) if args.summary_json else jobs_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(f"[prepare-no-text-vcdata] wrote jobs={written} -> {jobs_path}")
    print(f"[prepare-no-text-vcdata] summary -> {summary_path}")
    return 0


def result_ok(result: dict[str, Any], min_target_audio_bytes: int) -> bool:
    if not result.get("ok"):
        return False
    audio = result.get("audio") or result.get("output_audio")
    if not audio:
        return False
    path = Path(str(audio))
    if not path.exists():
        return False
    try:
        return path.stat().st_size >= min_target_audio_bytes
    except OSError:
        return False


def build_manifest_row(job: dict[str, Any], result: dict[str, Any], ordinal: int, run_name: str) -> dict[str, Any]:
    target_audio = resolve_path(result.get("audio") or job.get("output_audio"))
    source_audio = resolve_path(job["source_audio"])
    timbre_audio = resolve_path(job["timbre_ref_audio"])
    digest = stable_id(source_audio, timbre_audio, target_audio, job.get("target_text"), length=12)
    meta = dict(job.get("metadata") or {})
    meta.update(
        {
            "seedvc_job_id": job.get("job_id"),
            "seedvc_result": result,
            "target_duration": result.get("duration_sec"),
            "target_audio_backend": "seed_vc_v1_zero_shot_voice_conversion",
        }
    )
    return {
        "sample_id": f"{run_name}:{PAIR_TYPE}:{ordinal:08d}:{digest}",
        "source_audio": source_audio,
        "source_text": normalize_text(job.get("source_text")),
        "timbre_ref_audio": timbre_audio,
        "timbre_ref_text": normalize_text(job.get("timbre_ref_text")),
        "target_audio": target_audio,
        "target_text": normalize_text(job.get("target_text")),
        "language": job.get("language") or meta.get("language"),
        "source_speaker_id": job.get("source_speaker_id") or f"source:{stable_id(source_audio, length=16)}",
        "timbre_ref_speaker_id": job.get("timbre_ref_speaker_id") or f"timbre:{stable_id(timbre_audio, length=16)}",
        "target_speaker_id": job.get("target_speaker_id") or job.get("timbre_ref_speaker_id") or f"timbre:{stable_id(timbre_audio, length=16)}",
        "source_gender": job.get("source_gender") or "unknown",
        "timbre_ref_gender": job.get("timbre_ref_gender") or "unknown",
        "target_gender": job.get("target_gender") or job.get("timbre_ref_gender") or "unknown",
        "pair_type": PAIR_TYPE,
        "instruction": DEFAULT_NO_TEXT_INSTRUCTION,
        "preferred_emit_mode": "no_text",
        "meta": meta,
    }


def collect(args: argparse.Namespace) -> int:
    jobs_path = Path(args.jobs_jsonl).expanduser().resolve(strict=False)
    results_path = Path(args.results_jsonl).expanduser().resolve(strict=False)
    output_path = Path(args.output_jsonl).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output_jsonl already exists, pass --overwrite to replace: {output_path}")

    results = {row.get("job_id"): row for row in iter_jsonl(results_path)}
    stats: Counter = Counter({"results_rows": len(results)})
    seen_pairs: set[tuple[str, str, str, str]] = set()
    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for job in iter_jsonl(jobs_path):
            stats["jobs_read"] += 1
            result = results.get(job.get("job_id"))
            if result is None:
                stats["skip_missing_result"] += 1
                continue
            if not result_ok(result, args.min_target_audio_bytes):
                stats["skip_bad_result_or_audio"] += 1
                continue
            if args.require_source_audio and not path_exists(resolve_path(job.get("source_audio"))):
                stats["skip_missing_source_audio"] += 1
                continue
            if args.require_source_audio and not path_exists(resolve_path(job.get("timbre_ref_audio"))):
                stats["skip_missing_timbre_audio"] += 1
                continue
            key = (
                resolve_path(job.get("source_audio")),
                resolve_path(job.get("timbre_ref_audio")),
                resolve_path(result.get("audio") or job.get("output_audio")),
                PAIR_TYPE,
            )
            if key in seen_pairs:
                stats["duplicates"] += 1
                continue
            seen_pairs.add(key)
            row = build_manifest_row(job, result, written, args.run_name)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            stats[f"written_{row.get('language') or 'unknown'}"] += 1
            if args.max_rows > 0 and written >= args.max_rows:
                break

    stats["written"] = written
    summary = {
        "stage": "collect",
        "pair_type": PAIR_TYPE,
        "run_name": args.run_name,
        "jobs_jsonl": str(jobs_path),
        "results_jsonl": str(results_path),
        "output_jsonl": str(output_path),
        "stats": dict(stats),
        "schema_note": "Final manifest is no_text only. Downstream SFT must emit <NO_TEXT> via no_text placeholder mode.",
    }
    summary_path = Path(args.summary_json).expanduser().resolve(strict=False) if args.summary_json else output_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(f"[collect-no-text-vcdata] wrote rows={written} -> {output_path}")
    print(f"[collect-no-text-vcdata] summary -> {summary_path}")
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", default="no_text_mosstts_vcdata_seedvc")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--overwrite", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no_text VC data from MOSS-TTS vcdata clone outputs.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Build Seed-VC no_text jobs from vcdata merged JSONL.")
    add_common_args(prep)
    prep.add_argument("--vcdata-jsonl", action="append", required=True, help="vcdata JSONL, directory, glob, or comma list.")
    prep.add_argument("--jobs-jsonl", required=True)
    prep.add_argument("--target-audio-root", required=True)
    prep.add_argument("--languages", default="zh,en")
    prep.add_argument("--max-rows-per-input", type=int, default=0)
    prep.add_argument("--max-jobs", type=int, default=0)
    prep.add_argument("--max-jobs-per-language", type=int, default=0)
    prep.add_argument("--require-existing-audio", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--progress-every", type=int, default=1000)
    prep.add_argument(
        "--timbre-ref-policy",
        choices=("random_original_different_row", "clone_same_source"),
        default="random_original_different_row",
        help=(
            "random_original_different_row keeps the original no_text VC semantics: S2 is an independent raw utterance. "
            "clone_same_source uses the MOSS-TTS clone of S1 as S2, mainly for diagnostics."
        ),
    )
    prep.add_argument("--timbre-ref-seed", type=int, default=20260629)
    prep.add_argument("--require-different-timbre-text", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--min-best-similarity", type=float, default=0.0)
    prep.add_argument("--min-dnsmos", type=float, default=0.0)
    prep.add_argument("--skip-flags", default="", help="Comma-separated vcdata flags to skip, e.g. LOW_SIM.")

    coll = sub.add_parser("collect", help="Collect Seed-VC results into final no_text manifest.")
    add_common_args(coll)
    coll.add_argument("--jobs-jsonl", required=True)
    coll.add_argument("--results-jsonl", required=True)
    coll.add_argument("--output-jsonl", required=True)
    coll.add_argument("--max-rows", type=int, default=0)
    coll.add_argument("--min-target-audio-bytes", type=int, default=4096)
    coll.add_argument("--require-source-audio", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "collect":
        return collect(args)
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
