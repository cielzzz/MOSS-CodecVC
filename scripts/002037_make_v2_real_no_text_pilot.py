#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import heapq
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield line_no, line, json.loads(line)


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


def language(row: dict[str, Any]) -> str:
    raw = str(nested_get(row, "language") or nested_get(row, "lang") or "unknown").strip().lower()
    if raw.startswith("zh"):
        return "zh"
    if raw.startswith("en"):
        return "en"
    return raw or "unknown"


def speaker_id(row: dict[str, Any]) -> str:
    for key in (
        "target_speaker_id",
        "timbre_ref_speaker_id",
        "target_pseudo_speaker_id",
        "timbre_ref_pseudo_speaker_id",
        "speaker_id",
    ):
        value = nested_get(row, key)
        if value not in (None, ""):
            return str(value)
    return str(row.get("sample_id") or row.get("pair_id") or row.get("id") or "unknown")


def score_for(seed: int, sample_id: str, line_no: int) -> float:
    payload = f"{seed}\t{sample_id}\t{line_no}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / float(2**64 - 1)


def push_candidate(
    heaps: dict[str, list[tuple[float, int, str, str, str]]],
    speaker_counts: dict[str, Counter[str]],
    quotas: dict[str, int],
    *,
    lang: str,
    speaker: str,
    line_no: int,
    score: float,
    line: str,
    max_rows_per_speaker: int,
    enforce_speaker_cap: bool,
) -> None:
    heap = heaps[lang]
    quota = quotas[lang]
    if quota <= 0:
        return
    spk_counts = speaker_counts[lang]
    if len(heap) < quota:
        if enforce_speaker_cap and spk_counts[speaker] >= max_rows_per_speaker:
            return
        heapq.heappush(heap, (score, line_no, speaker, lang, line))
        spk_counts[speaker] += 1
        return
    if not heap or score <= heap[0][0]:
        return
    if enforce_speaker_cap and spk_counts[speaker] >= max_rows_per_speaker:
        return
    _old_score, _old_line_no, old_speaker, _old_lang, _old_line = heapq.heapreplace(
        heap, (score, line_no, speaker, lang, line)
    )
    spk_counts[old_speaker] -= 1
    if spk_counts[old_speaker] <= 0:
        del spk_counts[old_speaker]
    spk_counts[speaker] += 1


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(src)
    tmp.replace(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a speaker/language-diverse v2 no-text pilot prepared dir.")
    ap.add_argument("--source-prepared-dir", required=True)
    ap.add_argument("--output-prepared-dir", required=True)
    ap.add_argument(
        "--text-prepared-dir",
        default="",
        help="Fallback directory for old text.*.jsonl splits when source-prepared-dir does not contain text symlinks yet.",
    )
    ap.add_argument("--input-split", default="no_text.v2.train.filtered.jsonl")
    ap.add_argument("--output-split", default="no_text.v2.pilot.jsonl")
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--langs", default="zh,en")
    ap.add_argument("--max-rows-per-speaker", type=int, default=2)
    ap.add_argument("--text-repeat", type=int, default=10)
    args = ap.parse_args()

    src_dir = Path(args.source_prepared_dir).expanduser().resolve()
    out_dir = Path(args.output_prepared_dir).expanduser().resolve()
    text_dir = Path(args.text_prepared_dir).expanduser().resolve() if args.text_prepared_dir else None
    input_jsonl = src_dir / args.input_split
    output_jsonl = out_dir / args.output_split
    if not input_jsonl.exists():
        raise SystemExit(f"missing input split: {input_jsonl}")

    langs = [item.strip().lower() for item in str(args.langs).split(",") if item.strip()]
    if not langs:
        langs = ["zh", "en"]
    base = int(args.sample_size) // len(langs)
    quotas = {lang: base for lang in langs}
    for lang in langs[: int(args.sample_size) - base * len(langs)]:
        quotas[lang] += 1

    heaps: dict[str, list[tuple[float, int, str, str, str]]] = {lang: [] for lang in langs}
    speaker_counts: dict[str, Counter[str]] = {lang: Counter() for lang in langs}
    full_counts: Counter[str] = Counter()
    full_speakers: dict[str, set[str]] = defaultdict(set)

    def scan(*, enforce_speaker_cap: bool, skip_existing_lines: set[int] | None = None) -> None:
        skip_existing_lines = skip_existing_lines or set()
        for line_no, line, row in iter_jsonl(input_jsonl):
            if line_no in skip_existing_lines:
                continue
            lang = language(row)
            spk = speaker_id(row)
            if enforce_speaker_cap:
                full_counts[lang] += 1
                full_speakers[lang].add(spk)
            if lang not in quotas:
                continue
            sample_id = str(row.get("sample_id") or row.get("pair_id") or line_no)
            score = score_for(int(args.seed), sample_id, line_no)
            push_candidate(
                heaps,
                speaker_counts,
                quotas,
                lang=lang,
                speaker=spk,
                line_no=line_no,
                score=score,
                line=line,
                max_rows_per_speaker=max(1, int(args.max_rows_per_speaker)),
                enforce_speaker_cap=enforce_speaker_cap,
            )

    scan(enforce_speaker_cap=True)
    selected_lines = {int(item[1]) for heap in heaps.values() for item in heap}
    if any(len(heaps[lang]) < quotas[lang] for lang in langs):
        # Fallback fills rare-language quotas even if the per-speaker cap is too strict.
        scan(enforce_speaker_cap=False, skip_existing_lines=selected_lines)

    selected = [item for heap in heaps.values() for item in heap]
    selected.sort(key=lambda item: item[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for _score, _line_no, _speaker, _lang, line in selected:
            handle.write(line if line.endswith("\n") else line + "\n")

    symlink_force(output_jsonl, out_dir / "no_text.train.jsonl")
    valid_aliases = {
        "no_text.valid.jsonl": "no_text.heldout_refdecorr_cross_channel_valid.filtered.jsonl",
        "no_text.seen_valid.jsonl": "no_text.same_episode_near_original_valid.filtered.jsonl",
        "no_text.unseen_valid.jsonl": "no_text.heldout_refdecorr_cross_channel_valid.filtered.jsonl",
        "no_text.same_episode_near_original_valid.jsonl": "no_text.same_episode_near_original_valid.filtered.jsonl",
        "no_text.heldout_refdecorr_cross_channel_valid.jsonl": "no_text.heldout_refdecorr_cross_channel_valid.filtered.jsonl",
        "no_text.same_episode_near_original_valid.filtered.jsonl": "no_text.same_episode_near_original_valid.filtered.jsonl",
        "no_text.heldout_refdecorr_cross_channel_valid.filtered.jsonl": "no_text.heldout_refdecorr_cross_channel_valid.filtered.jsonl",
    }
    valid_link_targets: dict[str, str] = {}
    for name, filtered_name in valid_aliases.items():
        src = src_dir / filtered_name
        if not (src.exists() or src.is_symlink()):
            src = src_dir / name
        if src.exists() or src.is_symlink():
            symlink_force(src, out_dir / name)
            valid_link_targets[name] = str(src)
    for name in ("text.train.jsonl", "text.valid.jsonl", "text.seen_valid.jsonl", "text.unseen_valid.jsonl"):
        src = src_dir / name
        if not (src.exists() or src.is_symlink()) and text_dir is not None:
            src = text_dir / name
        if src.exists() or src.is_symlink():
            symlink_force(src, out_dir / name)

    (out_dir / "mixed.train.spec.txt").write_text(
        f"{out_dir / 'no_text.train.jsonl'}::repeat=1\n{out_dir / 'text.train.jsonl'}::repeat={int(args.text_repeat)}\n",
        encoding="utf-8",
    )
    if (out_dir / "no_text.valid.jsonl").exists() and (out_dir / "text.valid.jsonl").exists():
        (out_dir / "mixed.valid.spec.txt").write_text(
            f"{out_dir / 'no_text.valid.jsonl'}::repeat=1\n{out_dir / 'text.valid.jsonl'}::repeat=1\n",
            encoding="utf-8",
        )

    selected_langs = Counter(item[3] for item in selected)
    selected_speakers: dict[str, set[str]] = defaultdict(set)
    for _score, _line_no, speaker, lang, _line in selected:
        selected_speakers[lang].add(speaker)
    summary = {
        "status": "complete",
        "source_prepared_dir": str(src_dir),
        "output_prepared_dir": str(out_dir),
        "input_split": str(input_jsonl),
        "output_split": str(output_jsonl),
        "sample_size_requested": int(args.sample_size),
        "sample_size_written": len(selected),
        "seed": int(args.seed),
        "quotas": quotas,
        "max_rows_per_speaker": int(args.max_rows_per_speaker),
        "source_language_counts": dict(full_counts.most_common()),
        "source_speaker_counts": {lang: len(values) for lang, values in full_speakers.items()},
        "pilot_language_counts": dict(selected_langs.most_common()),
        "pilot_speaker_counts": {lang: len(values) for lang, values in selected_speakers.items()},
        "used_uncapped_fallback": any(speaker_counts[lang][speaker] > int(args.max_rows_per_speaker) for lang in langs for speaker in speaker_counts[lang]),
        "valid_link_targets": valid_link_targets,
        "mixed_train_spec": str(out_dir / "mixed.train.spec.txt"),
    }
    summary_path = out_dir / "v2_no_text_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
