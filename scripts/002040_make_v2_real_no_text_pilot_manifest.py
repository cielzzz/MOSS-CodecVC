#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
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


def normalize_content_text(text: Any) -> str:
    out: list[str] = []
    for ch in str(text or "").lower():
        code = ord(ch)
        if ch.isalnum() or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            out.append(ch)
    return "".join(out)


def is_ref_content_leak(row: dict[str, Any]) -> bool:
    ref_text = normalize_content_text(nested_get(row, "timbre_ref_text"))
    target_text = normalize_content_text(nested_get(row, "target_text"))
    return bool(ref_text and target_text and ref_text == target_text)


def language(row: dict[str, Any]) -> str:
    raw = str(nested_get(row, "language") or nested_get(row, "lang") or "unknown").strip().lower()
    if raw.startswith("zh"):
        return "zh"
    if raw.startswith("en"):
        return "en"
    return raw or "unknown"


def speaker_id(row: dict[str, Any]) -> str:
    value = nested_get(row, "timbre_ref_speaker_id")
    if value not in (None, ""):
        return str(value)
    value = nested_get(row, "target_speaker_id")
    if value not in (None, ""):
        return str(value)
    return str(row.get("sample_id") or row.get("pair_id") or "unknown")


def stable_score(seed: int, *items: Any) -> int:
    payload = "\t".join(str(item) for item in (seed, *items)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def select_balanced(
    groups: dict[str, list[tuple[int, int, str]]],
    *,
    quota: int,
    seed: int,
    max_rounds: int = 1000,
) -> list[tuple[int, int, str]]:
    for speaker, rows in groups.items():
        rows.sort(key=lambda item: (-stable_score(seed, speaker, item[0]), item[0]))
    speaker_order = sorted(groups, key=lambda spk: (-stable_score(seed, spk), spk))
    selected: list[tuple[int, int, str]] = []
    used: set[int] = set()
    round_idx = 0
    while len(selected) < quota and round_idx < max_rounds:
        added = 0
        for spk in speaker_order:
            rows = groups[spk]
            if round_idx >= len(rows):
                continue
            line_no, score, line = rows[round_idx]
            if line_no in used:
                continue
            selected.append((line_no, score, line))
            used.add(line_no)
            added += 1
            if len(selected) >= quota:
                break
        if added == 0:
            break
        round_idx += 1
    selected.sort(key=lambda item: item[0])
    return selected[:quota]


def main() -> int:
    default_base = Path(
        "/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/VC_train/"
        "v2_real_target_no_text_300k_zh_en_balanced_20260707_seedvc_triples"
    )
    ap = argparse.ArgumentParser(description="Build a v2 real-target no-text pilot manifest directly from raw rows.")
    ap.add_argument("--input-jsonl", default=str(default_base / "no_text.train.refdecorr.train_minus_valid.manifest.jsonl"))
    ap.add_argument(
        "--output-jsonl",
        default=(
            "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/"
            "trainset/v2_real_target_no_text_refdecorr_pilot_10k_20260708/manifests/"
            "no_text.v2.pilot_10k.manifest.jsonl"
        ),
    )
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--langs", default="zh,en")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input_jsonl).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    summary_path = Path(args.summary_json).expanduser() if args.summary_json else output_path.with_suffix(".summary.json")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists, pass --overwrite: {output_path}")

    langs = [item.strip().lower() for item in str(args.langs).split(",") if item.strip()]
    if not langs:
        langs = ["zh", "en"]
    base_quota = int(args.sample_size) // len(langs)
    quotas = {lang: base_quota for lang in langs}
    for lang in langs[: int(args.sample_size) - base_quota * len(langs)]:
        quotas[lang] += 1

    groups: dict[str, dict[str, list[tuple[int, int, str]]]] = {lang: defaultdict(list) for lang in langs}
    stats: Counter[str] = Counter()
    input_langs: Counter[str] = Counter()
    kept_langs: Counter[str] = Counter()
    ref_target_same_spk = 0
    ref_target_checked = 0

    for line_no, line, row in iter_jsonl(input_path):
        stats["input_rows"] += 1
        lang = language(row)
        input_langs[lang] += 1
        if lang not in quotas:
            stats["dropped_language"] += 1
            continue
        if is_ref_content_leak(row):
            stats["dropped_ref_content_leak"] += 1
            continue
        timbre_spk = nested_get(row, "timbre_ref_speaker_id")
        target_spk = nested_get(row, "target_speaker_id")
        if timbre_spk not in (None, "") and target_spk not in (None, ""):
            ref_target_checked += 1
            if str(timbre_spk) == str(target_spk):
                ref_target_same_spk += 1
            else:
                stats["dropped_ref_target_speaker_mismatch"] += 1
                continue
        spk = speaker_id(row)
        score = stable_score(int(args.seed), row.get("sample_id") or line_no, line_no)
        groups[lang][spk].append((line_no, score, line if line.endswith("\n") else line + "\n"))
        kept_langs[lang] += 1

    selected_by_lang: dict[str, list[tuple[int, int, str]]] = {}
    for lang in langs:
        selected_by_lang[lang] = select_balanced(groups[lang], quota=quotas[lang], seed=int(args.seed))

    selected = [item for lang in langs for item in selected_by_lang[lang]]
    selected.sort(key=lambda item: item[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for _line_no, _score, line in selected:
            handle.write(line)
    tmp.replace(output_path)

    pilot_langs: Counter[str] = Counter()
    pilot_speakers: dict[str, set[str]] = defaultdict(set)
    for _line_no, _line, row in iter_jsonl(output_path):
        lang = language(row)
        pilot_langs[lang] += 1
        pilot_speakers[lang].add(speaker_id(row))

    summary = {
        "status": "complete",
        "input_jsonl": str(input_path.resolve(strict=False)),
        "output_jsonl": str(output_path.resolve(strict=False)),
        "sample_size_requested": int(args.sample_size),
        "sample_size_written": len(selected),
        "seed": int(args.seed),
        "quotas": quotas,
        "stats": dict(stats.most_common()),
        "input_language_counts": dict(input_langs.most_common()),
        "post_filter_language_counts": dict(kept_langs.most_common()),
        "pilot_language_counts": dict(pilot_langs.most_common()),
        "pilot_speaker_counts": {lang: len(values) for lang, values in pilot_speakers.items()},
        "ref_target_speaker_same_rate": None
        if ref_target_checked == 0
        else ref_target_same_spk / float(ref_target_checked),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
