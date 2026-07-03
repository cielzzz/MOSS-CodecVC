#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build fixed MOSS-CodecVC validation pairs from Seed-TTS-Eval EN/ZH records."
    )
    ap.add_argument(
        "--seedtts-jsonl",
        default="/inspire/ssd/project/embodied-multimodality/public/xyzhang/asr_eval_framework/data/benchmarks/seed-tts-eval.jsonl",
    )
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument(
        "--speaker-metadata-jsonl",
        default="",
        help="Optional metadata JSONL with id/audio_path and gender fields.",
    )
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--modes", default="no_text,text", help="Comma separated: no_text,text")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--require-existing-audio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def infer_lang(row: dict[str, Any]) -> str:
    for key in ("lang", "language"):
        value = str(row.get(key) or "").strip().lower()
        if value.startswith("zh") or value.startswith("cmn") or value == "chinese":
            return "zh"
        if value.startswith("en") or value == "english":
            return "en"
    task = str(row.get("task") or "").lower()
    if re.search(r"(^|[-_])zh($|[-_])", task) or "chinese" in task:
        return "zh"
    if re.search(r"(^|[-_])en($|[-_])", task) or "english" in task:
        return "en"
    text = str(row.get("text") or row.get("ref_text") or "")
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def normalize_gender(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"m", "male", "man", "boy"}:
        return "male"
    if text in {"f", "female", "woman", "girl"}:
        return "female"
    return "unknown"


def load_gender_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    mapping: dict[str, str] = {}
    for row in iter_jsonl(Path(path).expanduser()):
        gender = normalize_gender(row.get("gender") or row.get("speaker_gender") or row.get("sex"))
        if gender == "unknown":
            continue
        for key in ("id", "sample_id", "audio_path", "audio", "wav", "path"):
            value = row.get(key)
            if value in (None, ""):
                continue
            mapping[str(value)] = gender
            try:
                mapping[str(Path(str(value)).expanduser().resolve(strict=False))] = gender
            except Exception:
                pass
    return mapping


def record_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("sample_id") or Path(str(row.get("audio_path") or "")).stem)


def record_audio(row: dict[str, Any]) -> str:
    return str(row.get("audio_path") or row.get("audio") or row.get("wav") or row.get("path") or "")


def record_source_text(row: dict[str, Any]) -> str:
    return str(row.get("ref_text") or row.get("source_text") or row.get("transcript") or row.get("text") or "").strip()


def record_target_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("target_text") or row.get("normalized_text") or "").strip()


def enrich_records(seedtts_jsonl: Path, gender_map: dict[str, str], require_existing_audio: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in iter_jsonl(seedtts_jsonl):
        audio = record_audio(row)
        if not audio:
            continue
        if require_existing_audio and not Path(audio).exists():
            continue
        rid = record_id(row)
        gender = (
            normalize_gender(row.get("gender") or row.get("speaker_gender") or row.get("sex"))
            or gender_map.get(rid)
            or gender_map.get(audio)
            or gender_map.get(str(Path(audio).resolve(strict=False)))
        )
        if gender == "unknown":
            gender = gender_map.get(rid) or gender_map.get(audio) or gender_map.get(str(Path(audio).resolve(strict=False))) or "unknown"
        source_text = record_source_text(row)
        target_text = record_target_text(row)
        if not source_text and not target_text:
            continue
        records.append(
            {
                "id": rid,
                "audio_path": audio,
                "source_text": source_text or target_text,
                "target_text": target_text or source_text,
                "lang": infer_lang(row),
                "gender": normalize_gender(gender),
                "task": row.get("task"),
            }
        )
    return records


def group_records(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["lang"], row["gender"])].append(row)
        groups[(row["lang"], "any")].append(row)
    return groups


def sample_pairs(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    source_lang: str,
    ref_lang: str,
    source_gender: str,
    ref_gender: str,
    count: int,
    rng: random.Random,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    sources = groups.get((source_lang, source_gender)) or groups.get((source_lang, "any")) or []
    refs = groups.get((ref_lang, ref_gender)) or groups.get((ref_lang, "any")) or []
    if not sources or not refs:
        return []
    pairs = []
    attempts = 0
    while len(pairs) < count and attempts < count * 50:
        attempts += 1
        src = rng.choice(sources)
        ref = rng.choice(refs)
        if src["audio_path"] == ref["audio_path"]:
            continue
        pairs.append((src, ref))
    return pairs


def cell_specs() -> list[dict[str, str]]:
    return [
        {"cell": "en_src_zh_ref_m2f", "source_lang": "en", "ref_lang": "zh", "source_gender": "male", "ref_gender": "female"},
        {"cell": "en_src_zh_ref_f2m", "source_lang": "en", "ref_lang": "zh", "source_gender": "female", "ref_gender": "male"},
        {"cell": "zh_src_en_ref_m2f", "source_lang": "zh", "ref_lang": "en", "source_gender": "male", "ref_gender": "female"},
        {"cell": "zh_src_en_ref_f2m", "source_lang": "zh", "ref_lang": "en", "source_gender": "female", "ref_gender": "male"},
        {"cell": "en_src_en_ref_same_gender", "source_lang": "en", "ref_lang": "en", "source_gender": "any", "ref_gender": "any"},
        {"cell": "zh_src_zh_ref_same_gender", "source_lang": "zh", "ref_lang": "zh", "source_gender": "any", "ref_gender": "any"},
        {"cell": "en_src_zh_ref_same_gender", "source_lang": "en", "ref_lang": "zh", "source_gender": "any", "ref_gender": "any"},
        {"cell": "zh_src_en_ref_same_gender", "source_lang": "zh", "ref_lang": "en", "source_gender": "any", "ref_gender": "any"},
    ]


def build_case(mode: str, cell: dict[str, str], idx: int, src: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    base = {
        "case_id": f"seedtts_{mode}_{cell['cell']}_{idx:06d}",
        "mode": mode,
        "cell": cell["cell"],
        "source_audio": src["audio_path"],
        "source_text": src["source_text"],
        "source_lang": src["lang"],
        "source_gender": src["gender"],
        "source_id": src["id"],
        "timbre_ref_audio": ref["audio_path"],
        "timbre_ref_text": ref["source_text"],
        "ref_lang": ref["lang"],
        "ref_gender": ref["gender"],
        "ref_id": ref["id"],
    }
    if mode == "text":
        base.update(
            {
                "text": src["target_text"],
                "content_ref_text": src["target_text"],
                "eval_text_source": "input_text",
            }
        )
    else:
        base.update(
            {
                "text": "<NO_TEXT>",
                "content_ref_text": src["source_text"],
                "eval_text_source": "source_text",
            }
        )
    return base


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    output_path = Path(args.output_jsonl).expanduser()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists, pass --overwrite: {output_path}")
    records = enrich_records(
        Path(args.seedtts_jsonl).expanduser(),
        load_gender_map(args.speaker_metadata_jsonl),
        bool(args.require_existing_audio),
    )
    groups = group_records(records)
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "seedtts_jsonl": str(Path(args.seedtts_jsonl).expanduser().resolve(strict=False)),
        "records": len(records),
        "per_cell": int(args.per_cell),
        "modes": modes,
        "cells": {},
        "groups": {f"{lang}:{gender}": len(items) for (lang, gender), items in sorted(groups.items())},
    }
    for mode in modes:
        if mode not in {"no_text", "text"}:
            raise ValueError(f"unsupported mode: {mode}")
        for cell in cell_specs():
            pairs = sample_pairs(
                groups,
                source_lang=cell["source_lang"],
                ref_lang=cell["ref_lang"],
                source_gender=cell["source_gender"],
                ref_gender=cell["ref_gender"],
                count=int(args.per_cell),
                rng=rng,
            )
            summary["cells"][f"{mode}:{cell['cell']}"] = len(pairs)
            for idx, (src, ref) in enumerate(pairs):
                rows.append(build_case(mode, cell, idx, src, ref))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["rows"] = len(rows)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[seedtts-vc-val] wrote rows={len(rows)} output={output_path} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
