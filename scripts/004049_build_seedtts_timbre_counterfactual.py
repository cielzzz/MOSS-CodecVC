#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a SeedTTS no-text timbre counterfactual validation JSONL.")
    ap.add_argument("--input-jsonl", default=str(DEFAULT_VALIDATION_JSONL))
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--num-sources", type=int, default=20)
    ap.add_argument("--refs-per-source", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


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


def cell_tag(cell: str) -> str:
    cell = str(cell or "")
    for tag in ("m2f", "f2m", "same_gender"):
        if tag in cell:
            return tag
    return "unknown"


def safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return out[:160] or "case"


def round_robin_sources(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[str(row.get("cell") or "unknown")].append(row)
    selected: list[dict[str, Any]] = []
    cells = sorted(by_cell)
    while len(selected) < n and cells:
        progressed = False
        for cell in list(cells):
            bucket = by_cell[cell]
            if not bucket:
                cells.remove(cell)
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break
    return selected


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    input_jsonl = Path(args.input_jsonl).expanduser()
    output_jsonl = Path(args.output_jsonl).expanduser()
    summary_json = Path(args.summary_json).expanduser()
    rows = [row for row in iter_jsonl(input_jsonl) if str(row.get("mode") or "") == "no_text"]
    if not rows:
        raise RuntimeError(f"no no_text rows found in {input_jsonl}")
    rows = list(rows)
    rng.shuffle(rows)
    sources = round_robin_sources(rows, int(args.num_sources))
    if len(sources) < int(args.num_sources):
        raise RuntimeError(f"only selected {len(sources)} sources, requested {args.num_sources}")

    refs_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ref_audio: set[str] = set()
    for row in rows:
        ref_audio = str(row.get("timbre_ref_audio") or "")
        if not ref_audio or ref_audio in seen_ref_audio:
            continue
        seen_ref_audio.add(ref_audio)
        refs_by_tag[cell_tag(str(row.get("cell") or ""))].append(row)
    ref_tags = ["m2f", "f2m", "same_gender"]
    for tag in ref_tags:
        if not refs_by_tag[tag]:
            raise RuntimeError(f"no reference candidates for tag={tag}")

    out_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "num_sources": len(sources),
        "refs_per_source": int(args.refs_per_source),
        "seed": int(args.seed),
        "sources": [],
    }
    for src_idx, source in enumerate(sources):
        src_case_id = str(source.get("case_id") or f"source_{src_idx:03d}")
        source_summary = {
            "source_index": src_idx,
            "source_case_id": src_case_id,
            "source_cell": source.get("cell"),
            "source_audio": source.get("source_audio"),
            "refs": [],
        }
        chosen_refs: list[tuple[str, dict[str, Any]]] = []
        for ref_idx in range(int(args.refs_per_source)):
            tag = ref_tags[ref_idx % len(ref_tags)]
            bucket = refs_by_tag[tag]
            ref_row = bucket[(src_idx * int(args.refs_per_source) + ref_idx) % len(bucket)]
            chosen_refs.append((tag, ref_row))
        used_ref_audio: set[str] = set()
        for ref_idx, (tag, ref_row) in enumerate(chosen_refs):
            ref_audio = str(ref_row.get("timbre_ref_audio") or "")
            if ref_audio in used_ref_audio:
                raise RuntimeError(f"duplicate ref for {src_case_id}: {ref_audio}")
            used_ref_audio.add(ref_audio)
            ref_case_id = str(ref_row.get("case_id") or f"ref_{ref_idx:03d}")
            case_id = f"{safe_id(src_case_id)}__timbre_swap_{ref_idx}_{tag}_{safe_id(ref_case_id)}"
            out = dict(source)
            out.update(
                {
                    "case_id": case_id,
                    "mode": "no_text",
                    "text": "<NO_TEXT>",
                    "timbre_ref_audio": ref_row.get("timbre_ref_audio"),
                    "timbre_ref_text": ref_row.get("timbre_ref_text"),
                    "ref_lang": ref_row.get("ref_lang"),
                    "ref_gender": ref_row.get("ref_gender"),
                    "ref_id": ref_row.get("ref_id"),
                    "source_case_id": src_case_id,
                    "source_cell": source.get("cell"),
                    "ref_case_id": ref_case_id,
                    "ref_cell": ref_row.get("cell"),
                    "ref_swap_index": ref_idx,
                    "ref_swap_tag": tag,
                    "counterfactual_group": src_case_id,
                    "content_ref_text": source.get("source_text") or source.get("content_ref_text"),
                    "eval_text_source": "source_text",
                }
            )
            out_rows.append(out)
            source_summary["refs"].append(
                {
                    "ref_swap_index": ref_idx,
                    "ref_swap_tag": tag,
                    "ref_case_id": ref_case_id,
                    "ref_cell": ref_row.get("cell"),
                    "timbre_ref_audio": ref_audio,
                }
            )
        summary["sources"].append(source_summary)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["rows"] = len(out_rows)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[counterfactual] wrote rows={len(out_rows)} output={output_jsonl}")
    print(f"[counterfactual] wrote summary={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
