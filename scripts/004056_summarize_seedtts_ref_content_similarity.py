#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize generated ASR similarity to timbre-ref text.")
    ap.add_argument("--asr-jsonl", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", default="")
    return ap.parse_args()


def normalize_text(text: str | None) -> str:
    raw = str(text or "").lower()
    chars: list[str] = []
    for ch in raw:
        code = ord(ch)
        if ch.isalnum():
            chars.append(ch)
        elif 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
            chars.append(ch)
        elif ch.isspace():
            continue
    return "".join(chars)


def lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for ch_long in long:
        cur = [0]
        left_top = 0
        for idx, ch_short in enumerate(short, start=1):
            top = prev[idx]
            if ch_long == ch_short:
                value = left_top + 1
            else:
                value = max(top, cur[-1])
            cur.append(value)
            left_top = top
        prev = cur
    return prev[-1]


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float | None]) -> float | None:
    cur = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(cur) / len(cur) if cur else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "ref_content_lcs_f1_mean": mean([finite(row.get("ref_content_lcs_f1")) for row in rows]),
        "ref_content_lcs_recall_mean": mean([finite(row.get("ref_content_lcs_recall")) for row in rows]),
        "ref_content_lcs_precision_mean": mean([finite(row.get("ref_content_lcs_precision")) for row in rows]),
        "cer_mean": mean([finite(row.get("cer_tgt")) for row in rows]),
        "wer_mean": mean([finite(row.get("wer_tgt")) for row in rows]),
    }


def table_value(value: Any) -> str:
    num = finite(value)
    if num is None:
        return ""
    return f"{num:.4f}"


def main() -> int:
    args = parse_args()
    asr_path = Path(args.asr_jsonl)
    rows: list[dict[str, Any]] = []
    for line in asr_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gen = normalize_text(row.get("asr_tgt_text"))
        ref = normalize_text(row.get("timbre_ref_text"))
        lcs = lcs_len(gen, ref)
        precision = float(lcs) / max(1, len(gen))
        recall = float(lcs) / max(1, len(ref))
        f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
        row = dict(row)
        row.update(
            {
                "ref_content_lcs_len": int(lcs),
                "ref_content_lcs_precision": precision,
                "ref_content_lcs_recall": recall,
                "ref_content_lcs_f1": f1,
                "normalized_generated_len": len(gen),
                "normalized_timbre_ref_len": len(ref),
            }
        )
        rows.append(row)

    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_lang_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[str(row.get("cell") or "unknown")].append(row)
        pair = f"{row.get('source_lang') or ''}->{row.get('ref_lang') or ''}"
        by_lang_pair[pair].append(row)

    payload = {
        "asr_jsonl": str(asr_path),
        "overall": summarize(rows),
        "by_cell": {key: summarize(value) for key, value in sorted(by_cell.items())},
        "by_lang_pair": {key: summarize(value) for key, value in sorted(by_lang_pair.items())},
        "top_ref_content_matches": sorted(
            [
                {
                    "case_id": row.get("case_id") or row.get("sample_id"),
                    "cell": row.get("cell"),
                    "source_lang": row.get("source_lang"),
                    "ref_lang": row.get("ref_lang"),
                    "cer_tgt": row.get("cer_tgt"),
                    "ref_content_lcs_f1": row.get("ref_content_lcs_f1"),
                    "ref_content_lcs_recall": row.get("ref_content_lcs_recall"),
                    "asr_tgt_text": row.get("asr_tgt_text"),
                    "timbre_ref_text": row.get("timbre_ref_text"),
                }
                for row in rows
            ],
            key=lambda item: finite(item.get("ref_content_lcs_f1")) or 0.0,
            reverse=True,
        )[:20],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ref-content-sim] wrote {output_json}")

    if args.output_md:
        lines = [
            "# Ref-Content Similarity",
            "",
            f"ASR JSONL: `{asr_path}`",
            "",
            "| group | n | LCS F1 | LCS recall | LCS precision | CER | WER |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        overall = payload["overall"]
        lines.append(
            f"| all | {overall['n']} | {table_value(overall.get('ref_content_lcs_f1_mean'))} | "
            f"{table_value(overall.get('ref_content_lcs_recall_mean'))} | "
            f"{table_value(overall.get('ref_content_lcs_precision_mean'))} | "
            f"{table_value(overall.get('cer_mean'))} | {table_value(overall.get('wer_mean'))} |"
        )
        for key, summary in payload["by_cell"].items():
            lines.append(
                f"| {key} | {summary['n']} | {table_value(summary.get('ref_content_lcs_f1_mean'))} | "
                f"{table_value(summary.get('ref_content_lcs_recall_mean'))} | "
                f"{table_value(summary.get('ref_content_lcs_precision_mean'))} | "
                f"{table_value(summary.get('cer_mean'))} | {table_value(summary.get('wer_mean'))} |"
            )
        lines.append("")
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"[ref-content-sim] wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
