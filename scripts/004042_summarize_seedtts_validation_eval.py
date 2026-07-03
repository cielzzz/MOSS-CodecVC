#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SeedTTS validation ASR metrics for one run.")
    parser.add_argument("--asr-jsonl", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--model-path", default="")
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def primary_error(row: dict[str, Any]) -> float | None:
    lang = str(row.get("language") or row.get("source_lang") or "").lower()
    if lang.startswith("zh"):
        return finite(row.get("cer_tgt"))
    if lang.startswith("en"):
        return finite(row.get("wer_tgt"))
    cer = finite(row.get("cer_tgt"))
    wer = finite(row.get("wer_tgt"))
    if cer is not None and wer is not None:
        return min(cer, wer)
    return cer if cer is not None else wer


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "primary_error": mean(primary_error(row) for row in rows),
        "cer": mean(finite(row.get("cer_tgt")) for row in rows),
        "wer": mean(finite(row.get("wer_tgt")) for row in rows),
        "repeat": mean(finite(row.get("repeat_score")) for row in rows),
        "duration": mean(finite(row.get("duration_ratio_tgt_src")) for row in rows),
        "keep": sum(1 for row in rows if bool_value(row.get("content_keep"))),
    }


def table_row(label: str, row: dict[str, Any]) -> str:
    return "| {label} | {n} | {primary} | {cer} | {wer} | {repeat} | {duration} | {keep} |".format(
        label=label,
        n=row["n"],
        primary=fmt(row["primary_error"]),
        cer=fmt(row["cer"]),
        wer=fmt(row["wer"]),
        repeat=fmt(row["repeat"]),
        duration=fmt(row["duration"]),
        keep=row["keep"],
    )


def main() -> int:
    args = parse_args()
    asr_jsonl = Path(args.asr_jsonl)
    metrics_csv = Path(args.metrics_csv)
    summary_md = Path(args.summary_md)
    summary_json = Path(args.summary_json) if args.summary_json else summary_md.with_suffix(".json")
    rows = [json.loads(line) for line in asr_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

    fields = [
        "sample_id",
        "case_id",
        "run_id",
        "mode",
        "cell",
        "language",
        "source_lang",
        "ref_lang",
        "cer_tgt",
        "wer_tgt",
        "repeat_score",
        "duration_ratio_tgt_src",
        "content_keep",
        "content_filter_reason",
        "target_audio",
        "asr_tgt_text",
        "content_ref_text",
    ]
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_mode_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        cell = str(row.get("cell") or "unknown")
        by_mode[mode].append(row)
        by_cell[cell].append(row)
        by_mode_cell[f"{mode}:{cell}"].append(row)

    payload = {
        "run_id": args.run_id,
        "run_label": args.run_label,
        "model_path": args.model_path,
        "asr_jsonl": str(asr_jsonl),
        "metrics_csv": str(metrics_csv),
        "overall": summary(rows),
        "by_mode": {key: summary(group) for key, group in sorted(by_mode.items())},
        "by_cell": {key: summary(group) for key, group in sorted(by_cell.items())},
        "by_mode_cell": {key: summary(group) for key, group in sorted(by_mode_cell.items())},
        "filter_reasons": Counter(str(row.get("content_filter_reason") or "keep") for row in rows),
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SeedTTS Validation Eval",
        "",
        f"run_id: `{args.run_id}`",
        f"label: `{args.run_label}`",
        f"model: `{args.model_path}`",
        f"rows: `{len(rows)}`",
        "",
        "Primary error means CER for zh rows and WER for en rows.",
        "",
        "## Overall",
        "",
        "| group | n | primary error | CER | WER | repeat | duration | keep |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("all", payload["overall"]),
    ]
    lines.extend(["", "## By Mode", "", "| mode | n | primary error | CER | WER | repeat | duration | keep |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for key, item in payload["by_mode"].items():
        lines.append(table_row(key, item))
    lines.extend(["", "## By Cell", "", "| cell | n | primary error | CER | WER | repeat | duration | keep |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for key, item in payload["by_cell"].items():
        lines.append(table_row(key, item))
    lines.extend(["", "## By Mode/Cell", "", "| mode_cell | n | primary error | CER | WER | repeat | duration | keep |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for key, item in payload["by_mode_cell"].items():
        lines.append(table_row(key, item))
    lines.extend(["", "## Filter Reasons", ""])
    for reason, count in payload["filter_reasons"].most_common():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", f"metrics: `{metrics_csv}`", f"asr_jsonl: `{asr_jsonl}`", f"summary_json: `{summary_json}`", ""])
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[seedtts-summary] wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
