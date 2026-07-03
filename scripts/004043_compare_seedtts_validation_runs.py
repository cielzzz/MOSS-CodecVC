#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple SeedTTS validation eval summaries.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def find_summaries(eval_root: Path) -> list[Path]:
    paths = sorted(eval_root.glob("*/**/*.summary.json"))
    return [p for p in paths if p.is_file() and "/logs/" not in str(p)]


def get_metric(payload: dict[str, Any], group: str, metric: str) -> Any:
    if group == "overall":
        return payload.get("overall", {}).get(metric)
    if group.startswith("mode:"):
        return payload.get("by_mode", {}).get(group.split(":", 1)[1], {}).get(metric)
    return None


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root)
    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in find_summaries(eval_root)]
    payloads.sort(key=lambda item: (float(item.get("overall", {}).get("primary_error", 9999)), str(item.get("run_id", ""))))

    fields = [
        "rank",
        "run_id",
        "run_label",
        "overall_primary",
        "overall_cer",
        "overall_wer",
        "overall_repeat",
        "overall_duration",
        "overall_keep",
        "no_text_primary",
        "no_text_keep",
        "text_primary",
        "text_keep",
        "metrics_csv",
        "model_path",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, payload in enumerate(payloads, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "run_id": payload.get("run_id"),
                    "run_label": payload.get("run_label"),
                    "overall_primary": get_metric(payload, "overall", "primary_error"),
                    "overall_cer": get_metric(payload, "overall", "cer"),
                    "overall_wer": get_metric(payload, "overall", "wer"),
                    "overall_repeat": get_metric(payload, "overall", "repeat"),
                    "overall_duration": get_metric(payload, "overall", "duration"),
                    "overall_keep": get_metric(payload, "overall", "keep"),
                    "no_text_primary": get_metric(payload, "mode:no_text", "primary_error"),
                    "no_text_keep": get_metric(payload, "mode:no_text", "keep"),
                    "text_primary": get_metric(payload, "mode:text", "primary_error"),
                    "text_keep": get_metric(payload, "mode:text", "keep"),
                    "metrics_csv": payload.get("metrics_csv"),
                    "model_path": payload.get("model_path"),
                }
            )

    lines = [
        "# SeedTTS Validation Run Comparison",
        "",
        f"eval_root: `{eval_root}`",
        "",
        "Lower primary error is better. Primary error is CER for zh rows and WER for en rows.",
        "",
        "| rank | run | overall primary | overall CER | overall WER | repeat | duration | keep | no-text primary | no-text keep | text primary | text keep |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, payload in enumerate(payloads, start=1):
        lines.append(
            "| {rank} | {run} | {primary} | {cer} | {wer} | {repeat} | {duration} | {keep} | {ntp} | {ntk} | {tp} | {tk} |".format(
                rank=rank,
                run=payload.get("run_id", ""),
                primary=fmt(get_metric(payload, "overall", "primary_error")),
                cer=fmt(get_metric(payload, "overall", "cer")),
                wer=fmt(get_metric(payload, "overall", "wer")),
                repeat=fmt(get_metric(payload, "overall", "repeat")),
                duration=fmt(get_metric(payload, "overall", "duration")),
                keep=fmt(get_metric(payload, "overall", "keep")),
                ntp=fmt(get_metric(payload, "mode:no_text", "primary_error")),
                ntk=fmt(get_metric(payload, "mode:no_text", "keep")),
                tp=fmt(get_metric(payload, "mode:text", "primary_error")),
                tk=fmt(get_metric(payload, "mode:text", "keep")),
            )
        )
    lines.extend(["", f"CSV: `{output_csv}`", ""])
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare-seedtts] wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
