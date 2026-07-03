#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Ver2.5 no-text validation metrics.csv files."
    )
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-md", required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of largest per-sample improvements/regressions to show.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return sum(kept) / len(kept) if kept else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def row_key(row: dict[str, str]) -> str:
    return str(row.get("case_id") or row.get("sample_id") or "").strip()


def metric_summary(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    return {
        "n": len(rows),
        "cer": mean(finite(row.get("cer_tgt")) for row in rows),
        "wer": mean(finite(row.get("wer_tgt")) for row in rows),
        "repeat": mean(finite(row.get("repeat_score")) for row in rows),
        "duration": mean(finite(row.get("duration_ratio_tgt_src")) for row in rows),
        "keep": sum(1 for row in rows if bool_value(row.get("content_keep"))),
    }


def append_summary_row(
    lines: list[str],
    label: str,
    summary: dict[str, float | int | None],
) -> None:
    lines.append(
        "| {label} | {n} | {cer} | {wer} | {repeat} | {duration} | {keep} |".format(
            label=label,
            n=summary["n"],
            cer=fmt(summary["cer"] if isinstance(summary["cer"], float) else None),
            wer=fmt(summary["wer"] if isinstance(summary["wer"], float) else None),
            repeat=fmt(summary["repeat"] if isinstance(summary["repeat"], float) else None),
            duration=fmt(summary["duration"] if isinstance(summary["duration"], float) else None),
            keep=summary["keep"],
        )
    )


def delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def main() -> int:
    args = parse_args()
    baseline_csv = Path(args.baseline_csv)
    candidate_csv = Path(args.candidate_csv)
    output_md = Path(args.output_md)
    baseline_rows = read_csv(baseline_csv)
    candidate_rows = read_csv(candidate_csv)

    baseline_by_key = {row_key(row): row for row in baseline_rows if row_key(row)}
    candidate_by_key = {row_key(row): row for row in candidate_rows if row_key(row)}
    common_keys = sorted(set(baseline_by_key) & set(candidate_by_key))

    lines: list[str] = [
        "# Ver2.5 No-Text Eval Comparison",
        "",
        f"baseline: `{args.baseline_name}`",
        f"candidate: `{args.candidate_name}`",
        f"baseline_csv: `{baseline_csv}`",
        f"candidate_csv: `{candidate_csv}`",
        f"common_cases: `{len(common_keys)}`",
        "",
        "## Overall",
        "",
        "| run | n | CER | WER | repeat | duration | keep |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_summary = metric_summary([baseline_by_key[key] for key in common_keys])
    candidate_summary = metric_summary([candidate_by_key[key] for key in common_keys])
    append_summary_row(lines, args.baseline_name, baseline_summary)
    append_summary_row(lines, args.candidate_name, candidate_summary)
    lines.append(
        "| delta(candidate-baseline) | {n} | {cer} | {wer} | {repeat} | {duration} | {keep} |".format(
            n=len(common_keys),
            cer=fmt(delta(candidate_summary["cer"], baseline_summary["cer"])),
            wer=fmt(delta(candidate_summary["wer"], baseline_summary["wer"])),
            repeat=fmt(delta(candidate_summary["repeat"], baseline_summary["repeat"])),
            duration=fmt(delta(candidate_summary["duration"], baseline_summary["duration"])),
            keep=int(candidate_summary["keep"]) - int(baseline_summary["keep"]),
        )
    )

    by_cell: dict[str, list[str]] = defaultdict(list)
    for key in common_keys:
        cell = str(candidate_by_key[key].get("cell") or baseline_by_key[key].get("cell") or "unknown")
        by_cell[cell].append(key)

    lines.extend(
        [
            "",
            "## By Cell",
            "",
            "| cell | n | baseline CER | candidate CER | delta CER | baseline WER | candidate WER | delta WER | baseline keep | candidate keep |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell, keys in sorted(by_cell.items()):
        base_group = [baseline_by_key[key] for key in keys]
        cand_group = [candidate_by_key[key] for key in keys]
        base = metric_summary(base_group)
        cand = metric_summary(cand_group)
        lines.append(
            "| {cell} | {n} | {bcer} | {ccer} | {dcer} | {bwer} | {cwer} | {dwer} | {bkeep} | {ckeep} |".format(
                cell=cell,
                n=len(keys),
                bcer=fmt(base["cer"] if isinstance(base["cer"], float) else None),
                ccer=fmt(cand["cer"] if isinstance(cand["cer"], float) else None),
                dcer=fmt(delta(cand["cer"], base["cer"])),
                bwer=fmt(base["wer"] if isinstance(base["wer"], float) else None),
                cwer=fmt(cand["wer"] if isinstance(cand["wer"], float) else None),
                dwer=fmt(delta(cand["wer"], base["wer"])),
                bkeep=base["keep"],
                ckeep=cand["keep"],
            )
        )

    per_case: list[dict[str, object]] = []
    for key in common_keys:
        base = baseline_by_key[key]
        cand = candidate_by_key[key]
        base_cer = finite(base.get("cer_tgt"))
        cand_cer = finite(cand.get("cer_tgt"))
        base_wer = finite(base.get("wer_tgt"))
        cand_wer = finite(cand.get("wer_tgt"))
        per_case.append(
            {
                "key": key,
                "cell": cand.get("cell") or base.get("cell") or "",
                "delta_cer": delta(cand_cer, base_cer),
                "delta_wer": delta(cand_wer, base_wer),
                "baseline_cer": base_cer,
                "candidate_cer": cand_cer,
                "baseline_wer": base_wer,
                "candidate_wer": cand_wer,
                "baseline_asr": base.get("asr_tgt_text", ""),
                "candidate_asr": cand.get("asr_tgt_text", ""),
                "content_ref_text": cand.get("content_ref_text") or base.get("content_ref_text", ""),
            }
        )

    def show_cases(title: str, rows: list[dict[str, object]]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| case | cell | delta CER | baseline CER | candidate CER | delta WER | baseline WER | candidate WER |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows[: max(0, int(args.top_k))]:
            lines.append(
                "| {key} | {cell} | {dcer} | {bcer} | {ccer} | {dwer} | {bwer} | {cwer} |".format(
                    key=row["key"],
                    cell=row["cell"],
                    dcer=fmt(row["delta_cer"] if isinstance(row["delta_cer"], float) else None),
                    bcer=fmt(row["baseline_cer"] if isinstance(row["baseline_cer"], float) else None),
                    ccer=fmt(row["candidate_cer"] if isinstance(row["candidate_cer"], float) else None),
                    dwer=fmt(row["delta_wer"] if isinstance(row["delta_wer"], float) else None),
                    bwer=fmt(row["baseline_wer"] if isinstance(row["baseline_wer"], float) else None),
                    cwer=fmt(row["candidate_wer"] if isinstance(row["candidate_wer"], float) else None),
                )
            )

    finite_cases = [row for row in per_case if isinstance(row["delta_cer"], float)]
    show_cases("Largest Candidate Improvements", sorted(finite_cases, key=lambda row: row["delta_cer"]))
    show_cases("Largest Candidate Regressions", sorted(finite_cases, key=lambda row: row["delta_cer"], reverse=True))

    lines.extend(
        [
            "",
            "## Interpretation Hint",
            "",
            "Negative delta means the candidate is better than the baseline for error metrics.",
            "Positive delta means regression. Keep count should increase or stay flat.",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare] wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
