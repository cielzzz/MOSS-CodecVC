#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED = ROOT / "trainset/ver2_8_prepared"
SPLITS = ("no_text.train", "no_text.valid", "text.train", "text.valid")


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


def deep_get(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def duration_sec(row: dict[str, Any], role: str, frame_rate: float) -> float | None:
    for key in (
        f"{role}_duration_sec",
        f"{role}_duration",
        f"{role}_audio_duration_sec",
    ):
        value = as_float(row.get(key))
        if value is not None and value > 0:
            return value
    meta = row.get("moss_codecvc_meta") or {}
    for key in (f"{role}_duration_sec", f"{role}_duration"):
        value = as_float(meta.get(key))
        if value is not None and value > 0:
            return value
    frame_key = f"{role}_codec_frames"
    frames = as_float(row.get(frame_key))
    if frames is None:
        frames = as_float(meta.get(frame_key))
    if frames is not None and frames > 0 and frame_rate > 0:
        return frames / frame_rate
    codes_key = "audio_codes" if role == "target" else "reference_audio_codes"
    codes = row.get(codes_key)
    if role == "source" and isinstance(codes, list) and codes:
        frames = len(codes[0]) if codes and isinstance(codes[0], list) and codes and codes[0] and isinstance(codes[0][0], list) else None
    elif role == "target" and isinstance(codes, list):
        frames = len(codes)
    else:
        frames = None
    if frames:
        return float(frames) / frame_rate
    return None


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize(values: list[float]) -> dict[str, Any]:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    return {
        "n": len(vals),
        "p50": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "p99": percentile(vals, 0.99),
        "max": vals[-1] if vals else None,
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize source/target duration distribution for prepared Ver2.8 splits.")
    ap.add_argument("--prepared-dir", action="append", default=[])
    ap.add_argument("--frame-rate", type=float, default=12.5)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-md", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    prepared_dirs = [Path(p).expanduser() for p in args.prepared_dir] or [DEFAULT_PREPARED]
    summary: dict[str, Any] = {"frame_rate": float(args.frame_rate), "prepared_dirs": {}}
    csv_rows: list[dict[str, Any]] = []
    for prepared in prepared_dirs:
        prep_payload: dict[str, Any] = {}
        for split in SPLITS:
            path = prepared / f"{split}.jsonl"
            if not path.exists():
                continue
            source_values: list[float] = []
            target_values: list[float] = []
            rows = 0
            missing_source = 0
            missing_target = 0
            for row in iter_jsonl(path):
                rows += 1
                src = duration_sec(row, "source", float(args.frame_rate))
                tgt = duration_sec(row, "target", float(args.frame_rate))
                if src is None:
                    missing_source += 1
                else:
                    source_values.append(src)
                if tgt is None:
                    missing_target += 1
                else:
                    target_values.append(tgt)
            split_payload = {
                "rows": rows,
                "source": summarize(source_values),
                "target": summarize(target_values),
                "missing_source": missing_source,
                "missing_target": missing_target,
            }
            prep_payload[split] = split_payload
            for role in ("source", "target"):
                item = split_payload[role]
                csv_rows.append(
                    {
                        "prepared_dir": str(prepared),
                        "split": split,
                        "role": role,
                        "rows": rows,
                        "n": item["n"],
                        "p50": item["p50"],
                        "p90": item["p90"],
                        "p99": item["p99"],
                        "max": item["max"],
                        "missing": split_payload[f"missing_{role}"],
                    }
                )
        summary["prepared_dirs"][str(prepared)] = prep_payload

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prepared_dir", "split", "role", "rows", "n", "p50", "p90", "p99", "max", "missing"])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    lines = [
        "# Ver2.8 Prepared Duration Distribution",
        "",
        f"frame_rate: `{float(args.frame_rate):.3f}` codec frames/sec",
        "",
        "| prepared_dir | split | role | rows | n | P50 | P90 | P99 | max | missing |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in csv_rows:
        lines.append(
            "| {prepared_dir} | {split} | {role} | {rows} | {n} | {p50} | {p90} | {p99} | {max} | {missing} |".format(
                prepared_dir=row["prepared_dir"],
                split=row["split"],
                role=row["role"],
                rows=row["rows"],
                n=row["n"],
                p50=fmt(row["p50"]),
                p90=fmt(row["p90"]),
                p99=fmt(row["p99"]),
                max=fmt(row["max"]),
                missing=row["missing"],
            )
        )
    lines.extend(["", f"CSV: `{out_csv}`", f"JSON: `{out_json}`", ""])
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[duration-summary] wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
