#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Summarize recipe_final eval grid metrics and bootstrap usage."
    )
    ap.add_argument("--ablation-summary-json", required=True)
    ap.add_argument("--run", action="append", required=True, help="NAME=EVAL_DIR. May be repeated.")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-md", required=True)
    return ap.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value).expanduser()
        return path.name, path
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path).expanduser()


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


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def infer_metadata(run_name: str) -> dict[str, Any]:
    window = None
    cfg = None
    seed = None
    step = None
    for pattern, key, cast in (
        (r"(?:^|_)w([0-9]+(?:p[0-9]+)?)s(?:_|$)", "infer_window_s", lambda v: float(v.replace("p", "."))),
        (r"(?:^|_)cfg([0-9]+(?:p[0-9]+)?)(?:_|$)", "cfg_scale", lambda v: float(v.replace("p", "."))),
        (r"(?:^|_)seed([0-9]+)(?:_|$)", "seed", int),
        (r"(?:^|_)step-?([0-9]+)(?:_|$)", "step", int),
    ):
        match = re.search(pattern, run_name)
        if match:
            value = cast(match.group(1))
            if key == "infer_window_s":
                window = value
            elif key == "cfg_scale":
                cfg = value
            elif key == "seed":
                seed = value
            elif key == "step":
                step = value
    return {
        "step": step,
        "infer_window_s": window,
        "cfg_scale": cfg,
        "seed": seed,
    }


def manifest_bootstrap_summary(run_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(run_dir.glob("manifest*.jsonl")):
        rows.extend(iter_jsonl(path))
    stats_rows = []
    for row in rows:
        stats = row.get("ref_prompt_codec_permutation") or {}
        if isinstance(stats, dict):
            stats_rows.append(stats)
    bootstrap_used = [
        int(bool(stats.get("bootstrap_used")))
        for stats in stats_rows
        if "bootstrap_used" in stats
    ]
    requested = [finite(stats.get("requested_frames")) for stats in stats_rows]
    prompt = [finite(stats.get("prompt_frames")) for stats in stats_rows]
    source = [finite(stats.get("source_frames")) for stats in stats_rows]
    return {
        "manifest_rows": len(rows),
        "permutation_rows": len(stats_rows),
        "bootstrap_used": int(sum(bootstrap_used)) if bootstrap_used else 0,
        "bootstrap_rate": (sum(bootstrap_used) / len(bootstrap_used)) if bootstrap_used else None,
        "requested_frames_mean": mean(requested),
        "prompt_frames_mean": mean(prompt),
        "source_frames_mean": mean(source),
    }


def main() -> int:
    args = parse_args()
    ablation = json.loads(Path(args.ablation_summary_json).read_text(encoding="utf-8"))
    run_metrics = ablation.get("runs") or {}

    rows: list[dict[str, Any]] = []
    for run_spec in args.run:
        run_name, run_dir = parse_run(run_spec)
        metrics = run_metrics.get(run_name, {})
        row = {
            "run": run_name,
            "run_dir": str(run_dir),
            **infer_metadata(run_name),
            **manifest_bootstrap_summary(run_dir),
            "n": metrics.get("n"),
            "failure_rate_cer_gt_threshold": finite(metrics.get("failure_rate_cer_gt_threshold")),
            "cer_mean": finite(metrics.get("cer_mean")),
            "cer_std": finite(metrics.get("cer_std")),
            "coverage_mean": finite(metrics.get("coverage_mean")),
            "repeat_mean": finite(metrics.get("repeat_mean")),
            "tail_silence_ratio_mean": finite(metrics.get("tail_silence_ratio_mean")),
            "total_silence_ratio_mean": finite(metrics.get("total_silence_ratio_mean")),
            "sim_gen_ref_mean": finite(metrics.get("sim_gen_ref_mean")),
            "sim_gen_source_mean": finite(metrics.get("sim_gen_source_mean")),
            "ecapa_sim_gen_ref_mean": finite(metrics.get("ecapa_sim_gen_ref_mean")),
            "ecapa_sim_gen_source_mean": finite(metrics.get("ecapa_sim_gen_source_mean")),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row.get("step") if row.get("step") is not None else -1,
            row.get("infer_window_s") if row.get("infer_window_s") is not None else -1,
            row.get("cfg_scale") if row.get("cfg_scale") is not None else -1,
            row.get("seed") if row.get("seed") is not None else -1,
            str(row.get("run") or ""),
        )
    )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    for path in (output_json, output_csv, output_md):
        path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "failure_cer_threshold": ablation.get("failure_cer_threshold"),
        "ablation_summary_json": str(Path(args.ablation_summary_json)),
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "run",
        "step",
        "infer_window_s",
        "cfg_scale",
        "seed",
        "n",
        "failure_rate_cer_gt_threshold",
        "cer_mean",
        "cer_std",
        "coverage_mean",
        "repeat_mean",
        "tail_silence_ratio_mean",
        "total_silence_ratio_mean",
        "sim_gen_ref_mean",
        "sim_gen_source_mean",
        "ecapa_sim_gen_ref_mean",
        "ecapa_sim_gen_source_mean",
        "bootstrap_used",
        "bootstrap_rate",
        "requested_frames_mean",
        "prompt_frames_mean",
        "source_frames_mean",
        "run_dir",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Recipe Final Eval Grid Summary",
        "",
        f"failure threshold: `CER > {float(ablation.get('failure_cer_threshold', 0.30)):.2f}`",
        "",
        "| run | window | cfg | seed | n | fail | CER | coverage | repeat | WavLM-SV ref | WavLM-SV src | ECAPA ref | ECAPA src | bootstrap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        used = row.get("bootstrap_used")
        total = row.get("permutation_rows")
        if total:
            rate = fmt(row.get("bootstrap_rate"))
            bootstrap = f"{used}/{total} ({rate})" if rate else f"{used}/{total}"
        else:
            bootstrap = ""
        lines.append(
            "| {run} | {win} | {cfg} | {seed} | {n} | {fail} | {cer} | {cov} | {rep} | {ref} | {src} | {ecapa_ref} | {ecapa_src} | {boot} |".format(
                run=row["run"],
                win=fmt(row.get("infer_window_s")),
                cfg=fmt(row.get("cfg_scale")),
                seed=row.get("seed") if row.get("seed") is not None else "",
                n=row.get("n") or "",
                fail=fmt(row.get("failure_rate_cer_gt_threshold")),
                cer=fmt(row.get("cer_mean")),
                cov=fmt(row.get("coverage_mean")),
                rep=fmt(row.get("repeat_mean")),
                ref=fmt(row.get("sim_gen_ref_mean")),
                src=fmt(row.get("sim_gen_source_mean")),
                ecapa_ref=fmt(row.get("ecapa_sim_gen_ref_mean")),
                ecapa_src=fmt(row.get("ecapa_sim_gen_source_mean")),
                boot=bootstrap,
            )
        )
    lines.extend(["", f"CSV: `{output_csv}`", f"JSON: `{output_json}`", ""])
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[recipe-final-summary] wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
