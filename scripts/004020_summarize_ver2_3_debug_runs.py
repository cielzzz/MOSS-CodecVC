#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "outputs/lora_runs/ver2_3_debug"
RUN_SUBDIRS = {
    "tiny": "tiny_no_text128_ce_ctc_semantic",
    "A": "ablation_a_ce_route",
    "B": "ablation_b_ctc",
    "B_headlr10": "ablation_b_ctc_headlr10",
    "C": "ablation_c_hubert",
    "C_stop": "ablation_c_progress_stop",
    "D": "ablation_d_prosody",
    "E": "ablation_e_progress_stop_only",
}
TRAIN_STEP_RE = re.compile(r"^(\S+) step=(\d+)/(\d+).*? loss=([0-9.]+)(.*)$")


def parse_train_tail(path: Path) -> dict:
    if not path.exists():
        return {}
    matches = []
    for line in path.read_text(errors="ignore").splitlines():
        match = TRAIN_STEP_RE.search(line)
        if match:
            matches.append(match)
    if not matches:
        return {}
    match = matches[-1]
    rest = match.group(5)
    item = {
        "time": match.group(1),
        "step": int(match.group(2)),
        "total": int(match.group(3)),
        "train_loss": float(match.group(4)),
    }
    for key in (
        "content_ctc_aux_loss",
        "content_ctc_loss_raw",
        "content_ctc_loss_weighted",
        "content_ctc_nonblank_post",
        "content_ctc_head_grad_norm",
        "semantic_aux_loss",
        "progress_stop_aux_loss",
        "prosody_aux_loss",
        "route_loss",
    ):
        metric = re.search(rf"{key}=([0-9.eE+-]+)", rest)
        if metric:
            item[key] = float(metric.group(1))
    return item


def parse_eval(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {"eval_count": 0}
    return {"eval_count": len(rows), "eval_last": rows[-1]}


def parse_probes(run_dir: Path) -> dict:
    probes = {}
    for path in sorted(run_dir.glob("ctc_greedy_probe*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ters = [row["token_error_rate"] for row in rows if row.get("token_error_rate") is not None]
        blank_rates = [
            row["blank_frame_rate"] for row in rows if row.get("blank_frame_rate") is not None
        ]
        collapsed_lens = [
            row["collapsed_len"] for row in rows if row.get("collapsed_len") is not None
        ]
        item = {
            "rows": len(rows),
            "exact_rate": sum(bool(row.get("exact_match")) for row in rows) / max(1, len(rows)),
            "mean_token_error_rate": statistics.mean(ters) if ters else None,
            "nonempty_pred_rate": sum(bool(row.get("pred_text")) for row in rows)
            / max(1, len(rows)),
        }
        if blank_rates:
            item["mean_blank_frame_rate"] = statistics.mean(blank_rates)
        if collapsed_lens:
            item["mean_collapsed_len"] = statistics.mean(collapsed_lens)
        probes[path.stem] = item
    return {"ctc_probes": probes} if probes else {}


def summarize_run(name: str, run_dir: Path) -> dict:
    item = {"name": name, "out_dir": str(run_dir)}
    item.update(parse_train_tail(run_dir / "train.log"))
    item.update(parse_eval(run_dir / "eval_loss.jsonl"))
    item.update(parse_probes(run_dir))
    steps = []
    for path in run_dir.glob("step-*"):
        suffix = path.name.split("-", 1)[-1]
        if suffix.isdigit():
            steps.append(int(suffix))
    if steps:
        item["saved_steps"] = sorted(steps)
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Ver2.3 debug runs.")
    parser.add_argument("--runs-root", default=str(DEFAULT_OUT_ROOT), help="Directory containing Ver2.3 debug run subdirectories.")
    parser.add_argument("--output-json", default=str(ROOT / "outputs/lora_runs/ver2_3_debug/summary_latest.json"))
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    runs = {name: runs_root / subdir for name, subdir in RUN_SUBDIRS.items()}
    summary = [summarize_run(name, path) for name, path in runs.items()]
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[summary] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
