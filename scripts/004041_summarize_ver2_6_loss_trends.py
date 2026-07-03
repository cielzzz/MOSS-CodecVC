#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNS = {
    "ver2_6_1": ROOT / "outputs/lora_runs/ver2_6_full/ver2-6-1-p0a-from0-spk-full_20260701-163106-save1000",
    "ver2_6_2": ROOT / "outputs/lora_runs/ver2_6_full/ver2-6-2-p0a-from0-spk-progress-stop-full_20260701-163106-save1000",
    "ver2_6_3": ROOT / "outputs/lora_runs/ver2_6_full/ver2-6-3-p0a-from0-spk-prosody-full_20260701-163106-save1000",
    "ver2_6_4": ROOT / "outputs/lora_runs/ver2_6_full/ver2-6-4-p0a-from0-spk-gate0-full_20260701-163106-save1000",
    "ver2_6_5": ROOT / "outputs/lora_runs/ver2_6_full/ver2-6-5-p0c-from0-spk-full_20260701-163106-save1000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Ver2.6 train/eval loss trends.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/reports/ver2_6_loss_trends"))
    parser.add_argument("--eval-step", type=int, default=10000)
    parser.add_argument("--tail-points", type=int, default=50)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional run spec: run_id=/path/to/run_dir. Defaults to the five Ver2.6 runs.",
    )
    return parser.parse_args()


def parse_run_specs(items: list[str]) -> dict[str, Path]:
    if not items:
        return dict(DEFAULT_RUNS)
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --run spec, expected id=path: {item}")
        run_id, path = item.split("=", 1)
        out[run_id.strip()] = Path(path).expanduser()
    return out


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def parse_train_log(path: Path, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    key_value = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if " step=" not in line or " loss=" not in line:
            continue
        row: dict[str, Any] = {"run_id": run_id, "timestamp": line.split(" ", 1)[0]}
        for key, value in key_value.findall(line):
            if key == "step" and "/" in value:
                step, total = value.split("/", 1)
                row["step"] = int(step)
                row["max_steps"] = int(total)
                continue
            num = finite(value)
            row[key] = num if num is not None else value
        if "step" in row:
            rows.append(row)
    return rows


def parse_eval_loss(path: Path, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["run_id"] = run_id
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_run(run_id: str, run_dir: Path, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], *, tail_points: int, eval_step: int) -> dict[str, Any]:
    latest_train = train_rows[-1] if train_rows else {}
    tail = train_rows[-max(1, tail_points) :]
    train_losses = [finite(row.get("loss")) for row in tail]
    train_losses = [x for x in train_losses if x is not None]
    eval_losses = [row for row in eval_rows if finite(row.get("loss")) is not None]
    latest_eval = eval_losses[-1] if eval_losses else {}
    best_eval = min(eval_losses, key=lambda row: float(row["loss"])) if eval_losses else {}
    eval_at_step = next((row for row in eval_losses if int(row.get("step", -1)) == eval_step), {})
    checkpoints = sorted(
        int(p.name.split("-", 1)[1])
        for p in run_dir.glob("step-*")
        if p.is_dir() and p.name.split("-", 1)[1].isdigit()
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "latest_train_step": latest_train.get("step"),
        "latest_train_loss": latest_train.get("loss"),
        "tail_train_loss_mean": mean(train_losses) if train_losses else None,
        "latest_lr": latest_train.get("lr"),
        "latest_eval_step": latest_eval.get("step"),
        "latest_eval_loss": latest_eval.get("loss"),
        f"eval_loss_step_{eval_step}": eval_at_step.get("loss"),
        "best_eval_step": best_eval.get("step"),
        "best_eval_loss": best_eval.get("loss"),
        "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        "num_checkpoints": len(checkpoints),
        "num_train_points": len(train_rows),
        "num_eval_points": len(eval_rows),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    runs = parse_run_specs(args.run)

    all_train: list[dict[str, Any]] = []
    all_eval: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run_id, run_dir in runs.items():
        train_rows = parse_train_log(run_dir / "train.log", run_id)
        eval_rows = parse_eval_loss(run_dir / "eval_loss.jsonl", run_id)
        all_train.extend(train_rows)
        all_eval.extend(eval_rows)
        summaries.append(
            summarize_run(
                run_id,
                run_dir,
                train_rows,
                eval_rows,
                tail_points=int(args.tail_points),
                eval_step=int(args.eval_step),
            )
        )

    write_csv(
        output_dir / "train_loss_points.csv",
        all_train,
        [
            "run_id",
            "timestamp",
            "step",
            "max_steps",
            "loss",
            "lr",
            "lora_grad_norm",
            "timbre_adapter_grad_norm",
            "source_semantic_grad_norm",
            "routing_gate_grad_norm",
            "source_semantic_gate_grad_norm",
            "speaker_aux_loss",
            "progress_stop_aux_loss",
            "prosody_aux_loss",
            "source_semantic_gate_mean",
            "source_semantic_delta_ratio",
            "source_semantic_gate_delta_max",
            "role_gate_mean",
            "prosody_head_gate_mean",
            "timbre_head_gate_mean",
        ],
    )
    write_csv(
        output_dir / "eval_loss_points.csv",
        all_eval,
        [
            "run_id",
            "step",
            "loss",
            "speaker_aux_loss",
            "route_loss",
            "progress_stop_aux_loss",
            "prosody_aux_loss",
            "samples",
            "batches",
            "elapsed_sec",
            "reason",
        ],
    )
    write_csv(
        output_dir / "loss_summary.csv",
        summaries,
        [
            "run_id",
            "latest_train_step",
            "latest_train_loss",
            "tail_train_loss_mean",
            "latest_lr",
            "latest_eval_step",
            "latest_eval_loss",
            f"eval_loss_step_{int(args.eval_step)}",
            "best_eval_step",
            "best_eval_loss",
            "latest_checkpoint",
            "num_checkpoints",
            "num_train_points",
            "num_eval_points",
            "run_dir",
        ],
    )
    (output_dir / "loss_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Ver2.6 Loss Trends",
        "",
        f"eval_step: `{int(args.eval_step)}`",
        f"tail_points: `{int(args.tail_points)}`",
        "",
        "| run | latest train step | latest train loss | tail train mean | latest eval step | latest eval loss | eval@step | best eval | latest ckpt |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {run_id} | {lts} | {ltl} | {tlm} | {les} | {lel} | {eval_step_loss} | step {bes}: {bel} | {ckpt} |".format(
                run_id=row["run_id"],
                lts=row.get("latest_train_step") or "",
                ltl=fmt(row.get("latest_train_loss")),
                tlm=fmt(row.get("tail_train_loss_mean")),
                les=row.get("latest_eval_step") or "",
                lel=fmt(row.get("latest_eval_loss")),
                eval_step_loss=fmt(row.get(f"eval_loss_step_{int(args.eval_step)}")),
                bes=row.get("best_eval_step") or "",
                bel=fmt(row.get("best_eval_loss")),
                ckpt=row.get("latest_checkpoint") or "",
            )
        )
    lines.extend(
        [
            "",
            f"CSV: `{output_dir / 'loss_summary.csv'}`",
            f"Train points: `{output_dir / 'train_loss_points.csv'}`",
            f"Eval points: `{output_dir / 'eval_loss_points.csv'}`",
            "",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[loss-summary] wrote {output_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
