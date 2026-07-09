#!/usr/bin/env python3
"""Summarize ver2.9 speaker-side smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch


METRIC_RE = re.compile(r"([A-Za-z0-9_./-]+)=([^ \t\n]+)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="Training output dir.")
    ap.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint dir, defaults to latest step-*.")
    ap.add_argument("--infer-dir", type=Path, action="append", default=[], help="Optional inference output dir.")
    ap.add_argument(
        "--compare-infer-dirs",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("DIR_A", "DIR_B"),
        help="Compare generated wav hashes for CFG A/B outputs.",
    )
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--output-md", type=Path, default=None)
    return ap.parse_args()


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_train_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "metric_rows": 0, "last": {}, "max": {}}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "step=" not in line or "loss=" not in line:
                continue
            row: dict[str, Any] = {}
            for key, raw_value in METRIC_RE.findall(line):
                if key == "step" and "/" in raw_value:
                    left, right = raw_value.split("/", 1)
                    row["step"] = int(left) if left.isdigit() else left
                    row["max_step"] = int(right) if right.isdigit() else right
                    continue
                value = to_float(raw_value)
                row[key] = value if value is not None else raw_value
            if row:
                rows.append(row)
    max_values: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                max_values[key] = max(max_values.get(key, float("-inf")), float(value))
    return {
        "path": str(path),
        "exists": True,
        "metric_rows": len(rows),
        "rows": rows,
        "last": rows[-1] if rows else {},
        "max": {key: value for key, value in max_values.items() if value != float("-inf")},
    }


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = []
    for path in run_dir.glob("step-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1][1]


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def flatten_tensors(obj: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if torch.is_tensor(obj):
        tensors.append(obj.detach().float().reshape(-1).cpu())
    elif isinstance(obj, dict):
        for value in obj.values():
            tensors.extend(flatten_tensors(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            tensors.extend(flatten_tensors(value))
    return tensors


def tensor_group_stats(obj: Any) -> dict[str, Any]:
    tensors = flatten_tensors(obj)
    if not tensors:
        return {"present": False, "num_tensors": 0, "numel": 0}
    flat = torch.cat(tensors)
    return {
        "present": True,
        "num_tensors": len(tensors),
        "numel": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "mean_abs": float(flat.abs().mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(flat).item()),
        "max_abs": float(flat.abs().max().item()),
        "nonzero": int(torch.count_nonzero(flat).item()),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_checkpoint(checkpoint_dir: Path | None) -> dict[str, Any]:
    if checkpoint_dir is None:
        return {"exists": False}
    adapter_path = checkpoint_dir / "timbre_memory_adapter.pt"
    config_path = checkpoint_dir / "timbre_memory_config.json"
    out: dict[str, Any] = {
        "path": str(checkpoint_dir),
        "exists": checkpoint_dir.exists(),
        "adapter_path": str(adapter_path),
        "config_path": str(config_path),
    }
    if not adapter_path.exists():
        out["adapter_exists"] = False
        return out
    out["adapter_exists"] = True
    config = read_json(config_path)
    state = torch_load(adapter_path)
    gate_logits = state.get("speaker_side_gate_logits")
    gate_tensors = flatten_tensors(gate_logits)
    gate_init = float(config.get("speaker_side_pathway_gate_init", 0.0) or 0.0)
    init_sigmoid = float(torch.sigmoid(torch.tensor(gate_init)).item())
    if gate_tensors:
        flat_logits = torch.cat(gate_tensors)
        gates = torch.sigmoid(flat_logits)
        out["speaker_side_gate"] = {
            "present": True,
            "num_gates": int(gates.numel()),
            "init_logit": gate_init,
            "init_sigmoid": init_sigmoid,
            "mean": float(gates.mean().item()),
            "std": float(gates.std(unbiased=False).item()) if gates.numel() > 1 else 0.0,
            "min": float(gates.min().item()),
            "max": float(gates.max().item()),
            "mean_abs_drift_from_init": float((gates - init_sigmoid).abs().mean().item()),
            "max_abs_drift_from_init": float((gates - init_sigmoid).abs().max().item()),
        }
    else:
        out["speaker_side_gate"] = {"present": False}
    out["speaker_side_adaln"] = tensor_group_stats(state.get("speaker_side_adaln"))
    out["speaker_side_kv_bias"] = tensor_group_stats(state.get("speaker_side_kv_bias"))
    out["progress_stop_head"] = tensor_group_stats(state.get("progress_stop_head"))
    out["null_speaker_embedding"] = tensor_group_stats(state.get("null_speaker_embedding"))
    out["config"] = {
        "speaker_side_pathway_enabled": bool(config.get("speaker_side_pathway_enabled", False)),
        "speaker_side_pathway_layers": config.get("speaker_side_pathway_layers"),
        "speaker_side_pathway_kv_bias": bool(config.get("speaker_side_pathway_kv_bias", False)),
        "speaker_side_pathway_dropout": config.get("speaker_side_pathway_dropout"),
        "progress_loss_weight": config.get("progress_loss_weight"),
        "stop_loss_weight": config.get("stop_loss_weight"),
    }
    return out


def summarize_train_diagnostics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "ver2_9_smoke_train_diagnostics.json"
    if not path.exists():
        return {"path": str(path), "exists": False}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["path"] = str(path)
    payload["exists"] = True
    return payload


def jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def manifest_paths(infer_dir: Path) -> list[Path]:
    return sorted(infer_dir.glob("manifest*.jsonl")) + sorted(infer_dir.glob("**/manifest*.jsonl"))


def summarize_infer_dir(infer_dir: Path) -> dict[str, Any]:
    paths = []
    seen = set()
    for path in manifest_paths(infer_dir):
        resolved = str(path.resolve())
        if resolved not in seen:
            paths.append(path)
            seen.add(resolved)
    rows = []
    for path in paths:
        rows.extend(jsonl_rows(path))
    stats_rows = [row.get("progress_stop_infer_stats") or {} for row in rows]
    stats_rows = [item for item in stats_rows if item]
    out: dict[str, Any] = {
        "path": str(infer_dir),
        "manifest_files": [str(path) for path in paths],
        "rows": len(rows),
        "rows_with_progress_stop_stats": len(stats_rows),
    }
    for key in (
        "forced_rows",
        "steps",
        "stop_prob_max_mean",
        "stop_prob_max_max",
        "progress_value_max_mean",
        "progress_value_max_max",
    ):
        values = [float(item[key]) for item in stats_rows if key in item and item[key] is not None]
        if values:
            out[key] = {
                "mean": sum(values) / len(values),
                "max": max(values),
                "min": min(values),
            }
    return out


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def output_wavs_by_case(infer_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for manifest in manifest_paths(infer_dir):
        for row in jsonl_rows(manifest):
            case_id = str(row.get("case_id") or "")
            output_wav = row.get("output_wav")
            if case_id and output_wav:
                mapping[case_id] = Path(output_wav)
    return mapping


def compare_infer_dirs(dir_a: Path, dir_b: Path) -> dict[str, Any]:
    wavs_a = output_wavs_by_case(dir_a)
    wavs_b = output_wavs_by_case(dir_b)
    common = sorted(set(wavs_a) & set(wavs_b))
    changed = 0
    missing = 0
    examples = []
    for case_id in common:
        hash_a = sha256_file(wavs_a[case_id])
        hash_b = sha256_file(wavs_b[case_id])
        if hash_a is None or hash_b is None:
            missing += 1
            continue
        is_changed = hash_a != hash_b
        changed += int(is_changed)
        if is_changed and len(examples) < 5:
            examples.append(case_id)
    comparable = max(0, len(common) - missing)
    return {
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "common_cases": len(common),
        "missing_outputs": missing,
        "comparable_cases": comparable,
        "changed_cases": changed,
        "changed_fraction": (changed / comparable) if comparable else None,
        "changed_examples": examples,
    }


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def build_checks(summary: dict[str, Any]) -> dict[str, Any]:
    train = summary["train_log"]
    last = train.get("last") or {}
    rows = train.get("rows") or []
    max_values = train.get("max") or {}
    checkpoint = summary["checkpoint"]
    diagnostics = summary.get("train_diagnostics") or {}
    config = checkpoint.get("config") or {}
    gate = checkpoint.get("speaker_side_gate") or {}
    train_gate = diagnostics.get("speaker_side_gate_summary") or {}
    train_gate_by_layer = diagnostics.get("speaker_side_gate_by_layer") or {}
    gate_values = []
    for raw_value in train_gate_by_layer.values():
        if finite_number(raw_value):
            gate_values.append(float(raw_value))
    gate_near_init_count = sum(1 for value in gate_values if 0.47 <= value <= 0.53)
    gate_degenerate_count = sum(1 for value in gate_values if value <= 0.05 or value >= 0.95)
    gate_layer_count = len(gate_values)
    gate_drift_gt_0_03_count = sum(1 for value in gate_values if abs(value - 0.5) > 0.03)
    gate_above_init_count = sum(1 for value in gate_values if value > 0.5)
    gate_below_init_count = sum(1 for value in gate_values if value < 0.5)
    gate_std = float(train_gate.get("std", 0.0) or 0.0)
    tail_lr_values = [
        float(row["lr"])
        for row in rows
        if finite_number(row.get("step")) and int(row["step"]) >= 500 and finite_number(row.get("lr"))
    ]
    first_loss = next((float(row["loss"]) for row in rows if finite_number(row.get("loss"))), None)
    last_loss = float(last["loss"]) if finite_number(last.get("loss")) else None
    stop_stats = diagnostics.get("progress_stop_aux_stats_last_batch") or {}
    stop_pos_prob = float(stop_stats.get("content_order_stop_pos_prob", 0.0) or 0.0)
    stop_neg_prob = float(stop_stats.get("content_order_stop_neg_prob", 0.0) or 0.0)
    checks = {
        "loss_finite": finite_number(last.get("loss")),
        "loss_end_lt_initial": first_loss is not None and last_loss is not None and last_loss < first_loss,
        "lr_tail_min_ge_5e_6": bool(tail_lr_values) and min(tail_lr_values) >= 5.0e-6,
        "speaker_side_adaln_grad_nonzero": float(max_values.get("speaker_side_adaln_grad_norm", 0.0) or 0.0) > 0.0,
        "speaker_side_gate_grad_nonzero": float(max_values.get("speaker_side_gate_grad_norm", 0.0) or 0.0) > 0.0,
        "speaker_side_kv_bias_grad_nonzero": True
        if not bool(config.get("speaker_side_pathway_kv_bias", False))
        else float(max_values.get("speaker_side_kv_bias_grad_norm", 0.0) or 0.0) > 0.0,
        "speaker_side_gate_checkpoint_present": bool(gate.get("present", False)),
        "speaker_side_gate_drift_nonzero": float(gate.get("mean_abs_drift_from_init", 0.0) or 0.0) > 1.0e-6,
        "speaker_side_gate_layer_count_eq_32": gate_layer_count == 32,
        "speaker_side_gate_drift_gt_0_03_count_ge_15": gate_drift_gt_0_03_count >= 15,
        "speaker_side_gate_std_gt_0_005": gate_std > 0.005,
        "speaker_side_gate_not_all_same_direction": bool(gate_values)
        and gate_above_init_count > 0
        and gate_below_init_count > 0,
        "speaker_side_gate_not_degenerate_0_or_1": bool(gate_values) and gate_degenerate_count == 0,
        "progress_stop_aux_present": finite_number(last.get("progress_stop_aux_loss")),
        "stop_pos_minus_neg_gt_0_1": (stop_pos_prob - stop_neg_prob) > 0.1,
    }
    for idx, comparison in enumerate(summary.get("infer_comparisons") or []):
        checks[f"infer_comparison_{idx}_outputs_changed"] = (
            comparison.get("changed_fraction") is not None and float(comparison["changed_fraction"]) >= 0.5
        )
    checks["speaker_side_gate_layer_count_info"] = gate_layer_count
    checks["speaker_side_gate_near_init_count_info"] = gate_near_init_count
    checks["speaker_side_gate_drift_gt_0_03_count_info"] = gate_drift_gt_0_03_count
    checks["speaker_side_gate_above_init_count_info"] = gate_above_init_count
    checks["speaker_side_gate_below_init_count_info"] = gate_below_init_count
    checks["speaker_side_gate_degenerate_count_info"] = gate_degenerate_count
    checks["lr_tail_min_info"] = min(tail_lr_values) if tail_lr_values else None
    checks["loss_initial_info"] = first_loss
    checks["loss_final_info"] = last_loss
    checks["stop_pos_minus_neg_info"] = stop_pos_prob - stop_neg_prob
    checks["all_core_checks_pass"] = all(
        bool(value)
        for key, value in checks.items()
        if not key.startswith("infer_comparison_") and not key.endswith("_info")
    )
    return checks


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Ver2.9 Smoke Summary",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- checkpoint: `{summary['checkpoint'].get('path')}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in summary["checks"].items():
        lines.append(f"- {key}: `{value}`")
    rows = summary["train_log"].get("rows") or []
    if rows:
        lines.extend(["", "## LR And Grad Trajectory", ""])
        lines.append("| step | lr | loss | AdaLN grad | K/V grad | gate grad | gate mean | stop aux |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                "| {step} | {lr} | {loss} | {adaln} | {kv} | {gate_grad} | {gate_mean} | {stop} |".format(
                    step=row.get("step", ""),
                    lr=row.get("lr", ""),
                    loss=row.get("loss", ""),
                    adaln=row.get("speaker_side_adaln_grad_norm", ""),
                    kv=row.get("speaker_side_kv_bias_grad_norm", ""),
                    gate_grad=row.get("speaker_side_gate_grad_norm", ""),
                    gate_mean=row.get("speaker_side_gate_mean", ""),
                    stop=row.get("progress_stop_aux_loss", ""),
                )
            )
    last = summary["train_log"].get("last") or {}
    lines.extend(["", "## Last Train Metrics", ""])
    for key in sorted(last):
        lines.append(f"- {key}: `{last[key]}`")
    gate = summary["checkpoint"].get("speaker_side_gate") or {}
    lines.extend(["", "## Gate", ""])
    for key in sorted(gate):
        lines.append(f"- {key}: `{gate[key]}`")
    diagnostics = summary.get("train_diagnostics") or {}
    if diagnostics.get("exists"):
        lines.extend(["", "## Train Diagnostics", ""])
        gate_summary = diagnostics.get("speaker_side_gate_summary") or {}
        for key in sorted(gate_summary):
            lines.append(f"- gate_{key}: `{gate_summary[key]}`")
        gate_by_layer = diagnostics.get("speaker_side_gate_by_layer") or {}
        if gate_by_layer:
            lines.extend(["", "### Gate By Layer", ""])
            for key in sorted(gate_by_layer, key=lambda item: int(item) if str(item).isdigit() else str(item)):
                lines.append(f"- layer {key}: `{gate_by_layer[key]}`")
        stop_stats = diagnostics.get("progress_stop_aux_stats_last_batch") or {}
        for key in sorted(stop_stats):
            if "stop" in key:
                lines.append(f"- {key}: `{stop_stats[key]}`")
    if summary.get("infer_comparisons"):
        lines.extend(["", "## CFG/Infer Comparisons", ""])
        for item in summary["infer_comparisons"]:
            lines.append(
                "- `{}` vs `{}`: changed {}/{} ({})".format(
                    item["dir_a"],
                    item["dir_b"],
                    item["changed_cases"],
                    item["comparable_cases"],
                    item["changed_fraction"],
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else latest_checkpoint(run_dir)
    train_log = parse_train_log(run_dir / "train.log")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "train_log": train_log,
        "train_diagnostics": summarize_train_diagnostics(run_dir),
        "checkpoint": summarize_checkpoint(checkpoint_dir),
        "infer_dirs": [summarize_infer_dir(path.expanduser().resolve()) for path in args.infer_dir],
        "infer_comparisons": [
            compare_infer_dirs(left.expanduser().resolve(), right.expanduser().resolve())
            for left, right in args.compare_infer_dirs
        ],
    }
    summary["checks"] = build_checks(summary)
    output_json = args.output_json or (run_dir / "ver2_9_smoke_summary.json")
    output_md = args.output_md or (run_dir / "ver2_9_smoke_summary.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, output_md)
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
