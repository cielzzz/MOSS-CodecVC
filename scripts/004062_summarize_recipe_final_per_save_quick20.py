#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_LABEL = (
    "codecvc-ver2-8-timbre-repair-"
    "recipe_final_varlen_block_permuted_cref_prompt_a4_refsup_cosramp_infonce_dropout"
)
DEFAULT_RUN_DIR = (
    ROOT
    / "outputs/lora_runs/"
    "ver2_8_timbre_repair_recipe_final_varlen_block_permuted_cref_prompt_a4_refsup_cosramp_infonce_dropout_steps30000"
)
DEFAULT_EVAL_ROOT = ROOT / "testset/outputs/ver2_8_timbre_repair_quick_eval"
DEFAULT_QUICK20_INFERENCE_CONFIG: dict[str, Any] = {
    "source_semantic_monotonic_bias_strength": 0.0,
    "temperature": 0.7,
    "no_text_audio_temperature": 1.1,
    "no_text_audio_top_p": 0.7,
    "no_text_audio_top_k": 20,
    "audio_temperature": 1.1,
    "audio_top_p": 0.7,
    "audio_top_k": 20,
    "timbre_cfg_scale": 1.0,
    "ref_prompt_codec_permutation_enabled": True,
    "ref_prompt_codec_permutation_mode": "block_shuffle",
    "ref_prompt_codec_permutation_min_seconds": 8.0,
    "ref_prompt_codec_permutation_max_seconds": 8.0,
    "ref_prompt_codec_permutation_frame_rate": 12.5,
    "ref_prompt_codec_permutation_block_seconds": 0.4,
    "ref_prompt_codec_permutation_bootstrap": "block",
}


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_kv_tokens(line: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        if key == "step" and "/" in raw:
            cur, total = raw.split("/", 1)
            values["step"] = int(cur)
            values["total_steps"] = int(total)
            continue
        number = finite(raw)
        if number is not None:
            values[key] = number
        else:
            values[key] = raw
    return values


def parse_train_log(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+(?P<body>.*)", line)
        if not match:
            continue
        payload = parse_kv_tokens(match.group("body"))
        step = payload.get("step")
        if isinstance(step, int):
            payload["timestamp"] = match.group("ts")
            rows[step] = payload
    return rows


def parse_eval_log(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+eval\s+(?P<body>.*)", line)
        if not match:
            continue
        payload = parse_kv_tokens(match.group("body"))
        step = payload.get("step")
        split = payload.get("split")
        if isinstance(step, (int, float)) and split:
            payload["timestamp"] = match.group("ts")
            rows[(int(step), str(split))] = payload
    return rows


def nearest_train_row(log_rows: dict[int, dict[str, Any]], step: int) -> dict[str, Any] | None:
    if not log_rows:
        return None
    if step in log_rows:
        return log_rows[step]
    nearest = min(log_rows, key=lambda item: abs(item - step))
    row = dict(log_rows[nearest])
    row["nearest_logged_step"] = nearest
    return row


def manifest_bootstrap_summary(run_dir: Path) -> dict[str, Any]:
    stats_rows: list[dict[str, Any]] = []
    for manifest in sorted(run_dir.glob("manifest*.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stats = row.get("ref_prompt_codec_permutation") or {}
            if stats:
                stats_rows.append(stats)
    used = [int(bool(row.get("bootstrap_used"))) for row in stats_rows if "bootstrap_used" in row]
    requested = [finite(row.get("requested_frames")) for row in stats_rows]
    prompt = [finite(row.get("prompt_frames")) for row in stats_rows]
    return {
        "bootstrap_cases": len(used),
        "bootstrap_used": int(sum(used)) if used else None,
        "bootstrap_rate": (sum(used) / len(used)) if used else None,
        "requested_frames_mean": mean(requested),
        "prompt_frames_mean": mean(prompt),
    }


def manifest_inference_config(run_dir: Path) -> dict[str, Any]:
    config = dict(DEFAULT_QUICK20_INFERENCE_CONFIG)
    for manifest in sorted(run_dir.glob("manifest*.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("source_semantic_monotonic_bias_strength", "timbre_cfg_scale"):
                if row.get(key) is not None:
                    config[key] = row[key]
            permutation = row.get("ref_prompt_codec_permutation") or {}
            if permutation:
                config["ref_prompt_codec_permutation"] = permutation
            return config
    return config


def mean(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return sum(kept) / len(kept) if kept else None


def asr_failure_rate(asr_jsonl: Path, threshold: float) -> float | None:
    if not asr_jsonl.is_file():
        return None
    values: list[float] = []
    for line in asr_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = finite(row.get("cer_tgt"))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(1 for value in values if value > threshold) / len(values)


def discover_steps(run_dir: Path, eval_root: Path, run_label: str, seed: str) -> list[int]:
    steps: set[int] = set()
    for path in run_dir.glob("step-*"):
        if path.is_dir():
            try:
                step = int(path.name.split("-", 1)[1])
            except Exception:
                continue
            if step % 2000 == 0:
                steps.add(step)
    pattern = f"{run_label}_step-*_quick20_d2d3_seed{seed}"
    for path in eval_root.glob(pattern):
        match = re.search(r"_step-(\d+)_quick20_", path.name)
        if match:
            step = int(match.group(1))
            if step % 2000 == 0:
                steps.add(step)
    return sorted(steps)


def summarize_step(
    *,
    step: int,
    run_dir: Path,
    eval_root: Path,
    run_label: str,
    seed: str,
    log_rows: dict[int, dict[str, Any]],
    eval_rows: dict[tuple[int, str], dict[str, Any]],
    failure_threshold: float,
) -> dict[str, Any]:
    step_label = f"step-{step}"
    run_id = f"{run_label}_{step_label}_quick20_d2d3_seed{seed}"
    output_dir = eval_root / run_id
    summary = load_json(output_dir / f"{run_id}.summary.json") or {}
    speaker = load_json(output_dir / f"{run_id}.speaker_sim_summary.json") or {}
    ref_content = load_json(output_dir / f"{run_id}.ref_content_similarity_summary.json") or {}
    overall = summary.get("overall") or {}
    speaker_run = ((speaker.get("runs") or {}).get(run_id) or {}).get("all") or {}
    ref_overall = ref_content.get("overall") or {}
    asr_jsonl = output_dir / f"{run_id}.asr_eval.jsonl"
    log_row = nearest_train_row(log_rows, step) or {}
    bootstrap = manifest_bootstrap_summary(output_dir) if output_dir.is_dir() else {}
    complete = all(
        [
            output_dir.is_dir(),
            (output_dir / f"{run_id}.summary.json").is_file(),
            (output_dir / f"{run_id}.speaker_sim_summary.json").is_file(),
            (output_dir / f"{run_id}.ref_content_similarity_summary.json").is_file(),
            asr_jsonl.is_file(),
        ]
    )
    return {
        "step": step,
        "run_id": run_id,
        "checkpoint_exists": (run_dir / step_label).is_dir(),
        "quick20_complete": complete,
        "output_dir": str(output_dir),
        "inference_config": manifest_inference_config(output_dir) if output_dir.is_dir() else dict(DEFAULT_QUICK20_INFERENCE_CONFIG),
        "ramp_phase": "cosine_ramp" if step < 4000 else ("ramp_boundary" if step == 4000 else "post_ramp"),
        "train_timestamp": log_row.get("timestamp"),
        "logged_step": log_row.get("step"),
        "nearest_logged_step": log_row.get("nearest_logged_step", log_row.get("step")),
        "loss": finite(log_row.get("loss")),
        "spk_w": finite(log_row.get("spk_w")),
        "srcsup_w": finite(log_row.get("srcsup_w")),
        "ref_speaker_cos": finite(log_row.get("ref_speaker_cos")),
        "ref_content_cos": finite(log_row.get("ref_content_cos")),
        "eval_seen_loss": finite((eval_rows.get((step, "seen")) or {}).get("loss")),
        "eval_unseen_loss": finite((eval_rows.get((step, "unseen")) or {}).get("loss")),
        "eval_seen_samples": (eval_rows.get((step, "seen")) or {}).get("samples"),
        "eval_unseen_samples": (eval_rows.get((step, "unseen")) or {}).get("samples"),
        "n": overall.get("n"),
        "cer": finite(overall.get("cer")),
        "wer": finite(overall.get("wer")),
        "keep": overall.get("keep"),
        "failure_rate_cer_gt_threshold": asr_failure_rate(asr_jsonl, failure_threshold),
        "sim_gen_ref_mean": finite(speaker_run.get("sim_gen_ref_mean")),
        "sim_gen_source_mean": finite(speaker_run.get("sim_gen_source_mean")),
        "ref_content_lcs_f1_mean": finite(ref_overall.get("ref_content_lcs_f1_mean")),
        **bootstrap,
    }


def fmt(value: Any, digits: int = 3) -> str:
    number = finite(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    failure_threshold: float,
    failure_flat_tolerance: float,
) -> None:
    decision = stop_continue_decision(rows, failure_flat_tolerance=failure_flat_tolerance)
    lines = [
        "# Recipe Final Per-Save Quick20",
        "",
        "Protocol: `quick20`, 8s S2 inference window, block bootstrap enabled, `CFG=1.0`.",
        f"Failure threshold: `CER > {failure_threshold:.2f}`.",
        "Inference config: monotonic bias `0.0`; `TEMPERATURE=0.7`; "
        "`NO_TEXT_AUDIO_TEMPERATURE=1.1`; `NO_TEXT_AUDIO_TOP_P=0.7`; "
        "`NO_TEXT_AUDIO_TOP_K=20`; `AUDIO_TEMPERATURE=1.1`; "
        "`AUDIO_TOP_P=0.7`; `AUDIO_TOP_K=20`.",
        "",
        "| step | ramp | complete | CER | fail | keep | sim gen-ref | sim gen-src | ref-content F1 | bootstrap | spk_w | srcsup_w | ref_speaker_cos | ref_content_cos |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        used = row.get("bootstrap_used")
        cases = row.get("bootstrap_cases")
        if used is None or cases in (None, 0):
            bootstrap = ""
        else:
            bootstrap = f"{used}/{cases} ({fmt(row.get('bootstrap_rate'))})"
        lines.append(
            "| {step} | {ramp} | {complete} | {cer} | {fail} | {keep} | {sim_ref} | {sim_src} | {f1} | {boot} | {spk} | {srcsup} | {ref_spk} | {ref_content} |".format(
                step=row["step"],
                ramp=row.get("ramp_phase") or "",
                complete="yes" if row.get("quick20_complete") else "no",
                cer=fmt(row.get("cer")),
                fail=fmt(row.get("failure_rate_cer_gt_threshold")),
                keep=row.get("keep") if row.get("keep") is not None else "",
                sim_ref=fmt(row.get("sim_gen_ref_mean")),
                sim_src=fmt(row.get("sim_gen_source_mean")),
                f1=fmt(row.get("ref_content_lcs_f1_mean")),
                boot=bootstrap,
                spk=fmt(row.get("spk_w")),
                srcsup=fmt(row.get("srcsup_w")),
                ref_spk=fmt(row.get("ref_speaker_cos")),
                ref_content=fmt(row.get("ref_content_cos")),
            )
        )
    lines.extend(
        [
            "",
            "## Teacher-Forced Eval",
            "",
            "| step | seen loss | unseen loss | seen samples | unseen samples |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {step} | {seen} | {unseen} | {seen_n} | {unseen_n} |".format(
                step=row["step"],
                seen=fmt(row.get("eval_seen_loss")),
                unseen=fmt(row.get("eval_unseen_loss")),
                seen_n=row.get("eval_seen_samples") if row.get("eval_seen_samples") is not None else "",
                unseen_n=row.get("eval_unseen_samples") if row.get("eval_unseen_samples") is not None else "",
            )
        )
    if not rows:
        lines.append("")
        lines.append("No recipe_final save checkpoints or quick20 outputs found yet.")
    lines.extend(
        [
            "",
            "## Stop/Continue Audit",
            "",
            "Rule: stop at 30k only if the last 3 completed checkpoints have "
            "`sim_gen_ref_mean` slope `< 0.01` per save interval and failure-rate range "
            f"`<= {failure_flat_tolerance:.2f}`; otherwise report and continue with warm restart peak `3e-6` +10k.",
            "",
            f"- status: `{decision['status']}`",
            f"- recommendation: `{decision['recommendation']}`",
            f"- last3_steps: `{decision.get('last3_steps')}`",
            f"- sim_slope_per_save: `{fmt(decision.get('sim_slope_per_save'))}`",
            f"- failure_range: `{fmt(decision.get('failure_range'))}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stop_continue_decision(rows: list[dict[str, Any]], failure_flat_tolerance: float) -> dict[str, Any]:
    complete = [
        row
        for row in sorted(rows, key=lambda item: int(item["step"]))
        if row.get("quick20_complete")
        and finite(row.get("sim_gen_ref_mean")) is not None
        and finite(row.get("failure_rate_cer_gt_threshold")) is not None
    ]
    if len(complete) < 3:
        return {
            "status": "insufficient_data",
            "recommendation": "wait",
            "complete_checkpoints": len(complete),
            "required_complete_checkpoints": 3,
            "last3_steps": [row["step"] for row in complete[-3:]],
            "sim_slope_per_save": None,
            "failure_range": None,
            "failure_flat_tolerance": failure_flat_tolerance,
        }
    last3 = complete[-3:]
    sims = [finite(row.get("sim_gen_ref_mean")) for row in last3]
    failures = [finite(row.get("failure_rate_cer_gt_threshold")) for row in last3]
    assert all(value is not None for value in sims)
    assert all(value is not None for value in failures)
    sim_slope = (float(sims[-1]) - float(sims[0])) / max(1, len(sims) - 1)
    failure_range = max(float(value) for value in failures) - min(float(value) for value in failures)
    sim_flat = sim_slope < 0.01
    failure_flat = failure_range <= (failure_flat_tolerance + 1e-12)
    return {
        "status": "ready" if last3[-1]["step"] >= 30000 else "preview",
        "recommendation": "stop" if sim_flat and failure_flat else "report_then_warm_restart",
        "last3_steps": [row["step"] for row in last3],
        "last3_sim_gen_ref_mean": sims,
        "last3_failure_rate": failures,
        "sim_slope_per_save": sim_slope,
        "sim_slope_threshold": 0.01,
        "failure_range": failure_range,
        "failure_flat_tolerance": failure_flat_tolerance,
        "sim_flat": sim_flat,
        "failure_flat": failure_flat,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize recipe_final per-save quick20 metrics.")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    ap.add_argument("--seed", default="1234")
    ap.add_argument("--train-log", type=Path, default=None)
    ap.add_argument("--failure-cer-threshold", type=float, default=0.30)
    ap.add_argument("--failure-flat-tolerance", type=float, default=0.10)
    ap.add_argument("--output-json", type=Path, default=ROOT / "docs/assets/ver2_8_recipe_final_per_save_quick20_20260705.json")
    ap.add_argument("--output-md", type=Path, default=ROOT / "docs/ver2_8_recipe_final_per_save_quick20_20260705.md")
    args = ap.parse_args()

    train_log = args.train_log or (args.run_dir / "train.log")
    log_rows = parse_train_log(train_log)
    eval_rows = parse_eval_log(train_log)
    steps = discover_steps(args.run_dir, args.eval_root, args.run_label, args.seed)
    rows = [
        summarize_step(
            step=step,
            run_dir=args.run_dir,
            eval_root=args.eval_root,
            run_label=args.run_label,
            seed=args.seed,
            log_rows=log_rows,
            eval_rows=eval_rows,
            failure_threshold=float(args.failure_cer_threshold),
        )
        for step in steps
    ]
    payload = {
        "run_dir": str(args.run_dir),
        "eval_root": str(args.eval_root),
        "run_label": args.run_label,
        "seed": args.seed,
        "failure_cer_threshold": float(args.failure_cer_threshold),
        "failure_flat_tolerance": float(args.failure_flat_tolerance),
        "train_log": str(train_log),
        "inference_config": dict(DEFAULT_QUICK20_INFERENCE_CONFIG),
        "rows": rows,
        "stop_continue_decision": stop_continue_decision(rows, float(args.failure_flat_tolerance)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(
        args.output_md,
        rows,
        float(args.failure_cer_threshold),
        float(args.failure_flat_tolerance),
    )
    print(f"[recipe-final-per-save-summary] wrote {args.output_json}")
    print(f"[recipe-final-per-save-summary] wrote {args.output_md}")


if __name__ == "__main__":
    main()
