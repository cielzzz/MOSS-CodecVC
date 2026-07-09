#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PER_SAVE_JSON = ROOT / "docs/assets/ver2_8_recipe_final_per_save_quick20_20260705.json"
DEFAULT_RUN_DIR = ROOT / "outputs/lora_runs/ver2_8_timbre_repair_recipe_final_varlen_block_permuted_cref_prompt_a4_refsup_cosramp_infonce_dropout_steps30000"
DEFAULT_OUTPUT_ROOT = ROOT / "testset/outputs/ver2_8_recipe_final_eval_grid_best2_20260706"
DEFAULT_DOCS_DIR = ROOT / "docs"
DEFAULT_ASSETS_DIR = ROOT / "docs/assets"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Select the best recipe_final checkpoints from per-save quick20 and "
            "run the final 320/badcase eval grid for each selected checkpoint."
        )
    )
    ap.add_argument("--per-save-json", type=Path, default=DEFAULT_PER_SAVE_JSON)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--require-steps", default="26000,28000,30000", help="Comma-separated steps that must be present before executing.")
    ap.add_argument(
        "--candidate-steps",
        default="",
        help="Comma-separated steps eligible for best-k selection. Defaults to --require-steps when set.",
    )
    ap.add_argument(
        "--selection-mode",
        choices=("sim_ref", "balanced"),
        default="sim_ref",
        help="sim_ref selects by sim(gen,ref) desc; balanced keeps the older fail/keep/sim/CER ranking.",
    )
    ap.add_argument("--windows", default="4,8")
    ap.add_argument("--cfg-scales", default="")
    ap.add_argument("--best-cfg-scale", default="1.0")
    ap.add_argument("--badcase-windows", default="8")
    ap.add_argument("--badcase-cfg-scales", default="1.0")
    ap.add_argument("--badcase-seeds", default="1234,2025,3407")
    ap.add_argument("--gpu-count", default="")
    ap.add_argument("--num-shards", default="")
    ap.add_argument("--asr-num-shards", default="")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    ap.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    ap.add_argument("--selection-json", type=Path, default=DEFAULT_ASSETS_DIR / "ver2_8_recipe_final_best2_eval_grid_selection_20260706.json")
    ap.add_argument("--selection-md", type=Path, default=DEFAULT_DOCS_DIR / "ver2_8_recipe_final_best2_eval_grid_selection_20260706.md")
    ap.add_argument("--execute", action="store_true", help="Actually run scripts/004059 for selected checkpoints.")
    ap.add_argument("--dry-run-grid", action="store_true", help="Pass DRY_RUN=1 to scripts/004059 when --execute is set.")
    return ap.parse_args()


def finite(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if out == out else default


def unique_csv(values: list[str]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for item in str(raw or "").split(","):
            item = item.strip()
            if item and item not in seen:
                seen.add(item)
                out.append(item)
    return ",".join(out)


def completed_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("rows") or []:
        if not row.get("quick20_complete"):
            continue
        if not row.get("checkpoint_exists"):
            continue
        if row.get("sim_gen_ref_mean") is None:
            continue
        rows.append(row)
    return rows


def balanced_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    fail = finite(row.get("failure_rate_cer_gt_threshold"), 1e9)
    keep = finite(row.get("keep"), -1e9)
    sim_ref = finite(row.get("sim_gen_ref_mean"), -1e9)
    cer = finite(row.get("cer"), 1e9)
    ref_f1 = finite(row.get("ref_content_lcs_f1_mean"), 1e9)
    step = int(row.get("step") or 0)
    return (fail, -keep, -sim_ref, cer, ref_f1, -step)


def sim_ref_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    sim_ref = finite(row.get("sim_gen_ref_mean"), -1e9)
    fail = finite(row.get("failure_rate_cer_gt_threshold"), 1e9)
    keep = finite(row.get("keep"), -1e9)
    cer = finite(row.get("cer"), 1e9)
    ref_f1 = finite(row.get("ref_content_lcs_f1_mean"), 1e9)
    step = int(row.get("step") or 0)
    return (-sim_ref, fail, -keep, cer, ref_f1, -step)


def filter_candidate_steps(rows: list[dict[str, Any]], candidate_steps: str, require_steps: str) -> list[dict[str, Any]]:
    raw = candidate_steps.strip() or require_steps.strip()
    if not raw:
        return rows
    wanted = {int(item.strip()) for item in raw.split(",") if item.strip()}
    return [row for row in rows if int(row.get("step") or -1) in wanted]


def select_rows(rows: list[dict[str, Any]], top_k: int, selection_mode: str) -> list[dict[str, Any]]:
    key = sim_ref_rank_key if selection_mode == "sim_ref" else balanced_rank_key
    return sorted(rows, key=key)[: max(1, int(top_k))]


def require_steps_present(rows: list[dict[str, Any]], require_steps: str) -> list[int]:
    if not require_steps.strip():
        return []
    completed = {int(row.get("step") or -1) for row in rows}
    required = [int(item.strip()) for item in require_steps.split(",") if item.strip()]
    return [step for step in required if step not in completed]


def selection_rule_text(selection_mode: str) -> str:
    if selection_mode == "sim_ref":
        return "candidate steps only; sort by sim(gen,ref) desc, then fail asc, keep desc, CER asc, ref-content F1 asc, later step desc"
    return "sort by fail asc, keep desc, sim(gen,ref) desc, CER asc, ref-content F1 asc, later step desc"


def write_selection(args: argparse.Namespace, payload: dict[str, Any], rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], selected: list[dict[str, Any]], missing_required: list[int], cfg_scales: str) -> None:
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    args.selection_md.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "per_save_json": str(args.per_save_json),
        "run_dir": str(args.run_dir),
        "selection_mode": args.selection_mode,
        "selection_rule": selection_rule_text(args.selection_mode),
        "top_k": int(args.top_k),
        "candidate_steps": args.candidate_steps.strip() or args.require_steps.strip(),
        "require_steps_missing": missing_required,
        "windows": args.windows,
        "cfg_scales": cfg_scales,
        "badcase_windows": args.badcase_windows,
        "badcase_cfg_scales": args.badcase_cfg_scales,
        "badcase_seeds": args.badcase_seeds,
        "candidate_steps_completed": [row.get("step") for row in candidate_rows],
        "selected_steps": [row.get("step") for row in selected],
        "selected": selected,
    }
    args.selection_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Recipe Final Best-2 Eval Grid Selection",
        "",
        f"per-save JSON: `{args.per_save_json}`",
        f"run dir: `{args.run_dir}`",
        "",
        f"Selection rule: {selection_rule_text(args.selection_mode)}.",
        f"Candidate steps: `{args.candidate_steps.strip() or args.require_steps.strip() or 'all completed'}`",
        f"Completed candidate steps: `{[row.get('step') for row in candidate_rows]}`",
        "",
    ]
    if missing_required:
        lines.append(f"Missing required completed steps: `{missing_required}`")
        lines.append("")
    lines.extend(
        [
            f"windows: `{args.windows}`",
            f"cfg scales: `{cfg_scales}`",
            f"badcase windows: `{args.badcase_windows}`",
            f"badcase cfg scales: `{args.badcase_cfg_scales}`",
            f"badcase seeds: `{args.badcase_seeds}`",
            "",
            "| rank | step | CER | fail | keep | sim gen-ref | sim gen-src | ref-content F1 | output dir |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for idx, row in enumerate(selected, start=1):
        lines.append(
            "| {rank} | {step} | {cer:.4f} | {fail:.4f} | {keep} | {sim_ref:.4f} | {sim_src:.4f} | {f1:.4f} | `{out}` |".format(
                rank=idx,
                step=int(row.get("step") or 0),
                cer=finite(row.get("cer"), 0.0),
                fail=finite(row.get("failure_rate_cer_gt_threshold"), 0.0),
                keep=int(row.get("keep") or 0),
                sim_ref=finite(row.get("sim_gen_ref_mean"), 0.0),
                sim_src=finite(row.get("sim_gen_source_mean"), 0.0),
                f1=finite(row.get("ref_content_lcs_f1_mean"), 0.0),
                out=row.get("output_dir") or "",
            )
        )
    lines.extend(["", f"selection JSON: `{args.selection_json}`", ""])
    args.selection_md.write_text("\n".join(lines), encoding="utf-8")


def run_grid(args: argparse.Namespace, selected: list[dict[str, Any]], cfg_scales: str) -> None:
    script = ROOT / "scripts/004059_run_recipe_final_eval_grid.sh"
    for row in selected:
        step = int(row.get("step") or 0)
        checkpoint = args.run_dir / f"step-{step}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"selected checkpoint not found: {checkpoint}")
        output_root = args.output_root / f"step-{step}"
        docs_md = args.docs_dir / f"ver2_8_recipe_final_eval_grid_step{step}_20260706.md"
        asset_prefix = args.assets_dir / f"ver2_8_recipe_final_eval_grid_step{step}_20260706"
        env = os.environ.copy()
        env.update(
            {
                "CHECKPOINT": str(checkpoint),
                "OUTPUT_ROOT": str(output_root),
                "DOCS_MD": str(docs_md),
                "ASSET_PREFIX": str(asset_prefix),
                "WINDOWS": args.windows,
                "CFG_SCALES": cfg_scales,
                "BADCASE_WINDOWS": args.badcase_windows,
                "BADCASE_CFG_SCALES": args.badcase_cfg_scales,
                "BADCASE_SEEDS": args.badcase_seeds,
                "DRY_RUN": "1" if args.dry_run_grid else "0",
            }
        )
        for key, value in (
            ("GPU_COUNT", args.gpu_count),
            ("NUM_SHARDS", args.num_shards),
            ("ASR_NUM_SHARDS", args.asr_num_shards),
        ):
            if value:
                env[key] = value
        print(f"[best2-grid] step={step} checkpoint={checkpoint} output_root={output_root}")
        subprocess.run(["bash", str(script)], cwd=str(ROOT), env=env, check=True)


def main() -> int:
    args = parse_args()
    payload = json.loads(args.per_save_json.read_text(encoding="utf-8"))
    rows = completed_rows(payload)
    missing_required = require_steps_present(rows, args.require_steps)
    candidate_rows = filter_candidate_steps(rows, args.candidate_steps, args.require_steps)
    selected = select_rows(candidate_rows, args.top_k, args.selection_mode)
    cfg_scales = unique_csv(["1.0", args.best_cfg_scale, args.cfg_scales])
    write_selection(args, payload, rows, candidate_rows, selected, missing_required, cfg_scales)
    print(f"[best2-grid] selected steps: {', '.join(str(row.get('step')) for row in selected)}")
    print(f"[best2-grid] wrote {args.selection_md}")
    if missing_required:
        print(f"[best2-grid] required steps missing; skip execute: {missing_required}")
        return 0
    if args.execute:
        run_grid(args, selected, cfg_scales)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
