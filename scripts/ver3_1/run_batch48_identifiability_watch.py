#!/usr/bin/env python3
"""Batch-48 raw identifiability watcher (local RTX4090).

The watcher is marker-driven and never submits training.  It evaluates both
raw and EMA checkpoint lanes every 500 steps.  At step 1500 it applies the
pre-registered raw thresholds and writes a stop marker; an optional qzcli stop
request is attempted, but an authentication failure is recorded rather than
silently treated as a successful stop.  If step 1500 passes, step 3000 gets a
final identifiability decision.  Audio quick20/full320 inference is kept as a
separate local-only contract so the watcher cannot accidentally launch six
expensive audio evaluations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STEPS = (500, 1000, 1500, 2000, 2500, 3000)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--qz-job-id", default="")
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--device-raw", default="cuda:0")
    # The evaluation host is normally a single RTX4090.  An empty EMA device
    # means "reuse --device-raw"; users with a second GPU may override it.
    ap.add_argument("--device-ema", default="")
    ap.add_argument("--max-scans", type=int, default=0)
    ap.add_argument("--no-qz-stop", action="store_true")
    return ap.parse_args()


def write_local_eval_contract(output_root: Path, checkpoint_root: Path) -> Path:
    """Write the Batch-48 audio-evaluation contract without running it.

    Audio inference is intentionally not launched from this watcher: it can
    take substantially longer than the endpoint diagnostic and must remain on
    the local RTX4090.  The contract records the exact inputs, checkpoints,
    case counts and commands for the separate local evaluator.
    """

    quick20_jsonl = ROOT / "testset/outputs/ver3_1_step4_ddlfm_eval_20260715/conditions/validation_quick20_no_text.jsonl"
    full320_jsonl = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
    quick20_runner = ROOT / "scripts/ver3_1/run_batch47_local_quick20.py"
    evaluator = ROOT / "scripts/ver3_1/evaluate_ddlfm_validation.py"
    adapter = ROOT / "outputs/ver3_1_content_adapter_probe_20260715/step-003000"
    contract = {
        "schema": "ver3_1_batch48_local_audio_eval_contract_v1",
        "status": "contract_only_not_started",
        "backend": "local RTX4090 only",
        "never_submits_qz": True,
        "checkpoint_root": str(checkpoint_root),
        "quick20": {
            "steps": [1500, 3000],
            "mode": "no_text",
            "rows": 20,
            "validation_jsonl": str(quick20_jsonl),
            "runner": str(quick20_runner),
            "output_root": str(output_root / "quick20"),
            "note": "Run only after the corresponding ready marker; this command evaluates only steps 1500 and 3000.",
        },
        "full320_no_text": {
            "step": 3000,
            "mode": "no_text",
            "rows": 160,
            "validation_jsonl": str(full320_jsonl),
            "max_cases": 160,
            "evaluator": str(evaluator),
            "adapter_checkpoint": str(adapter),
            "output_root": str(output_root / "full320_no_text"),
            "note": "Run locally after step-3000 identifiability decision; no text rows are included.",
        },
        "commands": {
            "quick20": [
                "python",
                str(quick20_runner),
                "--action",
                "once",
                "--allow-run",
                "--checkpoint-root",
                str(checkpoint_root),
                "--output-root",
                str(output_root / "quick20"),
                "--record-root",
                str(output_root / "quick20_records"),
                "--task-name",
                "codecVC-ver3-1-batch48-local-quick20-20260717",
                "--steps",
                "1500,3000",
                "--inference-device",
                "cuda:0",
                "--condition-device",
                "cuda:0",
                "--asr-device",
                "cuda:0",
                "--speaker-device",
                "cuda:0",
            ],
            "full320_no_text": [
                "python",
                str(evaluator),
                "--validation-jsonl",
                str(full320_jsonl),
                "--mode",
                "no_text",
                "--max-cases",
                "160",
                "--adapter-checkpoint",
                str(adapter),
                "--sampling-steps",
                "20",
                "--speaker-cfg-scale",
                "2.5",
                "--semantic-cfg-scale",
                "2.0",
                "--zq-channel-stats",
                str(ROOT / "prepared/zq_targets_v1/channel_stats.pt"),
                "--device",
                "cuda:0",
                "--ecapa-device",
                "cuda:0",
                "--output-dir",
                str(output_root / "full320_no_text"),
                "--overwrite",
            ],
        },
    }
    path = output_root / "LOCAL_AUDIO_EVAL_CONTRACT.json"
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_loss_snapshot(checkpoint_root: Path, step: int) -> dict[str, object]:
    """Read the bounded training-loss history available at a checkpoint."""

    path = checkpoint_root / "train_log.jsonl"
    if not path.is_file():
        return {"status": "missing", "path": str(path), "step": int(step)}
    rows: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            row_step = int(payload.get("step", -1))
            value = float(payload.get("loss"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if row_step <= int(step) and row_step >= 0:
            rows.append((row_step, value))
    if not rows:
        return {"status": "no_points", "path": str(path), "step": int(step)}
    rows.sort()
    values = [value for _, value in rows]
    window = values[-min(5, len(values)):]
    slope = None
    if len(window) >= 2:
        slope = (window[-1] - window[0]) / float(len(window) - 1)
    return {
        "status": "completed",
        "path": str(path),
        "step": int(step),
        "points": len(rows),
        "first": values[0],
        "last": values[-1],
        "last_window_slope": slope,
        "direction": "down" if slope is not None and slope < -1.0e-8 else (
            "up" if slope is not None and slope > 1.0e-8 else "flat"
        ),
    }


def run_lane(checkpoint: Path, weight: str, output: Path, device: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python",
        str(ROOT / "scripts/ver3_1/run_batch47_endpoint_gate.py"),
        "--checkpoint", str(checkpoint),
        "--weights", weight,
        "--output-dir", str(output),
        "--device", device,
        "--speaker-cfg-scale", "2.5",
        "--semantic-cfg-scale", "2.0",
        "--zq-channel-stats", str(ROOT / "prepared/zq_targets_v1/channel_stats.pt"),
        "--overwrite",
    ]
    log = output / "watch.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"identifiability {weight} failed; see {log}")
    report = output / "checkpoint_identifiability.json"
    return json.loads(report.read_text(encoding="utf-8"))


def attempt_qz_stop(job_id: str) -> dict:
    if not job_id:
        return {"attempted": False, "reason": "no job id"}
    wrapper = "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh"
    env = dict(os.environ)
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    result = subprocess.run(
        [wrapper, "stop", "--yes", job_id],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "attempted": True,
        "returncode": int(result.returncode),
        "output": result.stdout[-2000:],
        "success": result.returncode == 0,
    }


def main() -> int:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = write_local_eval_contract(output_root, checkpoint_root)
    ema_device = args.device_ema or args.device_raw
    processed: set[int] = set()
    scans = 0
    while True:
        scans += 1
        for step in STEPS:
            if step in processed:
                continue
            marker = checkpoint_root / f"step-{step:06d}.ready.json"
            checkpoint = checkpoint_root / f"step-{step:06d}.infer.pt"
            if not marker.is_file() or not checkpoint.is_file():
                continue
            step_root = output_root / f"step-{step:06d}"
            step_root.mkdir(parents=True, exist_ok=True)
            reports = {}
            reports["raw"] = run_lane(checkpoint, "raw", step_root / "raw", args.device_raw)
            reports["ema"] = run_lane(checkpoint, "ema", step_root / "ema", ema_device)
            aggregate = {
                "schema": "ver3_1_batch48_identifiability_step_v1",
                "status": "completed",
                "step": step,
                "checkpoint": str(checkpoint),
                "raw": reports["raw"].get("metrics", {}),
                "ema": reports["ema"].get("metrics", {}),
                "loss": read_loss_snapshot(checkpoint_root, step),
            }
            (step_root / "identifiability.json").write_text(
                json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            processed.add(step)
            raw = aggregate["raw"]
            if step == 1500:
                gates = {
                    "raw_t0_semantic_gt_15pct": float(raw.get("raw_matched_t0_advantage", -1.0)) > 0.15,
                    "raw_speaker_gt_12pct": float(raw.get("raw_speaker_cond_vs_zero_relative_to_target_rms", -1.0)) > 0.12,
                    "raw_free_ode_gt_0_20": float(raw.get("raw_matched_ode_cosine", -1.0)) > 0.20,
                }
                stop_payload = {
                    "schema": "ver3_1_batch48_step1500_stop_v1",
                    "status": "pass" if all(gates.values()) else "stop_required",
                    "step": 1500,
                    "gates": gates,
                    "raw_metrics": raw,
                    "loss": aggregate["loss"],
                }
                if not all(gates.values()) and not args.no_qz_stop:
                    stop_payload["qz_stop"] = attempt_qz_stop(args.qz_job_id)
                (output_root / "STEP1500_DECISION.json").write_text(
                    json.dumps(stop_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                if not all(gates.values()):
                    return 0
            if step == 3000:
                gates = {
                    "raw_t0_semantic_gt_20pct": float(raw.get("raw_matched_t0_advantage", -1.0)) > 0.20,
                    "raw_speaker_gt_15pct": float(raw.get("raw_speaker_cond_vs_zero_relative_to_target_rms", -1.0)) > 0.15,
                    "raw_free_ode_gt_0_30": float(raw.get("raw_matched_ode_cosine", -1.0)) > 0.30,
                }
                final_payload = {
                    "schema": "ver3_1_batch48_step3000_decision_v1",
                    "status": "passed" if all(gates.values()) else "failed",
                    "step": 3000,
                    "gates": gates,
                    "raw_metrics": raw,
                    "ema_metrics": aggregate["ema"],
                    "loss": aggregate["loss"],
                    "audio_eval_contract": str(contract_path),
                    "audio_eval_status": "pending_local_quick20_and_full320_no_text",
                }
                (output_root / "STEP3000_DECISION.json").write_text(
                    json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                return 0
        available_steps = [
            step for step in STEPS
            if (checkpoint_root / f"step-{step:06d}.ready.json").is_file()
        ]
        if (checkpoint_root / "COMPLETED.json").is_file() and all(
            step in processed for step in available_steps
        ):
            return 0
        if args.max_scans and scans >= args.max_scans:
            return 0
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
