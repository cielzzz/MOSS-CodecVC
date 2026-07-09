#!/usr/bin/env python
"""Summarize stop/duration diagnostics from v2 full A/B/C quick eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def load_run(run_dir: Path, run_id: str, case_key: str, top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifests = []
    for manifest_path in sorted(run_dir.glob("manifest.shard*.jsonl")):
        manifests.extend(read_jsonl(manifest_path))
    manifest_by_id = {str(r.get("case_id") or r.get("sample_id")): r for r in manifests}
    asr_path = run_dir / f"{run_id}.asr_eval.jsonl"
    asr_rows = read_jsonl(asr_path)

    diag_rows: list[dict[str, Any]] = []
    for asr in asr_rows:
        case_id = str(asr.get("case_id") or asr.get("sample_id"))
        manifest = manifest_by_id.get(case_id, {})
        gen = manifest.get("generation_structure") or {}
        stop = manifest.get("progress_stop_infer_stats") or {}
        perm = manifest.get("ref_prompt_codec_permutation") or {}
        dedelayed = gen.get("dedelayed_segment_lengths") or []
        ref_text = asr.get("content_ref_text") or manifest.get("content_ref_text") or ""
        hyp = asr.get("asr_tgt_text") or ""
        diag_rows.append(
            {
                "case": case_key,
                "run_id": run_id,
                "case_id": case_id,
                "cell": asr.get("cell") or manifest.get("cell"),
                "language": asr.get("language") or manifest.get("language"),
                "content_keep": asr.get("content_keep"),
                "filter_reason": asr.get("content_filter_reason"),
                "cer": finite(asr.get("cer_tgt")),
                "wer": finite(asr.get("wer_tgt")),
                "repeat_score": finite(asr.get("repeat_score")),
                "duration_ratio_tgt_src": finite(asr.get("duration_ratio_tgt_src")),
                "source_frames": finite(perm.get("source_frames")),
                "prompt_frames": finite(perm.get("prompt_frames")),
                "gen_slot_count": finite(gen.get("gen_slot_count")),
                "delay_slot_count": finite(gen.get("delay_slot_count")),
                "dedelayed_first_len": finite(dedelayed[0] if dedelayed else None),
                "rows": finite(gen.get("rows")),
                "first_delay_pos": finite(gen.get("first_delay_pos")),
                "stop_steps": finite(stop.get("steps")),
                "forced_rows": finite(stop.get("forced_rows")),
                "stop_prob_max": finite(stop.get("stop_prob_max_max")),
                "stop_prob_mean": finite(stop.get("stop_prob_max_mean")),
                "progress_max": finite(stop.get("progress_value_max_max")),
                "hyp_ref_len_ratio": (len(hyp) / len(ref_text)) if ref_text else None,
                "ref_text": ref_text,
                "asr_hyp": hyp,
                "source_audio": manifest.get("source_audio") or asr.get("source_audio"),
                "timbre_ref_audio": manifest.get("timbre_ref_audio") or asr.get("timbre_ref_audio"),
                "target_audio": manifest.get("target_audio"),
                "generated_audio": manifest.get("output_wav") or asr.get("target_audio"),
            }
        )

    summary = {
        "case": case_key,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "n": len(diag_rows),
        "keep": sum(str(r.get("content_keep")) == "True" for r in diag_rows),
        "empty_hyp": sum(not str(r.get("asr_hyp") or "").strip() for r in diag_rows),
        "cer_mean": mean([r["cer"] for r in diag_rows]),
        "duration_ratio_mean": mean([r["duration_ratio_tgt_src"] for r in diag_rows]),
        "hyp_ref_len_ratio_mean": mean([r["hyp_ref_len_ratio"] for r in diag_rows]),
        "source_frames_mean": mean([r["source_frames"] for r in diag_rows]),
        "prompt_frames_mean": mean([r["prompt_frames"] for r in diag_rows]),
        "gen_slot_count_mean": mean([r["gen_slot_count"] for r in diag_rows]),
        "dedelayed_first_len_mean": mean([r["dedelayed_first_len"] for r in diag_rows]),
        "stop_prob_max_mean": mean([r["stop_prob_max"] for r in diag_rows]),
        "progress_max_mean": mean([r["progress_max"] for r in diag_rows]),
    }

    failures = [r for r in diag_rows if str(r.get("content_keep")) != "True"]
    failures.sort(key=lambda r: (r["cer"] is None, -(r["cer"] or -1.0), -(r["repeat_score"] or -1.0)))
    return summary, failures[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--run-label", default="v1_v2full_cross_attn_lite")
    parser.add_argument("--step", required=True, help="500 or step-500")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    step = str(args.step)
    if not step.startswith("step-"):
        step = f"step-{step}"
    eval_root = args.eval_root

    summaries = []
    failures = []
    for case_key in ("A", "B", "C"):
        prefix = f"{args.run_label}_case{case_key}_{step}_quick20_d2d3_seed"
        dirs = sorted(p for p in eval_root.iterdir() if p.is_dir() and p.name.startswith(prefix))
        if not dirs:
            summaries.append({"case": case_key, "missing": True, "prefix": prefix})
            continue
        run_dir = dirs[-1]
        summary, rows = load_run(run_dir, run_dir.name, case_key, int(args.top_k))
        summaries.append(summary)
        failures.extend(rows)

    payload = {"eval_root": str(eval_root), "step": step, "summaries": summaries, "top_failures": failures}
    output_json = args.output_json or eval_root / f"{args.run_label}_{step}_stop_head_diag.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    output_csv = args.output_csv or eval_root / f"{args.run_label}_{step}_stop_head_diag_failures.csv"
    if failures:
        fieldnames = [
            "case",
            "case_id",
            "cell",
            "language",
            "content_keep",
            "filter_reason",
            "cer",
            "repeat_score",
            "duration_ratio_tgt_src",
            "hyp_ref_len_ratio",
            "source_frames",
            "prompt_frames",
            "gen_slot_count",
            "dedelayed_first_len",
            "stop_steps",
            "stop_prob_max",
            "progress_max",
            "ref_text",
            "asr_hyp",
            "generated_audio",
            "source_audio",
            "timbre_ref_audio",
            "target_audio",
        ]
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in failures:
                writer.writerow({key: row.get(key) for key in fieldnames})
    else:
        output_csv.write_text("", encoding="utf-8")

    print(f"[v2-stop-diag] wrote {output_json}")
    print(f"[v2-stop-diag] wrote {output_csv}")
    for summary in summaries:
        print(
            f"case{summary.get('case')}: n={summary.get('n')} keep={summary.get('keep')} "
            f"cer={summary.get('cer_mean')} hyp/ref={summary.get('hyp_ref_len_ratio_mean')} "
            f"dur={summary.get('duration_ratio_mean')} stop_max={summary.get('stop_prob_max_mean')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
