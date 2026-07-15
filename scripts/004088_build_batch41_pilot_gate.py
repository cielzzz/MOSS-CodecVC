#!/usr/bin/env python3
"""Build the machine-readable Batch-41 repeat=3 pilot promotion gate.

The gate is intentionally derived from the same final320 run summary and
dual-encoder per-case CSV used for the Batch-33/B2 comparison.  In particular,
``text_en_src_fail`` uses the official ``content_keep`` definition over the 80
text-mode English-source cases; it is not the weaker ``CER > threshold`` rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "moss_codecvc.batch41_pilot_gate.v1"


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {value!r}")
    return result


def bool_value(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "keep"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "drop", "filtered"}:
        return False
    raise ValueError(f"invalid boolean metric: {value!r}")


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty metric list")
    return sum(materialized) / len(materialized)


def read_dual_rows(path: Path, run_id: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "run",
            "case_id",
            "mode",
            "cell",
            "cer_tgt",
            "content_keep",
            "sim_gen_ref",
            "sim_gen_source",
            "ecapa_sim_gen_ref",
        }
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(f"dual-encoder CSV is missing fields: {sorted(missing_fields)}")
        rows = [row for row in reader if row.get("run") == run_id]
    case_ids = [str(row.get("case_id") or "") for row in rows]
    if (
        len(rows) != 320
        or len(set(case_ids)) != 320
        or any(not case_id for case_id in case_ids)
    ):
        raise ValueError(
            f"expected 320 unique dual-encoder rows for {run_id}, got {len(rows)}"
        )
    return rows


def build_gate(
    *,
    summary_path: Path,
    cases_path: Path,
    pilot_job_id: str,
    run_id: str,
    checkpoint_step: int = 3000,
    text_repeat: int = 3,
) -> dict[str, Any]:
    if checkpoint_step != 3000 or text_repeat != 3:
        raise ValueError("Batch-41 gate is registered only for step=3000, repeat=3")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("run_id") or "") != run_id:
        raise ValueError(
            f"run summary provenance mismatch: {summary.get('run_id')!r} != {run_id!r}"
        )
    rows = read_dual_rows(cases_path, run_id)

    no_text_rows = [row for row in rows if row.get("mode") == "no_text"]
    text_rows = [row for row in rows if row.get("mode") == "text"]
    text_en_src_rows = [
        row
        for row in text_rows
        if str(row.get("cell") or "").startswith("en_src_")
    ]
    if len(no_text_rows) != 160 or len(text_rows) != 160 or len(text_en_src_rows) != 80:
        raise ValueError(
            "unexpected scopes: "
            f"no_text={len(no_text_rows)} text={len(text_rows)} "
            f"text_en_src={len(text_en_src_rows)}"
        )

    no_text_summary = summary["by_mode"]["no_text"]
    text_summary = summary["by_mode"]["text"]
    if int(no_text_summary["n"]) != 160 or int(text_summary["n"]) != 160:
        raise ValueError("run summary does not contain complete 160+160 modes")

    no_text_cer = mean(finite(row["cer_tgt"]) for row in no_text_rows)
    text_cer = mean(finite(row["cer_tgt"]) for row in text_rows)
    summary_no_text_cer = finite(no_text_summary["cer"])
    summary_text_cer = finite(text_summary["cer"])
    if not math.isclose(no_text_cer, summary_no_text_cer, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "no_text CER disagrees between dual-encoder CSV and run summary: "
            f"{no_text_cer} != {summary_no_text_cer}"
        )
    if not math.isclose(text_cer, summary_text_cer, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "text CER disagrees between dual-encoder CSV and run summary: "
            f"{text_cer} != {summary_text_cer}"
        )

    no_text_keep = sum(bool_value(row.get("content_keep")) for row in no_text_rows)
    text_keep = sum(bool_value(row.get("content_keep")) for row in text_rows)
    if no_text_keep != int(no_text_summary["keep"]):
        raise ValueError("no_text content_keep count disagrees with run summary")
    if text_keep != int(text_summary["keep"]):
        raise ValueError("text content_keep count disagrees with run summary")

    text_en_src_keep = sum(
        bool_value(row.get("content_keep")) for row in text_en_src_rows
    )
    metrics = {
        "no_text_cer": no_text_cer,
        "no_text_fail": (160 - no_text_keep) / 160,
        "text_cer": text_cer,
        "text_fail": (160 - text_keep) / 160,
        "text_en_src_fail": (80 - text_en_src_keep) / 80,
        "wavlm_sim_ref": mean(finite(row["sim_gen_ref"]) for row in no_text_rows),
        "wavlm_sim_src": mean(finite(row["sim_gen_source"]) for row in no_text_rows),
        "wavlm_ref_bound": sum(
            finite(row["sim_gen_ref"]) - finite(row["sim_gen_source"]) > 0.05
            for row in no_text_rows
        )
        / 160,
        "speechbrain_ecapa_sim_ref": mean(
            finite(row["ecapa_sim_gen_ref"]) for row in no_text_rows
        ),
    }
    checks = {
        "no_text_cer_lt_0p12": metrics["no_text_cer"] < 0.12,
        "text_en_src_fail_lt_0p15": metrics["text_en_src_fail"] < 0.15,
        "text_cer_lt_0p06": metrics["text_cer"] < 0.06,
        "wavlm_sim_ref_ge_0p42": metrics["wavlm_sim_ref"] >= 0.42,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "pass" if all(checks.values()) else "fail",
        "pilot_job_id": pilot_job_id,
        "checkpoint_step": checkpoint_step,
        "text_repeat": text_repeat,
        **metrics,
        "checks": checks,
        "run_id": run_id,
        "run_summary": str(summary_path.resolve()),
        "dual_encoder_cases": str(cases_path.resolve()),
        "metric_scope": {
            "no_text_cer": "no_text-160",
            "text_cer": "text-160",
            "text_en_src_fail": (
                "text en_src cells, n=80, official content_keep definition"
            ),
            "wavlm_sim_ref": "no_text-160 SeedTTS/WavLM speaker encoder",
        },
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--dual-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-job-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=3000)
    parser.add_argument("--text-repeat", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_gate(
        summary_path=args.run_summary,
        cases_path=args.dual_cases,
        pilot_job_id=args.pilot_job_id,
        run_id=args.run_id,
        checkpoint_step=args.checkpoint_step,
        text_repeat=args.text_repeat,
    )
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[batch41-pilot-gate] decision={payload['decision']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
