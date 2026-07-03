#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-jsonl", required=True, help="Output from 001005_build_counterfactual_source_plan.py")
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--require-audio-exists", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    out: list[dict[str, Any]] = []
    missing = 0
    for row in iter_jsonl(args.plan_jsonl):
        cf_audio = row.get("expected_counterfactual_source_audio")
        if not cf_audio:
            missing += 1
            continue
        if args.require_audio_exists and not Path(cf_audio).exists():
            missing += 1
            continue
        out.append(
            {
                "sample_id": row.get("counterfactual_id"),
                "pair_type": "MOSSCodecVCCounterfactualSource",
                "language": row.get("language"),
                "source_audio": cf_audio,
                "source_text": row.get("source_text"),
                "timbre_ref_audio": row.get("timbre_ref_audio"),
                "timbre_ref_text": row.get("timbre_ref_text"),
                "target_audio": row.get("target_audio"),
                "target_text": row.get("target_text") or row.get("source_text"),
                "instruction": (
                    "Voice conversion task with a counterfactual source view. [S1] is a timbre-shifted "
                    "source speech that should keep content, timing and prosody; [S2] is the target timbre "
                    "reference. Generate the same target as other views in the same counterfactual group."
                ),
                "counterfactual_group_id": row.get("counterfactual_group_id"),
                "condition_view": "counterfactual_source",
                "base_sample_id": row.get("base_sample_id"),
                "source_timbre_donor_audio": row.get("source_timbre_donor_audio"),
                "meta": {
                    "builder": "001006_materialize_counterfactual_manifest",
                    "plan_counterfactual_id": row.get("counterfactual_id"),
                    "source_audio_before_shift": row.get("source_audio"),
                },
            }
        )
    n = write_jsonl(args.output_jsonl, out)
    print(f"wrote {n} rows -> {Path(args.output_jsonl).resolve()} missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
