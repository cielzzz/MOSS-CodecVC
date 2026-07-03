#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl, stable_id, write_jsonl


def compatible(a: dict[str, Any], b: dict[str, Any], *, same_language: bool) -> bool:
    if not same_language:
        return True
    la = a.get("language")
    lb = b.get("language")
    return not la or not lb or la == lb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True, help="VC manifest or encoded manifest")
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--output-audio-root", required=True, help="Where generated counterfactual source wavs should be written")
    ap.add_argument("--views-per-source", type=int, default=1)
    ap.add_argument("--offset", type=int, default=17)
    ap.add_argument("--same-language", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    rows = list(iter_jsonl(args.input_jsonl))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    usable = [r for r in rows if r.get("source_audio") and r.get("timbre_ref_audio") and r.get("target_audio")]
    out = []
    n = len(usable)
    if n == 0:
        write_jsonl(args.output_jsonl, [])
        print(f"wrote 0 rows -> {Path(args.output_jsonl).resolve()}")
        return 0

    root = Path(args.output_audio_root)
    for i, row in enumerate(usable):
        chosen = 0
        probe = 0
        while chosen < args.views_per_source and probe < n * 2:
            j = (i + args.offset + probe) % n
            probe += 1
            if j == i:
                continue
            donor = usable[j]
            if not compatible(row, donor, same_language=args.same_language):
                continue
            group_id = row.get("counterfactual_group_id") or stable_id(
                row.get("source_audio"),
                row.get("timbre_ref_audio"),
                row.get("target_audio"),
                length=20,
            )
            cf_id = f"{row.get('sample_id') or stable_id(row.get('source_audio'), i)}:cf{chosen:02d}"
            rel = f"{group_id}/{cf_id}.wav"
            out.append(
                {
                    "counterfactual_id": cf_id,
                    "counterfactual_group_id": group_id,
                    "base_sample_id": row.get("sample_id"),
                    "language": row.get("language"),
                    "source_audio": row.get("source_audio"),
                    "source_text": row.get("source_text") or row.get("target_text"),
                    "source_timbre_donor_audio": donor.get("timbre_ref_audio") or donor.get("source_audio"),
                    "source_timbre_donor_text": donor.get("timbre_ref_text") or donor.get("source_text"),
                    "expected_counterfactual_source_audio": str(root / rel),
                    "target_audio": row.get("target_audio"),
                    "target_text": row.get("target_text") or row.get("source_text"),
                    "timbre_ref_audio": row.get("timbre_ref_audio"),
                    "timbre_ref_text": row.get("timbre_ref_text"),
                    "instruction": (
                        "Generate a counterfactual source view: keep source content, duration, pauses and prosody, "
                        "but replace only the source speaker timbre with the donor timbre. "
                        "This audio will be used as an alternative S1 condition for invariance training."
                    ),
                    "meta": {
                        "builder": "001005_build_counterfactual_source_plan",
                        "donor_sample_id": donor.get("sample_id"),
                        "view_index": chosen,
                    },
                }
            )
            chosen += 1
    n_out = write_jsonl(args.output_jsonl, out)
    print(f"wrote {n_out} counterfactual plan rows -> {Path(args.output_jsonl).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
