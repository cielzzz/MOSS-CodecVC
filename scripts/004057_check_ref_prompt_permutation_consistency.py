#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.ref_prompt_permutation import permute_ref_prompt_codes


def row_counter(tensor: torch.Tensor) -> Counter[tuple[int, ...]]:
    if tensor.dim() != 2:
        raise ValueError(f"row_counter expects [T, C], got {tuple(tensor.shape)}")
    return Counter(tuple(int(v) for v in row.tolist()) for row in tensor.cpu())


def frame_row_integrity_payload(
    original_codes: torch.Tensor,
    prompt_codes: torch.Tensor,
    stats: dict[str, int],
    *,
    expected_codebooks: int = 32,
) -> dict[str, Any]:
    start = int(stats["start"])
    take = int(stats["prompt_frames"])
    source_frames = int(stats["source_frames"])
    selected = original_codes[start : start + take]
    if int(original_codes.shape[0]) != source_frames:
        raise ValueError(
            f"stats source_frames={source_frames} but original shape is {tuple(original_codes.shape)}"
        )
    if int(prompt_codes.shape[0]) != take:
        raise ValueError(f"stats prompt_frames={take} but prompt shape is {tuple(prompt_codes.shape)}")
    if int(original_codes.shape[1]) != int(prompt_codes.shape[1]):
        raise ValueError(
            f"codebook dimension mismatch: original={tuple(original_codes.shape)} prompt={tuple(prompt_codes.shape)}"
        )

    selected_counts = row_counter(selected)
    prompt_counts = row_counter(prompt_codes)
    row_bijection_ok = selected_counts == prompt_counts
    each_prompt_row_from_original = all(prompt_counts[row] <= row_counter(original_codes)[row] for row in prompt_counts)
    full_frame_rows_ok = int(prompt_codes.shape[1]) == int(expected_codebooks)
    row_order_changed = bool(
        take > 1
        and tuple(tuple(int(v) for v in row.tolist()) for row in selected.cpu())
        != tuple(tuple(int(v) for v in row.tolist()) for row in prompt_codes.cpu())
    )
    return {
        "expected_codebooks": int(expected_codebooks),
        "codebook_dim": int(prompt_codes.shape[1]),
        "selected_span_shape": list(selected.shape),
        "prompt_shape": list(prompt_codes.shape),
        "selected_span_start": start,
        "selected_span_end_exclusive": start + take,
        "full_frame_rows_ok": bool(full_frame_rows_ok),
        "each_prompt_row_from_original": bool(each_prompt_row_from_original),
        "row_bijection_ok": bool(row_bijection_ok),
        "row_order_changed": bool(row_order_changed),
        "checked_before_delay_pack": True,
        "check_description": (
            "The permuted S2 prompt is compared as raw [T,32] frame rows against the sampled "
            "C_ref span before MOSS delay-pattern packing."
        ),
    }


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def to_code_tensor(value: Any, *, field: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.dim() == 3 and int(tensor.shape[0]) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"{field} must be [T, C] or [1, T, C], got {tuple(tensor.shape)}")
    return tensor.contiguous()


def load_timbre_codes(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.timbre_codes_pt:
        payload = torch.load(Path(args.timbre_codes_pt).expanduser(), map_location="cpu")
        if isinstance(payload, dict):
            for key in ("timbre_ref_codes", "ref_codec", "codes", "audio_codes"):
                if key in payload:
                    return to_code_tensor(payload[key], field=key), {"source": str(args.timbre_codes_pt), "field": key}
        return to_code_tensor(payload, field="pt_payload"), {"source": str(args.timbre_codes_pt), "field": "pt_payload"}

    if not args.jsonl:
        raise ValueError("one of --jsonl or --timbre-codes-pt is required")
    jsonl = Path(args.jsonl).expanduser()
    for line_no, row in iter_jsonl(jsonl):
        references = row.get("reference_audio_codes") or row.get("ref_audio_codes")
        if not references or len(references) < 2 or references[1] is None:
            continue
        if args.case_id and str(row.get("case_id") or row.get("sample_id") or "") != str(args.case_id):
            continue
        return to_code_tensor(references[1], field="reference_audio_codes[1]"), {
            "source": str(jsonl),
            "line_no": line_no,
            "case_id": row.get("case_id") or row.get("sample_id"),
        }
    raise ValueError(f"no row with reference_audio_codes[1] found in {jsonl}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check 1d C_ref prompt permutation train-vs-infer consistency.")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--case-id", default="")
    ap.add_argument("--timbre-codes-pt", default="")
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--max-seconds", type=float, default=4.0)
    ap.add_argument("--frame-rate", type=float, default=12.5)
    ap.add_argument("--mode", default="shuffle", choices=("shuffle", "contiguous", "block_shuffle"))
    ap.add_argument("--block-seconds", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    timbre_codes, source = load_timbre_codes(args)
    train_prompt, train_stats = permute_ref_prompt_codes(
        timbre_codes,
        enabled=True,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        frame_rate=args.frame_rate,
        seed=args.seed,
        mode=args.mode,
        block_seconds=args.block_seconds,
    )
    infer_prompt, infer_stats = permute_ref_prompt_codes(
        timbre_codes,
        enabled=True,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        frame_rate=args.frame_rate,
        seed=args.seed,
        mode=args.mode,
        block_seconds=args.block_seconds,
    )
    train_stats_dict = train_stats.as_dict()
    infer_stats_dict = infer_stats.as_dict()
    row_integrity = frame_row_integrity_payload(timbre_codes, train_prompt, train_stats_dict)
    payload = {
        "status": "ok",
        "source": source,
        "seed": int(args.seed),
        "min_seconds": float(args.min_seconds),
        "max_seconds": float(args.max_seconds),
        "frame_rate": float(args.frame_rate),
        "mode": str(args.mode),
        "block_seconds": float(args.block_seconds),
        "timbre_ref_shape": list(timbre_codes.shape),
        "training_pack_stats": train_stats_dict,
        "inference_prompt_stats": infer_stats_dict,
        "row_integrity": row_integrity,
        "prompt_equal": bool(torch.equal(train_prompt, infer_prompt)),
        "stats_equal": train_stats_dict == infer_stats_dict,
        "training_pack_vector": [
            int(train_stats.enabled),
            int(train_stats.source_frames),
            int(train_stats.prompt_frames),
            int(train_stats.start),
            int(train_stats.shuffled),
        ],
    }
    if train_stats_dict != infer_stats_dict:
        payload["status"] = "failed"
        payload["error"] = "stats mismatch"
    elif not torch.equal(train_prompt, infer_prompt):
        payload["status"] = "failed"
        payload["error"] = "prompt tensor mismatch"
    elif args.mode != "contiguous" and int(train_stats.prompt_frames) > 1 and int(train_stats.shuffled) != 1:
        payload["status"] = "failed"
        payload["error"] = "permutation did not shuffle frames"
    elif not bool(row_integrity["full_frame_rows_ok"]):
        payload["status"] = "failed"
        payload["error"] = "S2 prompt rows are not full 32-codebook frames"
    elif not bool(row_integrity["each_prompt_row_from_original"]):
        payload["status"] = "failed"
        payload["error"] = "at least one permuted S2 row is not an original C_ref frame"
    elif not bool(row_integrity["row_bijection_ok"]):
        payload["status"] = "failed"
        payload["error"] = "permuted S2 rows are not a bijection of the sampled C_ref rows"
    if args.output_json:
        out = Path(args.output_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
