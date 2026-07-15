#!/usr/bin/env python
"""Audit v1 manifests before launching Batch-45 Step 3."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


def parse_specs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected SPLIT=JSONL, got {value!r}")
        split, path = value.split("=", 1)
        result.append((split, Path(path).expanduser().resolve()))
    return result


def nested(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    return meta.get(key) if isinstance(meta, dict) else None


def feature_shape(path: str | None) -> tuple[int, int] | None:
    if not path or not Path(path).is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload if torch.is_tensor(payload) else None
    if isinstance(payload, dict):
        for key in ("wavlm_bnf_features", "wavlm_features", "semantic_features"):
            if torch.is_tensor(payload.get(key)):
                value = payload[key]
                break
    if value is None or value.dim() != 2:
        return None
    return int(value.shape[0]), int(value.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", required=True, metavar="SPLIT=JSONL")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--shape-probe", type=int, default=100)
    args = ap.parse_args()
    specs = parse_specs(args.input)
    summary: dict[str, Any] = {
        "schema": "ver3_1_step3_input_audit_v1",
        "manifests": {},
        "label_fields": Counter(),
        "feature_dims": Counter(),
        "phoneme_alignment_rows": 0,
    }
    for split, path in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        stats = Counter()
        lengths: list[int] = []
        shapes: list[tuple[int, int]] = []
        samples = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                stats["rows"] += 1
                mode = str(row.get("moss_codecvc_mode") or split).lower()
                stats[f"mode_{mode}"] += 1
                source_sidecar = nested(row, "source_wavlm_bnf_features_path")
                target_sidecar = nested(row, "target_wavlm_bnf_features_path")
                source_audio = nested(row, "source_audio")
                stats["source_sidecar_present"] += int(bool(source_sidecar and Path(str(source_sidecar)).is_file()))
                stats["target_sidecar_present"] += int(bool(target_sidecar and Path(str(target_sidecar)).is_file()))
                stats["source_audio_present"] += int(bool(source_audio and Path(str(source_audio)).is_file()))
                label = next(
                    (key for key in ("phoneme_ids", "phoneme_token_ids", "content_phoneme_ids", "content_token_ids") if isinstance(row.get(key), list) and row.get(key)),
                    None,
                )
                if label:
                    stats[f"label_{label}"] += 1
                    summary["label_fields"][label] += 1
                if any(key in row for key in ("phoneme_ids", "phoneme_token_ids", "content_phoneme_ids")):
                    summary["phoneme_alignment_rows"] += 1
                if samples < int(args.shape_probe):
                    shape = feature_shape(str(source_sidecar) if source_sidecar else None)
                    if shape is not None:
                        shapes.append(shape)
                        lengths.append(shape[0])
                        summary["feature_dims"][str(shape[1])] += 1
                    samples += 1
        summary["manifests"][split] = {
            "path": str(path),
            "stats": dict(stats),
            "shape_probe": {
                "n": len(shapes),
                "dims": sorted({dim for _, dim in shapes}),
                "frames_min": min(lengths) if lengths else None,
                "frames_mean": statistics.mean(lengths) if lengths else None,
                "frames_max": max(lengths) if lengths else None,
            },
        }
    summary["label_fields"] = dict(summary["label_fields"])
    summary["feature_dims"] = dict(summary["feature_dims"])
    summary["interpretation"] = {
        "actual_wavlm_contract": "v1 sidecars are microsoft/wavlm-base-plus layer 9, 768-D, about 50 Hz",
        "phoneme_alignment_available": bool(summary["phoneme_alignment_rows"]),
        "text_source_semantic_policy": "do not use text source_audio BNF for target-text supervision; text source is prosody/style only",
        "gate_warning": "content_token_ids are SentencePiece pseudo-labels, not MFA/WhisperX phoneme alignments",
    }
    out_json = Path(args.output_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Batch-45 Step 3 input audit",
        "",
        f"- JSON: `{out_json}`",
        f"- True phoneme/MFA/WhisperX fields present: **{summary['phoneme_alignment_rows'] > 0}**",
        "- Existing semantic labels are `content_token_ids` (SentencePiece), so the adapter probe reports pseudo-token accuracy, not phoneme accuracy.",
        "- v1 source sidecars use WavLM-base-plus layer 9 (768-D); the 1024-D WavLM-Large contract is not present in this workspace.",
        "- Text rows without a source sidecar are intentionally excluded from this BNF adapter pretraining; their source audio is prosody/style only, and target-side BNF is not used as a fallback.",
        "",
        "## Per manifest",
        "",
        "| Split | Rows | Source sidecar | Target sidecar | Source audio | Labels | Feature dims |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for split, payload in summary["manifests"].items():
        s = payload["stats"]
        labels = ", ".join(f"{k.removeprefix('label_')}={v}" for k, v in s.items() if k.startswith("label_")) or "none"
        dims = ",".join(str(x) for x in payload["shape_probe"]["dims"]) or "unknown"
        lines.append(
            f"| {split} | {s.get('rows', 0)} | {s.get('source_sidecar_present', 0)} | "
            f"{s.get('target_sidecar_present', 0)} | {s.get('source_audio_present', 0)} | {labels} | {dims} |"
        )
    Path(args.output_report).expanduser().resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
