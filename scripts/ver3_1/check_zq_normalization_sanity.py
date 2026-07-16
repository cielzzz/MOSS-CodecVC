#!/usr/bin/env python
"""Verify zq normalization round-trip against the real frozen MOSS decoder.

This check consumes the canonical ``channel_stats.pt`` and a deterministic
prefix of the verified zq manifest.  It compares direct latent decoding with
``normalize -> inverse-normalize -> decode`` and publishes both JSON and
Markdown evidence.  No waveform or latent files are copied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.audio import decode_latents, denormalize_zq, load_zq_channel_stats, normalize_zq
from moss_codecvc.audio.zq_normalization import atomic_json_save, sha256_file
from moss_codecvc.config import deep_get, load_config
from moss_codecvc.moss_codec import MossCodec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats",
        default=str(ROOT / "prepared/zq_targets_v1/channel_stats.pt"),
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "prepared/zq_targets_v1/manifest.jsonl"),
    )
    parser.add_argument("--config", default=str(ROOT / "configs/remote_full.yaml"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--codec-dtype", choices=("float32",), default="float32")
    parser.add_argument("--latent-atol", type=float, default=1.0e-6)
    parser.add_argument("--waveform-atol", type=float, default=1.0e-5)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "testset/outputs/ver3_1_batch46_zq_normalization_sanity_20260716"),
    )
    return parser.parse_args(argv)


def _manifest_prefix(path: Path, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_no} is not an object")
            row["_manifest_line_no"] = line_no
            rows.append(row)
            if len(rows) >= count:
                break
    if len(rows) != count:
        raise ValueError(f"manifest has only {len(rows)} usable rows, requested {count}")
    return rows


def _case_name(row: dict[str, Any]) -> str:
    return str(row.get("utterance_id") or row.get("record_id") or f"line-{row['_manifest_line_no']}")


@torch.inference_mode()
def check_case(
    codec: MossCodec,
    stats: dict[str, Any],
    row: dict[str, Any],
    *,
    latent_atol: float,
    waveform_atol: float,
) -> dict[str, Any]:
    path = Path(str(row.get("output_path") or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, allow_pickle=False)
    expected_dim = int(stats["latent_dim"])
    if array.dtype != np.float32 or array.ndim != 2 or int(array.shape[0]) != expected_dim:
        raise ValueError(f"invalid zq target {path}: dtype={array.dtype}, shape={array.shape}")
    zq = torch.from_numpy(array).unsqueeze(0).to(device=codec.device, dtype=torch.float32)
    normalized = normalize_zq(zq, stats, channel_dim=1)
    restored = denormalize_zq(normalized, stats, channel_dim=1)
    latent_max_abs = float((restored - zq).abs().max().item())
    lengths = torch.tensor([int(zq.shape[-1])], dtype=torch.long, device=codec.device)
    direct_wav, direct_lengths = decode_latents(codec.model, zq, lengths)
    restored_wav, restored_lengths = decode_latents(codec.model, restored, lengths)
    lengths_equal = bool(torch.equal(direct_lengths, restored_lengths))
    waveform_max_abs = float((restored_wav - direct_wav).abs().max().item())
    latent_pass = bool(latent_max_abs < float(latent_atol))
    waveform_pass = bool(lengths_equal and waveform_max_abs <= float(waveform_atol))
    return {
        "case": _case_name(row),
        "manifest_line_no": int(row["_manifest_line_no"]),
        "split": row.get("split"),
        "zq_path": str(path),
        "zq_sha256": str(row.get("output_sha256") or sha256_file(path)),
        "shape": list(zq.shape),
        "latent_max_abs_diff": latent_max_abs,
        "latent_exact": bool(torch.equal(restored, zq)),
        "latent_atol": float(latent_atol),
        "latent_pass": latent_pass,
        "waveform_max_abs_diff": waveform_max_abs,
        "waveform_exact": bool(torch.equal(restored_wav, direct_wav)),
        "waveform_atol": float(waveform_atol),
        "waveform_lengths_equal": lengths_equal,
        "waveform_samples": int(direct_lengths[0].item()),
        "finite": bool(torch.isfinite(normalized).all() and torch.isfinite(restored_wav).all()),
        "pass": bool(latent_pass and waveform_pass),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Batch-46 zq normalization sanity",
        "",
        f"- Status: **{payload['status']}**",
        f"- Stats: `{payload['stats']}`",
        f"- Stats SHA256: `{payload['stats_sha256']}`",
        f"- Cases: `{payload['passed_cases']}/{payload['num_cases']}` passed",
        f"- Max latent abs diff: `{payload['max_latent_abs_diff']:.9g}`",
        f"- Max waveform abs diff: `{payload['max_waveform_abs_diff']:.9g}`",
        "",
        "| Case | Split | Frames | latent max abs | waveform max abs | length | Pass |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in payload["cases"]:
        lines.append(
            f"| {row['case']} | {row.get('split') or ''} | {row['shape'][-1]} | "
            f"{row['latent_max_abs_diff']:.9g} | {row['waveform_max_abs_diff']:.9g} | "
            f"{'equal' if row['waveform_lengths_equal'] else 'DIFF'} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.num_samples) <= 0:
        raise ValueError("num-samples must be positive")
    if float(args.latent_atol) <= 0 or float(args.waveform_atol) < 0:
        raise ValueError("latent-atol must be positive and waveform-atol non-negative")
    stats_path = Path(args.stats).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    stats = load_zq_channel_stats(stats_path)
    rows = _manifest_prefix(manifest_path, int(args.num_samples))
    config = load_config(config_path)

    # Transformers 5 emits a 1600-line materialization progress bar for this
    # local checkpoint unless progress bars are disabled explicitly.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
    except Exception:
        pass
    started = time.time()
    codec = MossCodec(
        deep_get(config, "moss.codec_path"),
        moss_root=deep_get(config, "moss.root"),
        device=str(args.device),
        dtype=str(args.codec_dtype),
    )
    cases = [
        check_case(
            codec,
            stats,
            row,
            latent_atol=float(args.latent_atol),
            waveform_atol=float(args.waveform_atol),
        )
        for row in rows
    ]
    passed = sum(bool(row["pass"]) for row in cases)
    payload = {
        "schema": "ver3_1_batch46_zq_normalization_sanity_v1",
        "status": "passed" if passed == len(cases) else "failed",
        "created_at_unix": time.time(),
        "elapsed_sec": time.time() - started,
        "stats": str(stats_path),
        "stats_sha256": sha256_file(stats_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "codec_path": str(Path(str(deep_get(config, "moss.codec_path"))).expanduser().resolve()),
        "device": str(codec.device),
        "num_cases": len(cases),
        "passed_cases": passed,
        "max_latent_abs_diff": max(row["latent_max_abs_diff"] for row in cases),
        "max_waveform_abs_diff": max(row["waveform_max_abs_diff"] for row in cases),
        "cases": cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(payload, output_dir / "report.json")
    _write_markdown(payload, output_dir / "REPORT.md")
    print(json.dumps({key: payload[key] for key in (
        "status", "num_cases", "passed_cases", "max_latent_abs_diff", "max_waveform_abs_diff"
    )}, ensure_ascii=False, indent=2))
    if payload["status"] != "passed":
        raise RuntimeError(f"zq normalization sanity failed; see {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
