#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def nested_get(row: dict[str, Any], key: str) -> Any | None:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def pick_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = nested_get(row, key)
        if value not in (None, ""):
            return str(value)
    return ""


def normalize_text(text: str) -> str:
    out: list[str] = []
    for ch in str(text or "").lower():
        code = ord(ch)
        if ch.isalnum() or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            out.append(ch)
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for idx, ca in enumerate(a, start=1):
        cur = [idx]
        for jdx, cb in enumerate(b, start=1):
            cur.append(min(prev[jdx] + 1, cur[-1] + 1, prev[jdx - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def text_cer(ref: str, hyp: str) -> float | None:
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    if not ref_norm and not hyp_norm:
        return None
    return float(edit_distance(ref_norm, hyp_norm)) / float(max(1, len(ref_norm)))


def load_embedding(path: str | Path) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu")
    if torch.is_tensor(payload):
        emb = payload
    elif isinstance(payload, dict):
        emb = None
        for key in ("speaker_embedding", "embedding", "emb", "xvector", "vector"):
            value = payload.get(key)
            if value is not None:
                emb = value
                break
        if emb is None:
            raise ValueError(f"no speaker embedding tensor in {path}")
    else:
        emb = torch.as_tensor(payload)
    emb = torch.as_tensor(emb, dtype=torch.float32)
    if emb.dim() > 1:
        emb = emb.reshape(-1, emb.shape[-1]).mean(dim=0)
    if emb.dim() != 1:
        raise ValueError(f"embedding must flatten to [D], got {tuple(emb.shape)} from {path}")
    return F.normalize(emb.float(), dim=0)


def finite_mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def sample_rows(path: Path, sample_size: int, seed: int) -> tuple[int, list[tuple[int, dict[str, Any]]]]:
    rng = random.Random(seed)
    sampled: list[tuple[int, dict[str, Any]]] = []
    rows = 0
    for line_no, row in iter_jsonl(path):
        rows += 1
        if len(sampled) < sample_size:
            sampled.append((line_no, row))
            continue
        idx = rng.randrange(rows)
        if idx < sample_size:
            sampled[idx] = (line_no, row)
    return rows, sampled


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity check v2 real-target no-text prepared data.")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    path = Path(args.jsonl).expanduser()
    output_json = Path(args.output_json).expanduser() if args.output_json else path.with_suffix(path.suffix + ".v2_sanity.json")
    total_rows, sampled = sample_rows(path, int(args.sample_size), int(args.seed))

    stats: Counter[str] = Counter()
    langs: Counter[str] = Counter()
    ref_target_same_flags: list[bool] = []
    ref_target_sims: list[float] = []
    source_target_sims: list[float] = []
    text_cers: list[float] = []
    examples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for line_no, row in sampled:
        stats["sampled"] += 1
        lang = str(nested_get(row, "language") or "unknown").lower()
        langs[lang] += 1
        timbre_spk = nested_get(row, "timbre_ref_speaker_id")
        target_spk = nested_get(row, "target_speaker_id")
        if timbre_spk not in (None, "") and target_spk not in (None, ""):
            ref_target_same_flags.append(str(timbre_spk) == str(target_spk))

        ref_emb_path = nested_get(row, "timbre_ref_speaker_embedding_path")
        target_emb_path = nested_get(row, "target_speaker_embedding_path")
        source_emb_path = nested_get(row, "source_speaker_embedding_path")
        try:
            if ref_emb_path and target_emb_path:
                ref = load_embedding(ref_emb_path)
                target = load_embedding(target_emb_path)
                ref_target_sims.append(float(torch.dot(ref, target).item()))
            if source_emb_path and target_emb_path:
                source = load_embedding(source_emb_path)
                target = load_embedding(target_emb_path)
                source_target_sims.append(float(torch.dot(source, target).item()))
        except Exception as exc:  # noqa: BLE001
            errors.append({"line_no": line_no, "sample_id": row.get("sample_id"), "error": f"{type(exc).__name__}: {exc}"})

        source_text = pick_text(row, ("source_text", "asr_src_text", "content_ref_text"))
        target_text = pick_text(row, ("target_text", "asr_tgt_text", "content_ref_text"))
        cer = text_cer(target_text, source_text)
        if cer is not None:
            text_cers.append(cer)

        if len(examples) < 10:
            examples.append(
                {
                    "line_no": line_no,
                    "sample_id": row.get("sample_id"),
                    "language": lang,
                    "timbre_ref_speaker_id": timbre_spk,
                    "target_speaker_id": target_spk,
                    "source_audio": nested_get(row, "source_audio"),
                    "timbre_ref_audio": nested_get(row, "timbre_ref_audio"),
                    "target_audio": nested_get(row, "target_audio"),
                }
            )

    ref_target_same_rate = None
    if ref_target_same_flags:
        ref_target_same_rate = sum(1 for item in ref_target_same_flags if item) / float(len(ref_target_same_flags))
    ref_target_mean = finite_mean(ref_target_sims)
    source_target_mean = finite_mean(source_target_sims)
    sim_delta = None
    if ref_target_mean is not None and source_target_mean is not None:
        sim_delta = ref_target_mean - source_target_mean

    summary = {
        "status": "complete",
        "jsonl": str(path.resolve(strict=False)),
        "total_rows": total_rows,
        "sample_size": len(sampled),
        "seed": int(args.seed),
        "language_counts": dict(langs.most_common()),
        "speaker_id_ref_target_same_rate": ref_target_same_rate,
        "ecapa_ref_u2_vs_target_u1_mean": ref_target_mean,
        "ecapa_source_u1prime_vs_target_u1_mean": source_target_mean,
        "ecapa_ref_target_minus_source_target": sim_delta,
        "source_target_text_cer_proxy_mean": finite_mean(text_cers),
        "source_target_text_cer_proxy_p95": None
        if not text_cers
        else sorted(text_cers)[min(len(text_cers) - 1, int(math.ceil(0.95 * len(text_cers)) - 1))],
        "notes": [
            "CER here is a manifest text proxy from source_text/target_text, not fresh ASR on u1' audio.",
            "ECAPA source is u1' if source_speaker_embedding_path was extracted from source_audio.",
        ],
        "errors": errors[:20],
        "examples": examples,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
