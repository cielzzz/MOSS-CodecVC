#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_SPEAKER_SIM_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize SeedTTS ablation runs with silence and speaker metrics.")
    ap.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION_JSONL))
    ap.add_argument("--run", action="append", required=True, help="NAME=EVAL_DIR. May be repeated.")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--summary-md", required=True)
    ap.add_argument("--skip-speaker", action="store_true")
    ap.add_argument("--speaker-device", default="cuda:0")
    ap.add_argument("--speaker-sim-root", default=str(DEFAULT_SPEAKER_SIM_ROOT))
    ap.add_argument("--failure-cer-threshold", type=float, default=0.30)
    return ap.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def std(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text


def lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for ch in long:
        cur = [0]
        for idx, other in enumerate(short, start=1):
            cur.append(prev[idx - 1] + 1 if ch == other else max(prev[idx], cur[-1]))
        prev = cur
    return prev[-1]


def coverage(reference: str, hypothesis: str) -> float | None:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return None
    return lcs_len(ref, hyp) / max(1, len(ref))


def silence_metrics(path: str | Path) -> dict[str, float | None]:
    path = Path(path)
    if not path.exists():
        return {
            "audio_sec": None,
            "tail_silence_sec": None,
            "tail_silence_ratio": None,
            "total_silence_ratio": None,
        }
    wav, sr = torchaudio.load(str(path))
    if wav.numel() == 0:
        return {
            "audio_sec": 0.0,
            "tail_silence_sec": 0.0,
            "tail_silence_ratio": None,
            "total_silence_ratio": None,
        }
    wav = wav.float().mean(dim=0)
    frame = max(1, int(round(0.050 * sr)))
    hop = max(1, int(round(0.010 * sr)))
    if wav.numel() < frame:
        pad = frame - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad))
    frames = wav.unfold(0, frame, hop)
    rms = frames.pow(2).mean(dim=1).sqrt()
    peak = float(rms.max().item()) if rms.numel() else 0.0
    if peak <= 1.0e-8:
        silent = torch.ones_like(rms, dtype=torch.bool)
    else:
        db = 20.0 * torch.log10(rms.clamp_min(1.0e-8))
        peak_db = 20.0 * math.log10(max(peak, 1.0e-8))
        threshold = max(-45.0, peak_db - 38.0)
        silent = db < threshold
    tail_frames = 0
    for item in reversed(silent.tolist()):
        if not item:
            break
        tail_frames += 1
    audio_sec = float(wav.numel()) / float(sr)
    tail_sec = min(audio_sec, tail_frames * hop / float(sr))
    return {
        "audio_sec": audio_sec,
        "tail_silence_sec": tail_sec,
        "tail_silence_ratio": tail_sec / audio_sec if audio_sec > 0 else None,
        "total_silence_ratio": float(silent.float().mean().item()) if silent.numel() else None,
    }


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value).expanduser()
        return path.name, path
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path).expanduser()


def find_asr_jsonl(run_name: str, run_dir: Path) -> Path:
    preferred = run_dir / f"{run_name}.asr_eval.jsonl"
    if preferred.exists():
        return preferred
    candidates = [p for p in sorted(run_dir.glob("*.asr_eval.jsonl")) if ".shard" not in p.name]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No merged *.asr_eval.jsonl found under {run_dir}")


def read_manifests(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("manifest*.jsonl")):
        for row in iter_jsonl(path):
            case_id = str(row.get("case_id") or "")
            if case_id:
                rows[case_id] = row
    return rows


class SpeakerScorer:
    def __init__(self, root: Path, device: str) -> None:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from speaker_similarity import SpeakerSimilarity

        self.backend = SpeakerSimilarity(device=device)
        self.cache: dict[str, Any] = {}

    def embed(self, path: str | Path):
        key = str(path)
        if key not in self.cache:
            self.cache[key] = self.backend.embed_from_file(key)
        return self.cache[key]

    def similarity(self, a: str | Path, b: str | Path) -> float | None:
        try:
            return float(self.backend.compute_similarity(self.embed(a), self.embed(b)))
        except Exception as exc:
            print(f"[speaker-sim] failed {a} vs {b}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None


def summarize_group(rows: list[dict[str, Any]], failure_threshold: float) -> dict[str, Any]:
    cer = [finite(row.get("cer_tgt")) for row in rows]
    failure_values = [v for v in cer if v is not None]
    return {
        "n": len(rows),
        "failure_rate_cer_gt_threshold": (
            sum(1 for value in failure_values if value > failure_threshold) / len(failure_values)
            if failure_values
            else None
        ),
        "cer_mean": mean(cer),
        "cer_std": std(cer),
        "coverage_mean": mean([finite(row.get("coverage")) for row in rows]),
        "coverage_std": std([finite(row.get("coverage")) for row in rows]),
        "repeat_mean": mean([finite(row.get("repeat_score")) for row in rows]),
        "repeat_std": std([finite(row.get("repeat_score")) for row in rows]),
        "tail_silence_ratio_mean": mean([finite(row.get("tail_silence_ratio")) for row in rows]),
        "tail_silence_ratio_std": std([finite(row.get("tail_silence_ratio")) for row in rows]),
        "total_silence_ratio_mean": mean([finite(row.get("total_silence_ratio")) for row in rows]),
        "total_silence_ratio_std": std([finite(row.get("total_silence_ratio")) for row in rows]),
        "sim_gen_ref_mean": mean([finite(row.get("sim_gen_ref")) for row in rows]),
        "sim_gen_ref_std": std([finite(row.get("sim_gen_ref")) for row in rows]),
        "sim_gen_source_mean": mean([finite(row.get("sim_gen_source")) for row in rows]),
        "sim_gen_source_std": std([finite(row.get("sim_gen_source")) for row in rows]),
        "delay_minus_min_audio_mean": mean([finite(row.get("delay_minus_min_audio")) for row in rows]),
        "delay_minus_min_audio_std": std([finite(row.get("delay_minus_min_audio")) for row in rows]),
    }


def main() -> int:
    args = parse_args()
    validation = {str(row.get("case_id") or ""): row for row in iter_jsonl(Path(args.validation_jsonl))}
    scorer = None if args.skip_speaker else SpeakerScorer(Path(args.speaker_sim_root), args.speaker_device)

    per_case: list[dict[str, Any]] = []
    run_summaries: dict[str, dict[str, Any]] = {}
    for run_spec in args.run:
        run_name, run_dir = parse_run(run_spec)
        asr_jsonl = find_asr_jsonl(run_name, run_dir)
        manifests = read_manifests(run_dir)
        rows: list[dict[str, Any]] = []
        for asr in iter_jsonl(asr_jsonl):
            case_id = str(asr.get("case_id") or asr.get("sample_id") or "")
            val = validation.get(case_id, {})
            manifest = manifests.get(case_id, {})
            target_audio = str(asr.get("target_audio") or manifest.get("output_wav") or "")
            source_audio = str(asr.get("source_audio") or val.get("source_audio") or manifest.get("source_audio") or "")
            timbre_ref_audio = str(asr.get("timbre_ref_audio") or val.get("timbre_ref_audio") or manifest.get("timbre_ref_audio") or "")
            ref_text = str(asr.get("content_ref_text") or val.get("content_ref_text") or val.get("source_text") or "")
            item = {
                "run": run_name,
                "case_id": case_id,
                "seed": manifest.get("seed"),
                "mode": asr.get("mode") or val.get("mode"),
                "cell": asr.get("cell") or val.get("cell"),
                "source_audio": source_audio,
                "timbre_ref_audio": timbre_ref_audio,
                "target_audio": target_audio,
                "content_ref_text": ref_text,
                "asr_tgt_text": asr.get("asr_tgt_text"),
                "cer_tgt": finite(asr.get("cer_tgt")),
                "wer_tgt": finite(asr.get("wer_tgt")),
                "repeat_score": finite(asr.get("repeat_score")),
                "coverage": coverage(ref_text, str(asr.get("asr_tgt_text") or "")),
                "content_keep": asr.get("content_keep"),
                "content_filter_reason": asr.get("content_filter_reason"),
                "source_semantic_monotonic_bias_strength": manifest.get("source_semantic_monotonic_bias_strength"),
                "source_semantic_progress_clock": manifest.get("source_semantic_progress_clock"),
                "source_semantic_release_after_progress": manifest.get("source_semantic_release_after_progress"),
                "no_text_soft_duration_budget": manifest.get("no_text_soft_duration_budget"),
            }
            item.update(silence_metrics(target_audio))
            structure = manifest.get("generation_structure") or {}
            if isinstance(structure, dict):
                item["first_delay_pos"] = structure.get("first_delay_pos")
                item["delay_minus_min_audio"] = structure.get("delay_minus_min_audio")
                item["gen_slot_count"] = structure.get("gen_slot_count")
                item["delay_slot_count"] = structure.get("delay_slot_count")
                item["audio_end_positions"] = json.dumps(structure.get("audio_end_positions"), ensure_ascii=False)
                item["im_end_positions"] = json.dumps(structure.get("im_end_positions"), ensure_ascii=False)
            item["generation_max_new_tokens"] = manifest.get("generation_max_new_tokens")
            item["generation_min_new_tokens"] = manifest.get("generation_min_new_tokens")
            item["generation_min_audio_tokens"] = manifest.get("generation_min_audio_tokens")
            if scorer is not None and target_audio and Path(target_audio).exists():
                item["sim_gen_ref"] = scorer.similarity(target_audio, timbre_ref_audio) if timbre_ref_audio else None
                item["sim_gen_source"] = scorer.similarity(target_audio, source_audio) if source_audio else None
            else:
                item["sim_gen_ref"] = None
                item["sim_gen_source"] = None
            rows.append(item)
            per_case.append(item)
        run_summaries[run_name] = summarize_group(rows, float(args.failure_cer_threshold))
        run_summaries[run_name]["asr_jsonl"] = str(asr_jsonl)
        run_summaries[run_name]["run_dir"] = str(run_dir)

    fields = [
        "run",
        "case_id",
        "seed",
        "mode",
        "cell",
        "cer_tgt",
        "wer_tgt",
        "coverage",
        "repeat_score",
        "tail_silence_sec",
        "tail_silence_ratio",
        "total_silence_ratio",
        "sim_gen_ref",
        "sim_gen_source",
        "first_delay_pos",
        "delay_minus_min_audio",
        "gen_slot_count",
        "delay_slot_count",
        "audio_end_positions",
        "im_end_positions",
        "generation_max_new_tokens",
        "generation_min_new_tokens",
        "generation_min_audio_tokens",
        "source_semantic_monotonic_bias_strength",
        "source_semantic_progress_clock",
        "source_semantic_release_after_progress",
        "no_text_soft_duration_budget",
        "content_keep",
        "content_filter_reason",
        "target_audio",
        "source_audio",
        "timbre_ref_audio",
        "content_ref_text",
        "asr_tgt_text",
    ]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in per_case:
            writer.writerow(row)

    summary_payload = {
        "failure_cer_threshold": float(args.failure_cer_threshold),
        "runs": run_summaries,
        "per_case_csv": str(output_csv),
    }
    Path(args.summary_json).write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SeedTTS Ablation Metrics",
        "",
        f"failure threshold: `CER > {float(args.failure_cer_threshold):.2f}`",
        "",
        "| run | n | fail rate | CER mean±std | coverage mean±std | repeat | tail silence | total silence | sim gen-ref | sim gen-source | delay-min_audio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, payload in run_summaries.items():
        lines.append(
            "| {run} | {n} | {fail} | {cer_m}±{cer_s} | {cov_m}±{cov_s} | {rep} | {tail} | {sil} | {ref} | {src} | {delay} |".format(
                run=run_name,
                n=payload["n"],
                fail=fmt(payload["failure_rate_cer_gt_threshold"]),
                cer_m=fmt(payload["cer_mean"]),
                cer_s=fmt(payload["cer_std"]),
                cov_m=fmt(payload["coverage_mean"]),
                cov_s=fmt(payload["coverage_std"]),
                rep=fmt(payload["repeat_mean"]),
                tail=fmt(payload["tail_silence_ratio_mean"]),
                sil=fmt(payload["total_silence_ratio_mean"]),
                ref=fmt(payload["sim_gen_ref_mean"]),
                src=fmt(payload["sim_gen_source_mean"]),
                delay=fmt(payload["delay_minus_min_audio_mean"]),
            )
        )
    lines.extend(["", f"per-case CSV: `{output_csv}`", f"summary JSON: `{args.summary_json}`", ""])
    Path(args.summary_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[ablation-summary] wrote {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
