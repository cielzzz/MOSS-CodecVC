#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
PERSISTENT_SCRIPT = ROOT / "scripts/004044_run_seedtts_validation_infer_persistent.py"
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/ver2_8_p0_e2_long_pause_no_text_valid20.jsonl"
DEFAULT_MODEL_PATH = (
    ROOT
    / "outputs/lora_runs/ver2_8_fixgate_sideonly_wavlmbnf_codecres_textrep5_lora_r16_a32_gbs64/step-9000"
)


def load_persistent_module():
    spec = importlib.util.spec_from_file_location("seedtts_persistent_infer", PERSISTENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load persistent infer module from {PERSISTENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem[:180] or "case"


def load_audio_mono(path: str | Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    wav = wav.float().mean(dim=0)
    return wav, int(sr)


def save_audio(path: Path, wav: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.float().view(1, -1), int(sr))


def rms_db(wav: torch.Tensor, sr: int, frame_ms: float, hop_ms: float) -> tuple[torch.Tensor, int]:
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if wav.numel() < frame:
        wav = torch.nn.functional.pad(wav, (0, frame - wav.numel()))
    frames = wav.unfold(0, frame, hop)
    rms = frames.pow(2).mean(dim=1).sqrt().clamp_min(1.0e-8)
    return 20.0 * torch.log10(rms), hop


def silence_intervals(
    wav: torch.Tensor,
    sr: int,
    *,
    frame_ms: float,
    hop_ms: float,
    silence_db_below_peak: float,
    min_silence_sec: float,
) -> list[tuple[float, float]]:
    db, hop = rms_db(wav, sr, frame_ms, hop_ms)
    if db.numel() == 0:
        return []
    threshold = float(db.max().item()) - abs(float(silence_db_below_peak))
    silent = (db <= threshold).cpu().bool().tolist()
    intervals: list[tuple[float, float]] = []
    start: int | None = None
    for idx, value in enumerate(silent):
        if value and start is None:
            start = idx
        if start is not None and ((not value) or idx == len(silent) - 1):
            end = idx if not value else idx + 1
            begin_sec = start * hop / float(sr)
            end_sec = min(float(wav.numel()) / float(sr), end * hop / float(sr))
            if end_sec - begin_sec >= float(min_silence_sec):
                intervals.append((begin_sec, end_sec))
            start = None
    return intervals


def choose_cut(
    desired: float,
    current_start: float,
    duration: float,
    intervals: list[tuple[float, float]],
    *,
    max_sec: float,
    min_segment_sec: float,
    search_sec: float,
) -> float:
    latest = min(duration, current_start + max_sec)
    earliest = current_start + min_segment_sec
    candidates = []
    for begin, end in intervals:
        center = 0.5 * (begin + end)
        if center < earliest or center > latest:
            continue
        if abs(center - desired) <= search_sec:
            candidates.append(center)
    if candidates:
        return min(candidates, key=lambda value: abs(value - desired))
    return latest


def split_audio(
    wav: torch.Tensor,
    sr: int,
    *,
    max_sec: float,
    min_segment_sec: float,
    frame_ms: float,
    hop_ms: float,
    silence_db_below_peak: float,
    min_silence_sec: float,
    search_sec: float,
) -> tuple[list[tuple[int, int]], list[tuple[float, float]]]:
    duration = float(wav.numel()) / float(sr)
    intervals = silence_intervals(
        wav,
        sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        silence_db_below_peak=silence_db_below_peak,
        min_silence_sec=min_silence_sec,
    )
    if duration <= max_sec:
        return [(0, int(wav.numel()))], intervals
    boundaries = [0.0]
    current = 0.0
    while duration - current > max_sec:
        desired = current + max_sec
        cut = choose_cut(
            desired,
            current,
            duration,
            intervals,
            max_sec=max_sec,
            min_segment_sec=min_segment_sec,
            search_sec=search_sec,
        )
        if cut <= current + 0.05:
            cut = min(duration, current + max_sec)
        boundaries.append(cut)
        current = cut
    boundaries.append(duration)
    spans = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        start = max(0, int(round(begin * sr)))
        stop = min(int(wav.numel()), int(round(end * sr)))
        if stop > start:
            spans.append((start, stop))
    return spans, intervals


def stitch_wavs(paths: list[Path], output_path: Path, *, crossfade_ms: float) -> dict[str, Any]:
    pieces = []
    sample_rate = None
    for path in paths:
        wav, sr = load_audio_mono(path)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            wav = torchaudio.functional.resample(wav.view(1, -1), sr, sample_rate).view(-1)
        pieces.append(wav)
    if not pieces:
        raise RuntimeError("no segment outputs to stitch")
    sr = int(sample_rate or 24000)
    cross = max(0, int(round(sr * float(crossfade_ms) / 1000.0)))
    stitched = pieces[0]
    for piece in pieces[1:]:
        if cross <= 0 or stitched.numel() < cross or piece.numel() < cross:
            stitched = torch.cat([stitched, piece], dim=0)
            continue
        fade_out = torch.linspace(1.0, 0.0, cross, dtype=stitched.dtype)
        fade_in = torch.linspace(0.0, 1.0, cross, dtype=piece.dtype)
        overlap = stitched[-cross:] * fade_out + piece[:cross] * fade_in
        stitched = torch.cat([stitched[:-cross], overlap, piece[cross:]], dim=0)
    save_audio(output_path, stitched, sr)
    return {
        "stitched_sample_rate": sr,
        "stitched_samples": int(stitched.numel()),
        "stitched_sec": float(stitched.numel()) / float(sr),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Segment long no-text source wavs and run per-segment CodecVC inference.")
    ap.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION_JSONL))
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--manifest-jsonl", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--case-id", action="append", default=[])
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--segment-max-sec", type=float, default=8.0)
    ap.add_argument("--min-segment-sec", type=float, default=2.0)
    ap.add_argument("--vad-frame-ms", type=float, default=30.0)
    ap.add_argument("--vad-hop-ms", type=float, default=10.0)
    ap.add_argument("--silence-db-below-peak", type=float, default=35.0)
    ap.add_argument("--min-silence-sec", type=float, default=0.18)
    ap.add_argument("--cut-search-sec", type=float, default=2.0)
    ap.add_argument("--crossfade-ms", type=float, default=20.0)

    # D2+D3 defaults. These remain explicit CLI args and do not change 004044 defaults.
    ap.add_argument("--source-semantic-monotonic-bias-strength", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--audio-temperature", type=float, default=1.1)
    ap.add_argument("--audio-top-p", type=float, default=0.7)
    ap.add_argument("--no-text-audio-temperature", type=float, default=1.1)
    ap.add_argument("--no-text-audio-top-p", type=float, default=0.7)
    ap.add_argument("--audio-top-k", type=int, default=None)
    ap.add_argument("--no-text-audio-top-k", type=int, default=None)
    return ap.parse_args()


def persistent_args(args: argparse.Namespace, module, manifest: Path) -> argparse.Namespace:
    argv = [
        str(PERSISTENT_SCRIPT),
        "--validation-jsonl",
        str(args.validation_jsonl),
        "--model-path",
        str(args.model_path),
        "--output-dir",
        str(args.output_dir),
        "--manifest-jsonl",
        str(manifest),
        "--mode",
        "no_text",
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--source-semantic-monotonic-bias-strength",
        str(float(args.source_semantic_monotonic_bias_strength)),
        "--temperature",
        str(float(args.temperature)),
        "--audio-temperature",
        str(float(args.audio_temperature)),
        "--audio-top-p",
        str(float(args.audio_top_p)),
        "--no-text-audio-temperature",
        str(float(args.no_text_audio_temperature)),
        "--no-text-audio-top-p",
        str(float(args.no_text_audio_top_p)),
    ]
    if args.audio_top_k is not None:
        argv.extend(["--audio-top-k", str(int(args.audio_top_k))])
    if args.no_text_audio_top_k is not None:
        argv.extend(["--no-text-audio-top-k", str(int(args.no_text_audio_top_k))])
    old_argv = sys.argv
    try:
        sys.argv = argv
        return module.parse_args()
    finally:
        sys.argv = old_argv


def selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = set(args.case_id)
    rows = []
    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards}), got {shard_index}")
    for idx, row in enumerate(iter_jsonl(Path(args.validation_jsonl))):
        case_id = str(row.get("case_id") or "")
        if wanted and case_id not in wanted:
            continue
        if str(row.get("mode") or "") != "no_text":
            continue
        if num_shards > 1 and (idx % num_shards) != shard_index:
            continue
        rows.append(row)
        if args.max_cases > 0 and len(rows) >= args.max_cases:
            break
    return rows


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    manifest = Path(args.manifest_jsonl).expanduser() if args.manifest_jsonl else output_dir / "manifest.jsonl"
    if args.overwrite and manifest.exists():
        manifest.unlink()
    rows = selected_rows(args)
    if not rows:
        print("[segmented-infer] no no-text rows selected", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    module = load_persistent_module()
    infer_args = persistent_args(args, module, manifest)
    engine = module.PersistentCodecVCInfer(infer_args)
    failures = 0

    for row in rows:
        case_id = str(row.get("case_id") or "")
        out_wav = output_dir / f"{safe_stem(case_id)}.wav"
        if out_wav.exists() and not args.overwrite:
            append_jsonl(manifest, {"case_id": case_id, "status": "skipped_exists", "output_wav": str(out_wav)})
            continue
        start_time = time.time()
        manifest_row: dict[str, Any] = {
            "case_id": case_id,
            "mode": row.get("mode"),
            "cell": row.get("cell"),
            "source_audio": row.get("source_audio"),
            "timbre_ref_audio": row.get("timbre_ref_audio"),
            "text": row.get("text"),
            "content_ref_text": row.get("content_ref_text"),
            "output_wav": str(out_wav),
            "seed": args.seed,
            "segmented_infer": True,
            "segment_max_sec": args.segment_max_sec,
            "crossfade_ms": args.crossfade_ms,
            "source_semantic_monotonic_bias_strength": args.source_semantic_monotonic_bias_strength,
            "no_text_d2_d3": True,
        }
        try:
            source_audio = str(row.get("source_audio") or "")
            wav, sr = load_audio_mono(source_audio)
            spans, silence = split_audio(
                wav,
                sr,
                max_sec=float(args.segment_max_sec),
                min_segment_sec=float(args.min_segment_sec),
                frame_ms=float(args.vad_frame_ms),
                hop_ms=float(args.vad_hop_ms),
                silence_db_below_peak=float(args.silence_db_below_peak),
                min_silence_sec=float(args.min_silence_sec),
                search_sec=float(args.cut_search_sec),
            )
            seg_src_dir = output_dir / "source_segments" / safe_stem(case_id)
            seg_out_dir = output_dir / "segment_outputs" / safe_stem(case_id)
            segment_outputs: list[Path] = []
            segment_rows = []
            for seg_idx, (begin, end) in enumerate(spans):
                seg_id = f"{case_id}__seg{seg_idx:02d}"
                seg_source = seg_src_dir / f"{safe_stem(seg_id)}.wav"
                seg_output = seg_out_dir / f"{safe_stem(seg_id)}.wav"
                save_audio(seg_source, wav[begin:end], sr)
                seg_row = dict(row)
                seg_row["case_id"] = seg_id
                seg_row["source_audio"] = str(seg_source)
                seg_row["source_segment_parent_case_id"] = case_id
                seg_row["source_segment_index"] = seg_idx
                seg_row["source_segment_begin_sec"] = begin / float(sr)
                seg_row["source_segment_end_sec"] = end / float(sr)
                if not seg_output.exists() or args.overwrite:
                    engine.run_case(seg_row, seg_output)
                segment_outputs.append(seg_output)
                segment_rows.append(
                    {
                        "index": seg_idx,
                        "source_wav": str(seg_source),
                        "output_wav": str(seg_output),
                        "begin_sec": begin / float(sr),
                        "end_sec": end / float(sr),
                        "duration_sec": (end - begin) / float(sr),
                    }
                )
            stitch_stats = stitch_wavs(segment_outputs, out_wav, crossfade_ms=float(args.crossfade_ms))
            elapsed = round(time.time() - start_time, 3)
            manifest_row.update(
                {
                    "status": "ok",
                    "returncode": 0,
                    "elapsed_sec": elapsed,
                    "output_exists": out_wav.exists(),
                    "source_duration_sec": float(wav.numel()) / float(sr),
                    "num_segments": len(spans),
                    "segments": segment_rows,
                    "vad_silence_intervals": silence,
                    **stitch_stats,
                }
            )
            print(f"[segmented-infer] done {case_id} segments={len(spans)} elapsed={elapsed}s", flush=True)
        except Exception as exc:
            failures += 1
            elapsed = round(time.time() - start_time, 3)
            manifest_row.update(
                {
                    "status": "failed",
                    "returncode": 1,
                    "elapsed_sec": elapsed,
                    "output_exists": out_wav.exists(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[segmented-infer] failed {case_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                append_jsonl(manifest, manifest_row)
                break
        append_jsonl(manifest, manifest_row)
    print(f"[segmented-infer] complete total={len(rows)} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
