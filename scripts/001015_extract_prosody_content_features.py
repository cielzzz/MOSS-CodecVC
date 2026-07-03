#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl, safe_stem, stable_id


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract fixed prosody/content features for Ver2 auxiliary losses.")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--frame-ms", type=float, default=20.0)
    ap.add_argument("--hop-ms", type=float, default=20.0)
    ap.add_argument("--f0-min", type=float, default=50.0)
    ap.add_argument("--f0-max", type=float, default=600.0)
    ap.add_argument("--pause-db-below-peak", type=float, default=35.0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--include-target", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--progress-every", type=int, default=1000)
    return ap.parse_args()


def get_row_path(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value:
        return str(value)
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict) and meta.get(key):
        return str(meta[key])
    return None


def resolve_audio(path: str | None, audio_root: str = "") -> Path | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute() and audio_root:
        p = Path(audio_root).expanduser() / p
    return p


def read_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        wav, sr = sf.read(path, always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
    except Exception:
        with wave.open(str(path), "rb") as f:
            sr = f.getframerate()
            channels = f.getnchannels()
            sampwidth = f.getsampwidth()
            raw = f.readframes(f.getnframes())
        if sampwidth != 2:
            raise RuntimeError(f"fallback wave reader supports PCM16 only, got sampwidth={sampwidth}: {path}")
        wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            wav = wav.reshape(-1, channels).mean(axis=1)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        try:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as exc:
            raise RuntimeError(f"audio sample rate {sr} != {target_sr} and librosa resample is unavailable") from exc
    return np.asarray(wav, dtype=np.float32), sr


def frame_signal(wav: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if wav.size == 0:
        return np.zeros((1, frame_length), dtype=np.float32)
    if wav.size < frame_length:
        wav = np.pad(wav, (0, frame_length - wav.size))
    n_frames = 1 + int(math.ceil((wav.size - frame_length) / hop_length))
    padded_len = (n_frames - 1) * hop_length + frame_length
    if wav.size < padded_len:
        wav = np.pad(wav, (0, padded_len - wav.size))
    shape = (n_frames, frame_length)
    strides = (wav.strides[0] * hop_length, wav.strides[0])
    return np.lib.stride_tricks.as_strided(wav, shape=shape, strides=strides).copy()


def estimate_f0_autocorr(
    frames: np.ndarray,
    sr: int,
    *,
    f0_min: float,
    f0_max: float,
    energy: np.ndarray,
    pause_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    min_lag = max(1, int(sr / f0_max))
    max_lag = max(min_lag + 1, int(sr / f0_min))
    f0 = np.zeros(frames.shape[0], dtype=np.float32)
    voiced = np.zeros(frames.shape[0], dtype=np.float32)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    for idx, frame in enumerate(frames):
        if pause_mask[idx] or energy[idx] <= 1.0e-6:
            continue
        x = (frame - frame.mean()) * window
        corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
        if corr[0] <= 1.0e-8:
            continue
        search = corr[min_lag : min(max_lag, len(corr))]
        if search.size == 0:
            continue
        lag = int(np.argmax(search) + min_lag)
        confidence = float(corr[lag] / corr[0])
        if confidence < 0.25:
            continue
        f0[idx] = float(sr / lag)
        voiced[idx] = 1.0
    return f0, voiced


def estimate_f0(wav: np.ndarray, sr: int, frames: np.ndarray, hop_length: int, f0_min: float, f0_max: float, energy: np.ndarray, pause_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import librosa

        f0 = librosa.yin(
            wav,
            fmin=float(f0_min),
            fmax=float(f0_max),
            sr=sr,
            frame_length=frames.shape[1],
            hop_length=hop_length,
        ).astype(np.float32)
        if f0.shape[0] < frames.shape[0]:
            f0 = np.pad(f0, (0, frames.shape[0] - f0.shape[0]), constant_values=0.0)
        f0 = f0[: frames.shape[0]]
        voiced = ((f0 > 0) & np.isfinite(f0) & ~pause_mask).astype(np.float32)
        f0[voiced < 0.5] = 0.0
        return f0, voiced
    except Exception:
        return estimate_f0_autocorr(frames, sr, f0_min=f0_min, f0_max=f0_max, energy=energy, pause_mask=pause_mask)


def extract_prosody(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    wav, sr = read_audio(path, args.sample_rate)
    frame_length = max(1, int(sr * args.frame_ms / 1000.0))
    hop_length = max(1, int(sr * args.hop_ms / 1000.0))
    frames = frame_signal(wav, frame_length, hop_length)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1.0e-10).astype(np.float32)
    log_energy = np.log(rms + 1.0e-5).astype(np.float32)
    peak = float(np.max(rms)) if rms.size else 0.0
    threshold = peak * float(10.0 ** (-args.pause_db_below_peak / 20.0))
    pause_mask = (rms <= max(threshold, 1.0e-5)).astype(np.float32)
    f0, voiced = estimate_f0(
        wav,
        sr,
        frames,
        hop_length,
        args.f0_min,
        args.f0_max,
        rms,
        pause_mask.astype(bool),
    )
    logf0 = np.zeros_like(f0, dtype=np.float32)
    voiced_bool = voiced > 0.5
    logf0[voiced_bool] = np.log(np.clip(f0[voiced_bool], 1.0, None))
    return {
        "logf0": torch.from_numpy(logf0),
        "voiced_mask": torch.from_numpy(voiced.astype(np.float32)),
        "energy": torch.from_numpy(log_energy),
        "pause_mask": torch.from_numpy(pause_mask.astype(np.float32)),
        "duration_sec": torch.tensor(float(wav.shape[0]) / float(sr), dtype=torch.float32),
        "sample_rate": int(sr),
        "frame_ms": float(args.frame_ms),
        "hop_ms": float(args.hop_ms),
        "audio_path": str(path),
    }


def feature_path(root: Path, split: str, audio_path: Path, sample_id: str) -> Path:
    name = f"{safe_stem(sample_id)}_{stable_id(audio_path, length=12)}.pt"
    return root / split / name


def temp_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".tmp")


def done_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".done.json")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    feature_root = Path(args.feature_root).expanduser()
    output = Path(args.output_jsonl).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = temp_path(output)
    done_output = done_path(output)
    if tmp_output.exists():
        tmp_output.unlink()

    stats = {"rows": 0, "source": 0, "target": 0, "missing_audio": 0}
    progress_every = max(1, int(args.progress_every))
    with tmp_output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(args.input_jsonl):
            if args.max_rows > 0 and stats["rows"] >= args.max_rows:
                break
            row = dict(row)
            sample_id = str(row.get("sample_id") or stats["rows"])
            for split, key, out_key in (
                ("source", "source_audio", "source_prosody_path"),
                ("target", "target_audio", "target_prosody_path"),
            ):
                if split == "target" and not args.include_target:
                    continue
                audio = resolve_audio(get_row_path(row, key), args.audio_root)
                if audio is None or not audio.exists():
                    stats["missing_audio"] += 1
                    continue
                out_path = feature_path(feature_root, split, audio, sample_id)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if args.overwrite or not out_path.exists():
                    payload = extract_prosody(audio, args)
                    torch.save(payload, out_path)
                row[out_key] = str(out_path.resolve(strict=False))
                stats[split] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            if stats["rows"] % progress_every == 0:
                handle.flush()
                print(f"processed rows={stats['rows']} source={stats['source']} target={stats['target']} missing_audio={stats['missing_audio']}", flush=True)
        handle.flush()
    tmp_output.replace(output)
    written = stats["rows"]
    summary_path = output.with_suffix(".summary.json")
    summary = {"status": "complete", "written": written, **stats}
    write_json_atomic(summary_path, summary)
    write_json_atomic(done_output, summary)
    print(f"wrote {written} rows -> {output}")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
