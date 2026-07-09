#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config
from moss_codecvc.io_utils import load_torch_file, safe_stem, stable_id
from moss_codecvc.moss_codec import MossCodec


DEFAULT_INPUT = ROOT / "trainset/ver2_8_prepared/no_text.train.jsonl"
DEFAULT_CONFIG = ROOT / "configs/remote_full.yaml"
DEFAULT_SPEAKER_SIM_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction"
)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_get(row: dict[str, Any], key: str) -> Any:
    if key in row and row[key] is not None:
        return row[key]
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def load_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_sr and int(sr) != int(target_sr):
        import librosa

        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(target_sr))
        sr = int(target_sr)
    return np.asarray(audio, dtype=np.float32), int(sr)


def fix_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) == length:
        return audio.astype(np.float32, copy=False)
    if len(audio) > length:
        return audio[:length].astype(np.float32, copy=False)
    return np.pad(audio, (0, length - len(audio))).astype(np.float32, copy=False)


def perturb_librosa(audio: np.ndarray, sr: int, *, formant_factor: float, pitch_steps: float) -> np.ndarray:
    import librosa

    original_len = int(len(audio))
    if original_len <= 0:
        return audio
    target_sr = max(1000, int(round(sr * float(formant_factor))))
    warped = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    if len(warped) > 8:
        rate = max(0.05, len(warped) / float(original_len))
        warped = librosa.effects.time_stretch(warped, rate=rate)
    warped = fix_length(np.asarray(warped, dtype=np.float32), original_len)
    shifted = librosa.effects.pitch_shift(warped, sr=sr, n_steps=float(pitch_steps))
    shifted = fix_length(np.asarray(shifted, dtype=np.float32), original_len)
    src_rms = float(np.sqrt(np.mean(np.square(audio))) + 1.0e-8)
    dst_rms = float(np.sqrt(np.mean(np.square(shifted))) + 1.0e-8)
    shifted = shifted * (src_rms / dst_rms)
    peak = float(np.max(np.abs(shifted))) if shifted.size else 0.0
    if peak > 0.98:
        shifted = shifted * (0.98 / peak)
    return shifted.astype(np.float32, copy=False)


def has_parselmouth() -> bool:
    try:
        import parselmouth  # noqa: F401
    except Exception:
        return False
    return True


def median_pitch_hz(sound: Any, *, floor_hz: float = 50.0, ceiling_hz: float = 800.0) -> float:
    import parselmouth

    pitch = sound.to_pitch(time_step=None, pitch_floor=floor_hz, pitch_ceiling=ceiling_hz)
    values = np.asarray(pitch.selected_array["frequency"], dtype=np.float32)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def perturb_parselmouth(audio: np.ndarray, sr: int, *, formant_factor: float, pitch_steps: float) -> np.ndarray:
    import parselmouth
    from parselmouth.praat import call

    original_len = int(len(audio))
    if original_len <= 0:
        return audio
    sound = parselmouth.Sound(audio.astype(np.float64, copy=False), sampling_frequency=int(sr))
    old_median = median_pitch_hz(sound)
    pitch_factor = float(2.0 ** (float(pitch_steps) / 12.0))
    new_median = old_median * pitch_factor if old_median > 0 else 0.0
    changed = call(
        sound,
        "Change gender",
        50.0,
        800.0,
        float(formant_factor),
        float(new_median),
        1.0,
        1.0,
    )
    shifted = np.asarray(changed.values[0], dtype=np.float32)
    shifted = fix_length(shifted, original_len)
    src_rms = float(np.sqrt(np.mean(np.square(audio))) + 1.0e-8)
    dst_rms = float(np.sqrt(np.mean(np.square(shifted))) + 1.0e-8)
    shifted = shifted * (src_rms / dst_rms)
    peak = float(np.max(np.abs(shifted))) if shifted.size else 0.0
    if peak > 0.98:
        shifted = shifted * (0.98 / peak)
    return shifted.astype(np.float32, copy=False)


def resolve_backend(name: str) -> str:
    backend = str(name or "auto").strip().lower()
    if backend not in {"auto", "parselmouth", "librosa"}:
        raise ValueError(f"unsupported backend: {name!r}")
    if backend == "auto":
        return "parselmouth" if has_parselmouth() else "librosa"
    if backend == "parselmouth" and not has_parselmouth():
        raise RuntimeError("backend=parselmouth requested but praat-parselmouth is not installed")
    return backend


def perturb_audio(audio: np.ndarray, sr: int, *, backend: str, formant_factor: float, pitch_steps: float) -> np.ndarray:
    if backend == "parselmouth":
        return perturb_parselmouth(audio, sr, formant_factor=formant_factor, pitch_steps=pitch_steps)
    if backend == "librosa":
        return perturb_librosa(audio, sr, formant_factor=formant_factor, pitch_steps=pitch_steps)
    raise ValueError(f"unsupported resolved backend: {backend!r}")


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate perturbed source prompt views for Ver2.8 timbre repair B1.")
    ap.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--asr-input-jsonl", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--perturbed-audio-dir", required=True)
    ap.add_argument("--codes-dir", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--max-rows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--audio-sr", type=int, default=24000)
    ap.add_argument("--formant-min", type=float, default=0.82)
    ap.add_argument("--formant-max", type=float, default=1.20)
    ap.add_argument("--pitch-min-semitones", type=float, default=2.0)
    ap.add_argument("--pitch-max-semitones", type=float, default=4.0)
    ap.add_argument(
        "--backend",
        choices=("auto", "parselmouth", "librosa"),
        default="auto",
        help="auto prefers praat-parselmouth; librosa is a local fallback and should not be used for full B1 without ASR validation.",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-vq", type=int, default=None)
    ap.add_argument("--inline-codes", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--speaker-device", default="cuda:0")
    ap.add_argument("--speaker-sim-root", default=str(DEFAULT_SPEAKER_SIM_ROOT))
    ap.add_argument("--skip-speaker-sim", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    cfg = load_config(args.config)
    moss_root = deep_get(cfg, "moss.root")
    codec_path = deep_get(cfg, "moss.codec_path")
    n_vq = args.n_vq or int(deep_get(cfg, "moss.default_n_vq", 32))
    codec = MossCodec(codec_path, moss_root=moss_root, device=args.device, dtype=args.dtype)
    scorer = None if args.skip_speaker_sim else SpeakerScorer(Path(args.speaker_sim_root), args.speaker_device)
    backend = resolve_backend(str(args.backend))
    print(f"[perturb-source] backend={backend} requested={args.backend}", flush=True)

    out_rows: list[dict[str, Any]] = []
    asr_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    start = time.time()
    failures = 0
    for idx, row in enumerate(iter_jsonl(Path(args.input_jsonl))):
        if args.max_rows > 0 and len(out_rows) >= args.max_rows:
            break
        sample_id = str(row.get("sample_id") or f"row{idx:08d}")
        source_audio = record_get(row, "source_audio")
        if not source_audio or not Path(str(source_audio)).exists():
            failures += 1
            continue
        sign = -1.0 if rng.random() < 0.5 else 1.0
        formant_factor = rng.uniform(float(args.formant_min), float(args.formant_max))
        pitch_steps = sign * rng.uniform(float(args.pitch_min_semitones), float(args.pitch_max_semitones))
        digest = stable_id(sample_id, source_audio, args.seed, formant_factor, pitch_steps, length=16)
        audio_out = Path(args.perturbed_audio_dir) / f"{safe_stem(sample_id)}.{digest}.wav"
        codes_out = Path(args.codes_dir) / f"{safe_stem(sample_id)}.{digest}.pt"
        try:
            audio, sr = load_audio(str(source_audio), target_sr=int(args.audio_sr))
            perturbed = perturb_audio(
                audio,
                sr,
                backend=backend,
                formant_factor=formant_factor,
                pitch_steps=pitch_steps,
            )
            audio_out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(audio_out), perturbed, int(sr))
            if codes_out.exists():
                payload = load_torch_file(codes_out)
                codes = torch.as_tensor(payload["codes"], dtype=torch.long)
                reused = True
            else:
                result = codec.encode_path(str(audio_out), n_vq=int(n_vq))
                codes = torch.as_tensor(result["codes"], dtype=torch.long)
                payload = {
                    "codes": codes,
                    "audio_path": str(audio_out),
                    "source_audio": str(source_audio),
                    "num_frames": int(result["num_frames"]),
                    "n_vq": int(result["n_vq"]),
                    "duration_sec": float(result["duration_sec"]),
                    "sample_rate": int(result["sample_rate"]),
                    "formant_factor": float(formant_factor),
                    "pitch_steps": float(pitch_steps),
                }
                codes_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, codes_out)
                reused = False
            sim = scorer.similarity(audio_out, source_audio) if scorer is not None else None
            new_row = dict(row)
            new_row["source_prompt_perturbed_audio"] = str(audio_out)
            new_row["source_prompt_perturbed_codes_path"] = str(codes_out)
            if bool(args.inline_codes):
                new_row["source_prompt_perturbed_codes"] = codes.cpu().tolist()
            new_row["source_prompt_perturbation"] = {
                "backend": backend,
                "formant_factor": float(formant_factor),
                "pitch_steps": float(pitch_steps),
                "source_audio": str(source_audio),
                "sim_perturbed_source": sim,
                "codes_reused": reused,
            }
            out_rows.append(new_row)
            text = str(row.get("content_ref_text") or row.get("asr_src_text") or row.get("text") or "").strip()
            lang = str(row.get("language") or row.get("source_lang") or "")
            asr_rows.append(
                {
                    "sample_id": sample_id,
                    "case_id": sample_id,
                    "run_id": "source_prompt_perturb_b1_pilot",
                    "mode": "no_text",
                    "language": lang,
                    "source_lang": lang,
                    "source_audio": str(source_audio),
                    "target_audio": str(audio_out),
                    "text": text,
                    "target_text": text,
                    "content_ref_text": text,
                    "asr_src_text": text,
                    "perturb_formant_factor": float(formant_factor),
                    "perturb_pitch_steps": float(pitch_steps),
                    "sim_perturbed_source": sim,
                }
            )
            csv_rows.append(
                {
                    "sample_id": sample_id,
                    "source_audio": str(source_audio),
                    "perturbed_audio": str(audio_out),
                    "codes_path": str(codes_out),
                    "source_sec": len(audio) / float(sr),
                    "perturbed_sec": len(perturbed) / float(sr),
                    "codec_frames": int(codes.shape[0]),
                    "formant_factor": float(formant_factor),
                    "pitch_steps": float(pitch_steps),
                    "sim_perturbed_source": sim,
                    "codes_reused": reused,
                }
            )
        except Exception as exc:
            failures += 1
            print(f"[perturb-source] failed idx={idx} sample_id={sample_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if len(out_rows) and len(out_rows) % 25 == 0:
            elapsed = time.time() - start
            print(f"[perturb-source] processed={len(out_rows)} failures={failures} elapsed={elapsed:.1f}s", flush=True)

    write_jsonl(Path(args.output_jsonl), out_rows)
    write_jsonl(Path(args.asr_input_jsonl), asr_rows)
    sim_values = [r["sim_perturbed_source"] for r in csv_rows if r.get("sim_perturbed_source") is not None]
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "asr_input_jsonl": str(args.asr_input_jsonl),
        "rows": len(out_rows),
        "failures": failures,
        "elapsed_sec": round(time.time() - start, 3),
        "mean_sec_per_row": (time.time() - start) / max(1, len(out_rows)),
        "speaker_sim_perturbed_source_mean": sum(sim_values) / len(sim_values) if sim_values else None,
        "speaker_sim_perturbed_source_lt_0_4_rate": (
            sum(1 for v in sim_values if float(v) < 0.4) / len(sim_values) if sim_values else None
        ),
        "backend": backend,
        "requested_backend": str(args.backend),
        "note": (
            "backend=auto prefers praat-parselmouth. The librosa backend is a local pilot fallback "
            "and must pass ASR validation before any full run."
        ),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.summary_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "source_audio",
                "perturbed_audio",
                "codes_path",
                "source_sec",
                "perturbed_sec",
                "codec_frames",
                "formant_factor",
                "pitch_steps",
                "sim_perturbed_source",
                "codes_reused",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"[perturb-source] wrote rows={len(out_rows)} failures={failures} summary={args.summary_json}")
    return 0 if out_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
