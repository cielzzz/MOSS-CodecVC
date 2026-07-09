#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FFMPEG = "/opt/conda/envs/speech/bin/ffmpeg"
MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*([-+]?\d+(?:\.\d+)?)\s*dB")


def stable_id(*values: Any, length: int = 16) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_ref_audio(row: dict[str, Any]) -> str:
    meta = row.get("meta")
    if isinstance(meta, dict):
        u2 = meta.get("u2")
        if isinstance(u2, dict) and u2.get("audio"):
            return str(u2.get("audio") or "")
    for key in ("timbre_ref_audio", "u2_timbre_ref_audio_path"):
        if row.get(key):
            return str(row.get(key) or "")
    return ""


def read_dataset(row: dict[str, Any], ref_audio: str) -> str:
    value = row.get("dataset_name")
    if not value:
        meta = row.get("meta")
        if isinstance(meta, dict):
            source_fields = meta.get("source_fields")
            if isinstance(source_fields, dict):
                value = source_fields.get("dataset_name")
    text = str(value or "unknown_dataset")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or stable_id(ref_audio)


def row_is_same_episode_risk(row: dict[str, Any]) -> bool:
    risk = row.get("channel_shortcut_risk")
    if isinstance(risk, dict) and "same_long_audio_channel_risk" in risk:
        return bool(risk.get("same_long_audio_channel_risk"))
    if "same_long_audio_channel_risk" in row:
        return bool(row.get("same_long_audio_channel_risk"))

    u1_episode = ""
    u2_episode = ""
    if isinstance(risk, dict):
        u1_episode = str(risk.get("u1_episode_id") or "")
        u2_episode = str(risk.get("u2_episode_id") or "")
    u1_episode = u1_episode or str(row.get("u1_episode_id") or "")
    u2_episode = u2_episode or str(row.get("u2_episode_id") or "")
    return bool(u1_episode and u2_episode and u1_episode == u2_episode)


def should_augment(row: dict[str, Any], risk_mode: str) -> bool:
    if risk_mode == "all":
        return True
    if risk_mode == "same_episode":
        return row_is_same_episode_risk(row)
    if risk_mode == "none":
        return False
    raise ValueError(f"unsupported risk mode: {risk_mode}")


def fraction_selected(row: dict[str, Any], ref_audio: str, *, fraction: float, seed: str) -> bool:
    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    digest = stable_id(row.get("sample_id") or "", ref_audio, seed, "sample_fraction", length=16)
    value = int(digest, 16) / float(16**16 - 1)
    return value < fraction


def output_audio_path(output_root: Path, row: dict[str, Any], ref_audio: str, extension: str) -> Path:
    sample = str(row.get("sample_id") or "")
    dataset = read_dataset(row, ref_audio)
    digest = stable_id(sample, ref_audio, "ref_channel_aug", length=20)
    prefix = digest[:2]
    safe_sample = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample[-64:]) or digest
    return output_root / dataset / prefix / f"{safe_sample}_{digest}{extension}"


def choose_profile(rng: random.Random) -> str:
    value = rng.random()
    if value < 0.65:
        return "near_flat"
    if value < 0.87:
        return "mild_eq"
    if value < 0.94:
        return "room_eq"
    if value < 0.985:
        return "codec_eq"
    return "phone_band"


def treatment_for_profile(profile: str) -> str:
    if profile == "near_flat":
        return "near_flat"
    return f"perturbed_{profile}"


def build_filter(seed: str, *, target_sample_rate: int) -> tuple[str, str, dict[str, Any]]:
    rng = random.Random(seed)
    profile = choose_profile(rng)
    severity = rng.random()

    if profile == "phone_band":
        highpass = rng.uniform(120.0, 190.0)
        lowpass = rng.uniform(3300.0, 4300.0)
        echo_delay = rng.randint(8, 24)
        echo_decay = rng.uniform(0.015, 0.055)
        noise_weight = rng.uniform(0.025, 0.070)
        noise_amp = rng.uniform(0.0010, 0.0028)
    elif profile == "room_eq":
        highpass = rng.uniform(50.0, 105.0)
        lowpass = rng.uniform(5600.0, 7600.0)
        echo_delay = rng.randint(28, 85)
        echo_decay = rng.uniform(0.030, 0.110)
        noise_weight = rng.uniform(0.015, 0.050)
        noise_amp = rng.uniform(0.0007, 0.0020)
    elif profile == "codec_eq":
        highpass = rng.uniform(70.0, 130.0)
        lowpass = rng.uniform(5600.0, 8600.0)
        echo_delay = rng.randint(10, 35)
        echo_decay = rng.uniform(0.010, 0.045)
        noise_weight = rng.uniform(0.010, 0.040)
        noise_amp = rng.uniform(0.0006, 0.0018)
    elif profile == "near_flat":
        highpass = rng.uniform(18.0, 45.0)
        lowpass = rng.uniform(10300.0, min(11800.0, target_sample_rate / 2 - 150.0))
        echo_delay = rng.randint(7, 26)
        echo_decay = rng.uniform(0.0015, 0.0110)
        noise_weight = rng.uniform(0.0010, 0.0060)
        noise_amp = rng.uniform(0.00005, 0.00025)
    else:
        highpass = rng.uniform(35.0, 90.0)
        lowpass = rng.uniform(7800.0, min(11200.0, target_sample_rate / 2 - 250.0))
        echo_delay = rng.randint(10, 45)
        echo_decay = rng.uniform(0.008, 0.040)
        noise_weight = rng.uniform(0.006, 0.025)
        noise_amp = rng.uniform(0.0004, 0.0012)

    if profile == "near_flat":
        eq1_gain = rng.uniform(-0.55, 0.55)
        eq2_gain = rng.uniform(-0.45, 0.45)
        eq3_gain = rng.uniform(-0.60, 0.60)
        comp_threshold = rng.uniform(0.100, 0.190)
        comp_ratio = rng.uniform(1.03, 1.18)
        intermediate_sr = rng.randint(max(8000, target_sample_rate - 1400), target_sample_rate + 1400)
    else:
        eq1_gain = rng.uniform(-3.0, 2.5)
        eq2_gain = rng.uniform(-2.5, 3.0)
        eq3_gain = rng.uniform(-3.5, 2.0)
        comp_threshold = rng.uniform(0.055, 0.120)
        comp_ratio = rng.uniform(1.5, 2.6)
        intermediate_sr = target_sample_rate
    noise_color = rng.choice(["white", "pink", "brown"])

    speech_filters = [
        f"aresample={intermediate_sr}",
        f"aresample={target_sample_rate}",
        "aformat=channel_layouts=mono",
        f"highpass=f={highpass:.1f}",
        f"lowpass=f={lowpass:.1f}",
        f"equalizer=f=240:t=q:w=0.9:g={eq1_gain:.2f}",
        f"equalizer=f=1100:t=q:w=1.1:g={eq2_gain:.2f}",
        f"equalizer=f=3300:t=q:w=1.0:g={eq3_gain:.2f}",
        f"acompressor=threshold={comp_threshold:.4f}:ratio={comp_ratio:.2f}:attack=8:release=80",
        f"aecho=0.80:0.88:{echo_delay}:{echo_decay:.4f}",
    ]
    speech = ",".join(speech_filters)
    graph = (
        f"[0:a]{speech}[speech];"
        f"anoisesrc=color={noise_color}:amplitude={noise_amp:.6f}:duration=3600,"
        f"aresample={target_sample_rate},aformat=channel_layouts=mono[noise];"
        f"[speech][noise]amix=inputs=2:duration=first:weights=1 {noise_weight:.4f},"
        "alimiter=limit=0.95[out]"
    )
    profile_meta = {
        "profile": profile,
        "treatment": treatment_for_profile(profile),
        "severity": round(severity, 6),
        "target_sample_rate": int(target_sample_rate),
        "intermediate_sample_rate": int(intermediate_sr),
        "highpass_hz": round(highpass, 3),
        "lowpass_hz": round(lowpass, 3),
        "eq_gains_db": [round(eq1_gain, 4), round(eq2_gain, 4), round(eq3_gain, 4)],
        "compressor_threshold": round(comp_threshold, 6),
        "compressor_ratio": round(comp_ratio, 4),
        "echo_delay_ms": int(echo_delay),
        "echo_decay": round(echo_decay, 6),
        "noise_color": noise_color,
        "noise_weight": round(noise_weight, 6),
        "noise_amp": round(noise_amp, 8),
    }
    return profile, graph, profile_meta


def codec_args_for_extension(extension: str) -> list[str]:
    if extension == ".flac":
        return ["-c:a", "flac", "-compression_level", "5"]
    return ["-c:a", "pcm_s16le"]


def measure_mean_volume_db(ffmpeg: str, audio: Path | str) -> float | None:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return None
    matches = MEAN_VOLUME_RE.findall(proc.stderr)
    if not matches:
        return None
    return float(matches[-1])


def apply_gain_db(
    *,
    ffmpeg: str,
    input_audio: Path,
    output_audio: Path,
    gain_db: float,
    extension: str,
    target_sample_rate: int,
) -> tuple[bool, str]:
    tmp = output_audio.with_name(output_audio.name + ".loudness.tmp" + extension)
    if tmp.exists():
        tmp.unlink()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_audio),
        "-af",
        f"volume={gain_db:.4f}dB,alimiter=limit=0.95",
        "-ar",
        str(target_sample_rate),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        *codec_args_for_extension(extension),
        str(tmp),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return False, proc.stderr.strip()[-2000:] or f"ffmpeg gain failed with {proc.returncode}"
    tmp.replace(output_audio)
    return True, ""


def augment_one(
    *,
    ffmpeg: str,
    input_audio: str,
    output_audio: Path,
    seed: str,
    extension: str,
    overwrite_audio: bool,
    min_bytes: int,
    loudness_match: str,
    loudness_match_clamp_db: float,
    target_sample_rate: int,
) -> dict[str, Any]:
    input_path = Path(input_audio)
    if not input_path.exists():
        return {"ok": False, "error": f"missing input audio: {input_audio}"}
    if output_audio.exists() and output_audio.stat().st_size >= min_bytes and not overwrite_audio:
        profile, _, profile_meta = build_filter(seed, target_sample_rate=target_sample_rate)
        return {
            "ok": True,
            "audio": str(output_audio),
            "profile": profile,
            "profile_meta": profile_meta,
            "skipped_existing": True,
        }

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_audio.with_name(output_audio.name + ".tmp" + extension)
    if tmp.exists():
        tmp.unlink()

    profile, graph, profile_meta = build_filter(seed, target_sample_rate=target_sample_rate)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_audio,
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-ar",
        str(target_sample_rate),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        *codec_args_for_extension(extension),
        str(tmp),
    ]
    started = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.time() - started
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return {
            "ok": False,
            "error": proc.stderr.strip()[-2000:] or f"ffmpeg failed with {proc.returncode}",
            "profile": profile,
            "profile_meta": profile_meta,
        }
    if not tmp.exists() or tmp.stat().st_size < min_bytes:
        if tmp.exists():
            tmp.unlink()
        return {"ok": False, "error": f"augmented audio too small: {tmp}", "profile": profile, "profile_meta": profile_meta}
    loudness_meta: dict[str, Any] = {"mode": loudness_match}
    if loudness_match == "mean_volume":
        original_mean = measure_mean_volume_db(ffmpeg, input_path)
        augmented_mean = measure_mean_volume_db(ffmpeg, tmp)
        loudness_meta["original_mean_volume_db"] = original_mean
        loudness_meta["pre_match_augmented_mean_volume_db"] = augmented_mean
        if original_mean is not None and augmented_mean is not None:
            raw_gain = original_mean - augmented_mean
            gain = max(-loudness_match_clamp_db, min(loudness_match_clamp_db, raw_gain))
            ok, error = apply_gain_db(
                ffmpeg=ffmpeg,
                input_audio=tmp,
                output_audio=output_audio,
                gain_db=gain,
                extension=extension,
                target_sample_rate=target_sample_rate,
            )
            if not ok:
                if tmp.exists():
                    tmp.unlink()
                return {"ok": False, "error": error, "profile": profile, "profile_meta": profile_meta}
            loudness_meta["gain_db"] = round(gain, 4)
            loudness_meta["raw_gain_db"] = round(raw_gain, 4)
            loudness_meta["clamp_db"] = loudness_match_clamp_db
            post_mean = measure_mean_volume_db(ffmpeg, output_audio)
            loudness_meta["post_match_augmented_mean_volume_db"] = post_mean
            if tmp.exists():
                tmp.unlink()
        else:
            tmp.replace(output_audio)
            loudness_meta["warning"] = "mean_volume_measurement_failed"
    else:
        tmp.replace(output_audio)
    return {
        "ok": True,
        "audio": str(output_audio),
        "profile": profile,
        "profile_meta": profile_meta,
        "elapsed_sec": round(elapsed, 3),
        "skipped_existing": False,
        "loudness": loudness_meta,
    }


def set_nested_u2(row: dict[str, Any], aug_audio: str, original_audio: str, aug_meta: dict[str, Any]) -> None:
    row["timbre_ref_audio_original"] = original_audio
    row["timbre_ref_audio"] = aug_audio
    row["timbre_ref_channel_augmented"] = True
    row["timbre_ref_channel_augmentation"] = aug_meta
    row["ref_channel_treatment"] = aug_meta.get("treatment")
    row["ref_channel_target_sample_rate"] = aug_meta.get("target_sample_rate")
    row["ref_channel_profile"] = aug_meta.get("profile")
    row["ref_channel_seed"] = aug_meta.get("seed")
    row["ref_channel_severity"] = aug_meta.get("severity")

    if "u2_timbre_ref_audio_path" in row:
        row["u2_timbre_ref_audio_path_original"] = original_audio
        row["u2_timbre_ref_audio_path"] = aug_audio

    risk = row.get("channel_shortcut_risk")
    if isinstance(risk, dict):
        applied = risk.setdefault("applied_mitigations", {})
        if isinstance(applied, dict):
            applied["ref_side_channel_augmentation"] = True
        risk["ref_side_channel_augmentation"] = aug_meta

    meta = row.get("meta")
    if isinstance(meta, dict):
        u2 = meta.get("u2")
        if isinstance(u2, dict):
            u2["original_audio"] = original_audio
            u2["audio"] = aug_audio
            u2["channel_augmented_audio"] = aug_audio
            u2["channel_augmentation"] = aug_meta
            u2["ref_channel_treatment"] = aug_meta.get("treatment")
            u2["ref_channel_target_sample_rate"] = aug_meta.get("target_sample_rate")
            u2["ref_channel_profile"] = aug_meta.get("profile")
            u2["ref_channel_seed"] = aug_meta.get("seed")
            u2["ref_channel_severity"] = aug_meta.get("severity")


def apply_noop_meta(row: dict[str, Any], reason: str) -> None:
    row["timbre_ref_channel_augmented"] = False
    row["timbre_ref_channel_augmentation_skip_reason"] = reason


def process_chunk(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
    stats: Counter,
) -> list[dict[str, Any]]:
    futures: dict[Any, tuple[int, str, Path, str]] = {}
    selected: dict[int, tuple[str, Path, str]] = {}
    for idx, row in enumerate(rows):
        stats["rows"] += 1
        ref_audio = read_ref_audio(row)
        if not ref_audio:
            stats["skip_missing_ref_audio_field"] += 1
            apply_noop_meta(row, "missing_ref_audio_field")
            continue
        if not should_augment(row, args.risk_mode):
            stats["skip_risk_mode"] += 1
            apply_noop_meta(row, f"risk_mode_{args.risk_mode}")
            continue
        if not fraction_selected(row, ref_audio, fraction=args.sample_fraction, seed=args.sample_seed):
            stats["skip_sample_fraction"] += 1
            apply_noop_meta(row, f"sample_fraction_{args.sample_fraction:g}")
            continue
        sample_seed = stable_id(row.get("sample_id") or idx, ref_audio, args.seed)
        out_audio = output_audio_path(Path(args.audio_output_root), row, ref_audio, args.audio_extension)
        selected[idx] = (ref_audio, out_audio, sample_seed)
        stats["selected_for_augmentation"] += 1
        if args.dry_run:
            continue
        future = executor.submit(
            augment_one,
            ffmpeg=args.ffmpeg,
            input_audio=ref_audio,
            output_audio=out_audio,
            seed=sample_seed,
            extension=args.audio_extension,
            overwrite_audio=args.overwrite_audio,
            min_bytes=args.min_audio_bytes,
            loudness_match=args.loudness_match,
            loudness_match_clamp_db=args.loudness_match_clamp_db,
            target_sample_rate=args.target_sample_rate,
        )
        futures[future] = (idx, ref_audio, out_audio, sample_seed)

    if args.dry_run:
        for idx, (ref_audio, out_audio, sample_seed) in selected.items():
            profile, _, profile_meta = build_filter(sample_seed, target_sample_rate=args.target_sample_rate)
            set_nested_u2(
                rows[idx],
                str(out_audio),
                ref_audio,
                {
                    "enabled": True,
                    "dry_run": True,
                    **profile_meta,
                    "seed": sample_seed,
                    "tool": "ffmpeg",
                    "audio_extension": args.audio_extension,
                    "output_sample_rate": args.target_sample_rate,
                    "output_channels": 1,
                    "output_sample_format": "s16",
                },
            )
            stats["dry_run_rows"] += 1
        return rows

    for future in as_completed(futures):
        idx, ref_audio, out_audio, sample_seed = futures[future]
        result = future.result()
        if not result.get("ok"):
            stats["augmentation_failed"] += 1
            if args.fail_fast:
                raise SystemExit(f"augmentation failed for {ref_audio}: {result.get('error')}")
            apply_noop_meta(rows[idx], str(result.get("error") or "augmentation_failed"))
            continue
        stats["augmented_rows"] += 1
        if result.get("skipped_existing"):
            stats["reused_existing_audio"] += 1
        profile = str(result.get("profile") or "")
        profile_meta = result.get("profile_meta")
        if not isinstance(profile_meta, dict):
            profile_meta = {
                "profile": profile,
                "treatment": treatment_for_profile(profile),
                "target_sample_rate": args.target_sample_rate,
            }
        if profile:
            stats[f"profile_{profile}"] += 1
        set_nested_u2(
            rows[idx],
            str(out_audio),
            ref_audio,
            {
                "enabled": True,
                **profile_meta,
                "seed": sample_seed,
                "tool": "ffmpeg",
                "audio_extension": args.audio_extension,
                "output_sample_rate": args.target_sample_rate,
                "output_channels": 1,
                "output_sample_format": "s16",
                "skipped_existing": bool(result.get("skipped_existing")),
                "loudness_match": result.get("loudness") or {"mode": args.loudness_match},
            },
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply deterministic ref-side channel augmentation to V2 VC manifests. "
            "This breaks same-recording channel shortcuts by replacing timbre_ref_audio/u2 "
            "with an augmented local file while keeping the original path in metadata."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--audio-output-root", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--ffmpeg", default=DEFAULT_FFMPEG)
    parser.add_argument("--risk-mode", choices=["same_episode", "all", "none"], default="same_episode")
    parser.add_argument("--audio-extension", choices=[".wav", ".flac"], default=".flac")
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-audio-bytes", type=int, default=4096)
    parser.add_argument("--seed", default="v2_ref_channel_aug_20260707")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Deterministically augment this fraction of eligible rows.")
    parser.add_argument("--sample-seed", default="v2_ref_channel_aug_sample_fraction_20260707")
    parser.add_argument("--loudness-match", choices=["mean_volume", "none"], default="mean_volume")
    parser.add_argument("--loudness-match-clamp-db", type=float, default=6.0)
    parser.add_argument("--overwrite-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve(strict=False)
    output_jsonl = Path(args.output_jsonl).expanduser().resolve(strict=False)
    audio_output_root = Path(args.audio_output_root).expanduser().resolve(strict=False)
    args.audio_output_root = str(audio_output_root)

    if not input_jsonl.exists():
        raise SystemExit(f"input JSONL not found: {input_jsonl}")
    if output_jsonl.exists() and not args.overwrite:
        raise SystemExit(f"output exists, pass --overwrite to replace: {output_jsonl}")
    if not args.dry_run and not Path(args.ffmpeg).exists():
        raise SystemExit(f"ffmpeg not found: {args.ffmpeg}")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")
    if args.target_sample_rate < 24000:
        raise SystemExit("--target-sample-rate must be >= 24000 for the v2 refdecorr profiles")
    if not 0.0 <= args.sample_fraction <= 1.0:
        raise SystemExit("--sample-fraction must be in [0, 1]")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audio_output_root.mkdir(parents=True, exist_ok=True)
    tmp = output_jsonl.with_name(output_jsonl.name + ".tmp")
    stats: Counter = Counter()
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.jobs) as executor, tmp.open("w", encoding="utf-8") as out:
        chunk: list[dict[str, Any]] = []
        for row in iter_jsonl(input_jsonl):
            chunk.append(row)
            if args.max_rows > 0 and stats["rows"] + len(chunk) >= args.max_rows:
                keep = args.max_rows - stats["rows"]
                if keep > 0:
                    chunk = chunk[:keep]
                    for item in process_chunk(chunk, args=args, executor=executor, stats=stats):
                        out.write(json.dumps(item, ensure_ascii=False) + "\n")
                chunk = []
                break
            if len(chunk) >= args.chunk_size:
                for item in process_chunk(chunk, args=args, executor=executor, stats=stats):
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                chunk = []
        if chunk:
            for item in process_chunk(chunk, args=args, executor=executor, stats=stats):
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

    tmp.replace(output_jsonl)
    summary = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "audio_output_root": str(audio_output_root),
        "risk_mode": args.risk_mode,
        "audio_extension": args.audio_extension,
        "target_sample_rate": args.target_sample_rate,
        "profile_distribution": {
            "near_flat": 0.65,
            "mild_eq": 0.22,
            "room_eq": 0.07,
            "codec_eq": 0.045,
            "phone_band": 0.015,
        },
        "jobs": args.jobs,
        "dry_run": bool(args.dry_run),
        "elapsed_sec": round(time.time() - started, 3),
        "stats": dict(stats),
        "training_note": (
            "Use the output JSONL for training. timbre_ref_audio/u2 points to augmented audio; "
            "timbre_ref_audio_original and meta.u2.original_audio keep the unmodified reference."
        ),
    }
    summary_path = Path(args.summary_json).expanduser().resolve(strict=False) if args.summary_json else output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
