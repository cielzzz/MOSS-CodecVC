#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FFMPEG = "/opt/conda/envs/speech/bin/ffmpeg"


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


def safe_name(text: str, fallback: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return value or fallback


def read_original_u2(row: dict[str, Any]) -> str:
    meta = row.get("meta")
    if isinstance(meta, dict):
        u2 = meta.get("u2")
        if isinstance(u2, dict):
            for key in ("original_audio", "source_audio", "audio"):
                value = u2.get(key)
                if value and key != "audio":
                    return str(value)
            if u2.get("audio") and not str(u2.get("audio")).endswith(".flac"):
                return str(u2.get("audio"))
    for key in ("timbre_ref_audio_original", "u2_timbre_ref_audio_path_original", "timbre_ref_audio", "u2_timbre_ref_audio_path"):
        if row.get(key):
            return str(row.get(key))
    return ""


def dataset_name(row: dict[str, Any], ref_audio: str) -> str:
    meta = row.get("meta")
    value = row.get("dataset_name")
    if not value and isinstance(meta, dict):
        source_fields = meta.get("source_fields")
        if isinstance(source_fields, dict):
            value = source_fields.get("dataset_name")
    return safe_name(str(value or "unknown_dataset"), stable_id(ref_audio))


def output_audio_path(output_root: Path, row: dict[str, Any], original_ref: str, split: str, extension: str) -> Path:
    sample = str(row.get("sample_id") or stable_id(original_ref))
    digest = stable_id(split, sample, original_ref, length=20)
    return output_root / dataset_name(row, original_ref) / digest[:2] / f"{safe_name(sample[-64:], digest)}_{digest}{extension}"


def choose_heldout_profile(rng: random.Random) -> str:
    value = rng.random()
    if value < 0.45:
        return "valid_heldout_near_flat"
    if value < 0.75:
        return "valid_heldout_device_eq"
    if value < 0.95:
        return "valid_heldout_room"
    return "valid_heldout_codec"


def heldout_filter(seed: str, *, target_sample_rate: int) -> tuple[str, str, dict[str, Any]]:
    rng = random.Random(seed)
    profile = choose_heldout_profile(rng)
    severity = rng.random()

    if profile == "valid_heldout_near_flat":
        highpass = rng.uniform(22.0, 55.0)
        lowpass = rng.uniform(10100.0, min(11750.0, target_sample_rate / 2 - 180.0))
        echo_delay = rng.randint(6, 22)
        echo_decay = rng.uniform(0.0010, 0.0090)
        noise_weight = rng.uniform(0.0008, 0.0045)
        noise_amp = rng.uniform(0.00004, 0.00020)
        eq = [
            (180, rng.uniform(-0.45, 0.45), 0.75),
            (760, rng.uniform(-0.35, 0.35), 1.25),
            (2400, rng.uniform(-0.50, 0.50), 1.10),
            (6100, rng.uniform(-0.45, 0.45), 0.95),
        ]
        comp_threshold = rng.uniform(0.120, 0.210)
        comp_ratio = rng.uniform(1.02, 1.14)
        intermediate_sr = rng.randint(target_sample_rate - 1100, target_sample_rate + 1100)
    elif profile == "valid_heldout_device_eq":
        highpass = rng.uniform(45.0, 115.0)
        lowpass = rng.uniform(8200.0, min(11300.0, target_sample_rate / 2 - 250.0))
        echo_delay = rng.randint(12, 38)
        echo_decay = rng.uniform(0.0040, 0.0260)
        noise_weight = rng.uniform(0.0030, 0.0160)
        noise_amp = rng.uniform(0.00015, 0.00075)
        eq = [
            (160, rng.uniform(-1.6, 1.2), 0.80),
            (920, rng.uniform(-1.3, 1.4), 1.30),
            (2850, rng.uniform(-1.6, 1.3), 1.05),
            (6900, rng.uniform(-1.2, 1.4), 0.85),
        ]
        comp_threshold = rng.uniform(0.080, 0.160)
        comp_ratio = rng.uniform(1.12, 1.55)
        intermediate_sr = target_sample_rate
    elif profile == "valid_heldout_room":
        highpass = rng.uniform(40.0, 100.0)
        lowpass = rng.uniform(7600.0, min(11000.0, target_sample_rate / 2 - 300.0))
        echo_delay = rng.randint(38, 125)
        echo_decay = rng.uniform(0.0180, 0.0900)
        noise_weight = rng.uniform(0.0040, 0.0220)
        noise_amp = rng.uniform(0.00020, 0.00100)
        eq = [
            (210, rng.uniform(-1.2, 1.5), 0.95),
            (1050, rng.uniform(-1.5, 1.7), 1.05),
            (3100, rng.uniform(-1.4, 1.6), 0.95),
            (7400, rng.uniform(-1.1, 1.2), 0.80),
        ]
        comp_threshold = rng.uniform(0.075, 0.150)
        comp_ratio = rng.uniform(1.10, 1.45)
        intermediate_sr = target_sample_rate
    else:
        highpass = rng.uniform(60.0, 140.0)
        lowpass = rng.uniform(6400.0, 9600.0)
        echo_delay = rng.randint(9, 31)
        echo_decay = rng.uniform(0.0050, 0.0300)
        noise_weight = rng.uniform(0.0040, 0.0200)
        noise_amp = rng.uniform(0.00018, 0.00085)
        eq = [
            (130, rng.uniform(-1.7, 1.1), 0.90),
            (720, rng.uniform(-1.4, 1.4), 1.10),
            (2200, rng.uniform(-1.7, 1.2), 1.15),
            (5400, rng.uniform(-1.6, 1.1), 0.90),
        ]
        comp_threshold = rng.uniform(0.060, 0.130)
        comp_ratio = rng.uniform(1.18, 1.75)
        intermediate_sr = rng.choice([22050, 23000, 25000, 26000])

    noise_color = rng.choice(["white", "pink", "brown"])
    filters = [
        f"aresample={intermediate_sr}",
        f"aresample={target_sample_rate}",
        "aformat=channel_layouts=mono",
        f"highpass=f={highpass:.1f}",
        f"lowpass=f={lowpass:.1f}",
    ]
    for freq, gain, width in eq:
        filters.append(f"equalizer=f={freq}:t=q:w={width:.2f}:g={gain:.2f}")
    filters.extend(
        [
            f"acompressor=threshold={comp_threshold:.4f}:ratio={comp_ratio:.2f}:attack=10:release=90",
            f"aecho=0.82:0.86:{echo_delay}:{echo_decay:.4f}",
        ]
    )
    graph = (
        f"[0:a]{','.join(filters)}[speech];"
        f"anoisesrc=color={noise_color}:amplitude={noise_amp:.6f}:duration=3600,"
        f"aresample={target_sample_rate},aformat=channel_layouts=mono[noise];"
        f"[speech][noise]amix=inputs=2:duration=first:weights=1 {noise_weight:.4f},"
        "alimiter=limit=0.95[out]"
    )
    meta = {
        "profile": profile,
        "treatment": "heldout_refdecorr_cross_channel",
        "severity": round(severity, 6),
        "target_sample_rate": int(target_sample_rate),
        "intermediate_sample_rate": int(intermediate_sr),
        "highpass_hz": round(highpass, 3),
        "lowpass_hz": round(lowpass, 3),
        "eq": [{"freq_hz": f, "gain_db": round(g, 4), "q_width": round(w, 4)} for f, g, w in eq],
        "echo_delay_ms": int(echo_delay),
        "echo_decay": round(echo_decay, 6),
        "noise_color": noise_color,
        "noise_weight": round(noise_weight, 6),
        "noise_amp": round(noise_amp, 8),
    }
    return profile, graph, meta


def codec_args(extension: str) -> list[str]:
    if extension == ".flac":
        return ["-c:a", "flac", "-compression_level", "5"]
    return ["-c:a", "pcm_s16le"]


def run_ffmpeg(
    *,
    ffmpeg: str,
    input_audio: str,
    output_audio: Path,
    extension: str,
    target_sample_rate: int,
    mode: str,
    seed: str,
    overwrite_audio: bool,
    min_audio_bytes: int,
) -> dict[str, Any]:
    if output_audio.exists() and output_audio.stat().st_size >= min_audio_bytes and not overwrite_audio:
        if mode == "heldout":
            profile, _, meta = heldout_filter(seed, target_sample_rate=target_sample_rate)
        else:
            profile = "valid_same_episode_near_original"
            meta = {
                "profile": profile,
                "treatment": "same_episode_near_original_24k",
                "severity": 0.0,
                "target_sample_rate": int(target_sample_rate),
            }
        return {"ok": True, "profile": profile, "profile_meta": meta, "skipped_existing": True}

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_audio.with_name(output_audio.name + ".tmp" + extension)
    if tmp.exists():
        tmp.unlink()

    if mode == "heldout":
        profile, graph, meta = heldout_filter(seed, target_sample_rate=target_sample_rate)
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
            *codec_args(extension),
            str(tmp),
        ]
    else:
        profile = "valid_same_episode_near_original"
        meta = {
            "profile": profile,
            "treatment": "same_episode_near_original_24k",
            "severity": 0.0,
            "target_sample_rate": int(target_sample_rate),
        }
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_audio,
            "-ar",
            str(target_sample_rate),
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            *codec_args(extension),
            str(tmp),
        ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return {"ok": False, "profile": profile, "profile_meta": meta, "error": proc.stderr.strip()[-2000:]}
    if not tmp.exists() or tmp.stat().st_size < min_audio_bytes:
        if tmp.exists():
            tmp.unlink()
        return {"ok": False, "profile": profile, "profile_meta": meta, "error": f"audio too small: {tmp}"}
    tmp.replace(output_audio)
    return {"ok": True, "profile": profile, "profile_meta": meta, "skipped_existing": False}


def set_ref(row: dict[str, Any], *, audio: str, original_audio: str, aug_meta: dict[str, Any], valid_name: str) -> dict[str, Any]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["validation_set"] = valid_name
    out["timbre_ref_audio_original"] = original_audio
    out["timbre_ref_audio"] = audio
    out["ref_channel_treatment"] = aug_meta.get("treatment")
    out["ref_channel_target_sample_rate"] = aug_meta.get("target_sample_rate")
    out["ref_channel_profile"] = aug_meta.get("profile")
    out["ref_channel_seed"] = aug_meta.get("seed")
    out["ref_channel_severity"] = aug_meta.get("severity")
    out["timbre_ref_channel_augmented"] = True
    out["timbre_ref_channel_augmentation"] = aug_meta
    risk = out.get("channel_shortcut_risk")
    if isinstance(risk, dict):
        risk["validation_set"] = valid_name
        risk["ref_side_channel_augmentation"] = aug_meta
    meta = out.get("meta")
    if isinstance(meta, dict):
        u2 = meta.get("u2")
        if isinstance(u2, dict):
            u2["original_audio"] = original_audio
            u2["audio"] = audio
            u2["channel_augmented_audio"] = audio
            u2["channel_augmentation"] = aug_meta
            u2["ref_channel_treatment"] = aug_meta.get("treatment")
            u2["ref_channel_target_sample_rate"] = aug_meta.get("target_sample_rate")
            u2["ref_channel_profile"] = aug_meta.get("profile")
            u2["ref_channel_seed"] = aug_meta.get("seed")
            u2["ref_channel_severity"] = aug_meta.get("severity")
    return out


def simple_row(manifest: dict[str, Any]) -> dict[str, Any]:
    meta = manifest["meta"]
    u1 = meta["u1"]
    u2 = meta["u2"]
    u1_prime = meta["u1_prime"]
    source_fields = meta["source_fields"]
    risk = manifest["channel_shortcut_risk"]
    text_policy = manifest["text_policy"]
    return {
        "sample_id": manifest["sample_id"],
        "validation_set": manifest.get("validation_set") or "",
        "mode": manifest.get("preferred_emit_mode") or "no_text",
        "ready": not bool(manifest.get("source_audio_pending")),
        "dataset_name": source_fields.get("dataset_name") or "",
        "language": manifest.get("language") or "",
        "u1_target_audio_path": u1.get("audio") or "",
        "u1_text": u1.get("text") or "",
        "u2_timbre_ref_audio_path": u2.get("audio") or "",
        "u2_timbre_ref_audio_path_original": u2.get("original_audio") or manifest.get("timbre_ref_audio_original") or "",
        "u2_text": u2.get("text") or "",
        "u1_prime_source_audio_path": u1_prime.get("audio") or "",
        "u1_prime_pending": bool(u1_prime.get("pending")),
        "source_audio": manifest.get("source_audio") or "",
        "timbre_ref_audio": manifest.get("timbre_ref_audio") or "",
        "timbre_ref_audio_original": manifest.get("timbre_ref_audio_original") or "",
        "target_audio": manifest.get("target_audio") or "",
        "target_text": manifest.get("target_text") or "",
        "text": manifest.get("text") or "",
        "label": source_fields.get("label") or "",
        "reference": source_fields.get("reference") or "",
        "segs": source_fields.get("segs"),
        "similarity": source_fields.get("similarity"),
        "label_idx": source_fields.get("label_idx"),
        "ref_idx": source_fields.get("ref_idx"),
        "same_long_audio_channel_risk": risk.get("same_long_audio_channel_risk"),
        "u1_episode_id": risk.get("u1_episode_id") or "",
        "u2_episode_id": risk.get("u2_episode_id") or "",
        "text_target_differs_from_timbre_text": text_policy.get("target_text_differs_from_timbre_text"),
        "text_target_differs_from_source_text": text_policy.get("target_text_differs_from_source_text"),
        "source_seedvc_job_id": manifest.get("source_generation_job_id") or "",
        "source_resultf_jsonl": source_fields.get("source_resultf_jsonl") or "",
        "source_prepare_jsonl": source_fields.get("source_prepare_jsonl") or "",
        "u1_target_audio_source_uri": u1.get("source_uri") or "",
        "u2_timbre_ref_audio_source_uri": u2.get("source_uri") or "",
        "lance_table_uri": source_fields.get("lance_table_uri") or "",
        "ref_channel_treatment": manifest.get("ref_channel_treatment"),
        "ref_channel_target_sample_rate": manifest.get("ref_channel_target_sample_rate"),
        "ref_channel_profile": manifest.get("ref_channel_profile"),
        "ref_channel_seed": manifest.get("ref_channel_seed"),
        "ref_channel_severity": manifest.get("ref_channel_severity"),
    }


def select_valid_rows(rows: list[dict[str, Any]], *, en_count: int, zh_count: int, seed: str) -> list[dict[str, Any]]:
    by_lang: dict[str, list[tuple[str, dict[str, Any]]]] = {"en": [], "zh": []}
    for row in rows:
        lang = str(row.get("language") or "")
        if lang not in by_lang:
            continue
        sample_id = str(row.get("sample_id") or "")
        rank = stable_id(seed, lang, sample_id, length=24)
        by_lang[lang].append((rank, row))
    selected: list[dict[str, Any]] = []
    for lang, count in (("en", en_count), ("zh", zh_count)):
        by_lang[lang].sort(key=lambda item: item[0])
        if len(by_lang[lang]) < count:
            raise SystemExit(f"not enough {lang} rows: {len(by_lang[lang])} < {count}")
        selected.extend(row for _, row in by_lang[lang][:count])
    selected.sort(key=lambda row: stable_id(seed, row.get("sample_id") or "", length=24))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build held-out ref-channel validation sets for V2 no-text refdecorr manifests.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-minus-valid-manifest", required=True)
    parser.add_argument("--train-minus-valid-simple", required=True)
    parser.add_argument("--valid-rows-per-language", type=int, default=1000)
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--audio-extension", choices=[".flac", ".wav"], default=".flac")
    parser.add_argument("--ffmpeg", default=DEFAULT_FFMPEG)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--seed", default="v2_refdecorr_valid_20260708")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-audio-bytes", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve(strict=False)
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    train_minus_manifest = Path(args.train_minus_valid_manifest).expanduser().resolve(strict=False)
    train_minus_simple = Path(args.train_minus_valid_simple).expanduser().resolve(strict=False)
    if not input_jsonl.exists():
        raise SystemExit(f"input not found: {input_jsonl}")
    if not Path(args.ffmpeg).exists():
        raise SystemExit(f"ffmpeg not found: {args.ffmpeg}")
    for path in [train_minus_manifest, train_minus_simple]:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite: {path}")

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = list(iter_jsonl(input_jsonl))
    valid_rows = select_valid_rows(
        rows,
        en_count=args.valid_rows_per_language,
        zh_count=args.valid_rows_per_language,
        seed=args.seed,
    )
    valid_ids = {str(row.get("sample_id") or "") for row in valid_rows}
    (output_root / "valid_sample_ids.txt").write_text("\n".join(sorted(valid_ids)) + "\n", encoding="utf-8")

    audio_same_root = output_root / "audio_same_episode_near_original_24k"
    audio_heldout_root = output_root / "audio_heldout_refdecorr_cross_channel"
    valid_specs = [
        ("same_episode_near_original_valid", "same", audio_same_root),
        ("heldout_refdecorr_cross_channel_valid", "heldout", audio_heldout_root),
    ]
    valid_outputs: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in valid_specs}
    stats: Counter = Counter()

    futures: dict[Any, tuple[str, dict[str, Any], str, Path, str]] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for row in valid_rows:
            original_ref = read_original_u2(row)
            if not original_ref:
                raise SystemExit(f"missing original u2 for {row.get('sample_id')}")
            for valid_name, mode, audio_root in valid_specs:
                seed = stable_id(args.seed, valid_name, row.get("sample_id") or "", original_ref)
                out_audio = output_audio_path(audio_root, row, original_ref, valid_name, args.audio_extension)
                future = executor.submit(
                    run_ffmpeg,
                    ffmpeg=args.ffmpeg,
                    input_audio=original_ref,
                    output_audio=out_audio,
                    extension=args.audio_extension,
                    target_sample_rate=args.target_sample_rate,
                    mode=mode,
                    seed=seed,
                    overwrite_audio=args.overwrite_audio,
                    min_audio_bytes=args.min_audio_bytes,
                )
                futures[future] = (valid_name, row, original_ref, out_audio, seed)
        for future in as_completed(futures):
            valid_name, row, original_ref, out_audio, seed = futures[future]
            result = future.result()
            if not result.get("ok"):
                raise SystemExit(f"ffmpeg failed for {valid_name} {original_ref}: {result.get('error')}")
            aug_meta = dict(result.get("profile_meta") or {})
            aug_meta.update(
                {
                    "enabled": True,
                    "seed": seed,
                    "tool": "ffmpeg",
                    "audio_extension": args.audio_extension,
                    "output_sample_rate": args.target_sample_rate,
                    "output_channels": 1,
                    "output_sample_format": "s16",
                    "validation_set": valid_name,
                    "heldout_from_training": True,
                    "skipped_existing": bool(result.get("skipped_existing")),
                }
            )
            profile = str(aug_meta.get("profile") or "")
            stats[f"{valid_name}_rows"] += 1
            stats[f"{valid_name}_profile_{profile}"] += 1
            valid_outputs[valid_name].append(
                set_ref(
                    row,
                    audio=str(out_audio),
                    original_audio=original_ref,
                    aug_meta=aug_meta,
                    valid_name=valid_name,
                )
            )

    for name in valid_outputs:
        valid_outputs[name].sort(key=lambda row: str(row.get("sample_id") or ""))

    train_minus_manifest.parent.mkdir(parents=True, exist_ok=True)
    train_minus_simple.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = train_minus_manifest.with_name(train_minus_manifest.name + ".tmp")
    tmp_simple = train_minus_simple.with_name(train_minus_simple.name + ".tmp")
    train_rows = 0
    with tmp_manifest.open("w", encoding="utf-8") as mf, tmp_simple.open("w", encoding="utf-8") as sf:
        for row in rows:
            if str(row.get("sample_id") or "") in valid_ids:
                continue
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            sf.write(json.dumps(simple_row(row), ensure_ascii=False) + "\n")
            train_rows += 1
    tmp_manifest.replace(train_minus_manifest)
    tmp_simple.replace(train_minus_simple)

    output_files: dict[str, str] = {
        "train_minus_valid_manifest": str(train_minus_manifest),
        "train_minus_valid_simple": str(train_minus_simple),
        "valid_sample_ids": str(output_root / "valid_sample_ids.txt"),
    }
    for valid_name, rows_for_valid in valid_outputs.items():
        manifest_path = output_root / f"{valid_name}.manifest.jsonl"
        simple_path = output_root / f"{valid_name}.simple.jsonl"
        with manifest_path.open("w", encoding="utf-8") as mf, simple_path.open("w", encoding="utf-8") as sf:
            for row in rows_for_valid:
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                sf.write(json.dumps(simple_row(row), ensure_ascii=False) + "\n")
        output_files[f"{valid_name}_manifest"] = str(manifest_path)
        output_files[f"{valid_name}_simple"] = str(simple_path)

    language_counts = Counter(str(row.get("language") or "") for row in valid_rows)
    summary = {
        "input_jsonl": str(input_jsonl),
        "output_root": str(output_root),
        "outputs": output_files,
        "train_minus_valid_rows": train_rows,
        "valid_rows": len(valid_rows),
        "valid_language_counts": dict(language_counts),
        "target_sample_rate": args.target_sample_rate,
        "audio_extension": args.audio_extension,
        "stats": dict(stats),
        "elapsed_sec": round(time.time() - started, 3),
        "concept": (
            "Held-out validation rows are excluded from train_minus_valid outputs. "
            "same_episode_near_original_valid keeps near-original same-episode ref channel as the shortcut-prone control; "
            "heldout_refdecorr_cross_channel_valid uses a ref transform family and seed namespace not used by training."
        ),
    }
    write_json(output_root / "summary.valid_ref_channel.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
