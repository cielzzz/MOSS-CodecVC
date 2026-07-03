#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import traceback
from importlib.resources import files
from pathlib import Path
from typing import Any

import soundfile as sf
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    speed,
    sway_sampling_coef,
    target_rms,
)


def patch_torchaudio_load_with_soundfile_fallback() -> None:
    import numpy as np
    import torch
    import torchaudio

    original_load = torchaudio.load

    def safe_load(path, *args, **kwargs):
        try:
            return original_load(path, *args, **kwargs)
        except RuntimeError as exc:
            msg = str(exc)
            if "torchcodec" not in msg and "FFmpeg" not in msg and "libtorchcodec" not in msg:
                raise
            wav, sr = sf.read(path, always_2d=True, dtype="float32")
            return torch.from_numpy(np.asarray(wav).T.copy()), int(sr)

    torchaudio.load = safe_load


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_checkpoint(model: str, vocoder_name: str, ckpt_file: str) -> tuple[str, str]:
    model_name = model
    repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
    if model_name == "F5TTS_Base":
        if vocoder_name == "vocos":
            ckpt_step = 1200000
        else:
            model_name = "F5TTS_Base_bigvgan"
            ckpt_type = "pt"
    elif model_name == "E2TTS_Base":
        repo_name = "E2-TTS"
        ckpt_step = 1200000
    if ckpt_file:
        return model_name, ckpt_file
    return model_name, str(cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"))


def build_output_path(output_root: Path, row: dict[str, Any]) -> Path:
    sample_id = str(row["sample_id"]).replace("moss_tts_", "f5_tts_")
    return output_root / "style_carrier" / "f5_tts" / str(row.get("language") or "unknown") / f"{sample_id}.wav"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate F5-TTS style carrier wavs from text style-clone plan.")
    parser.add_argument("--plan-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--languages", default="", help="Optional comma-separated language filter.")
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--vocoder-name", default=mel_spec_type, choices=["vocos", "bigvgan"])
    parser.add_argument("--ckpt-file", default="")
    parser.add_argument("--vocab-file", default="")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-report", action="store_true")
    parser.add_argument("--print-traceback", action="store_true")
    args = parser.parse_args()
    patch_torchaudio_load_with_soundfile_fallback()

    plan_path = Path(args.plan_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if args.overwrite_report and output_path.exists():
        output_path.unlink()
    languages = {x.strip() for x in args.languages.split(",") if x.strip()}
    rows = load_jsonl(plan_path)
    if languages:
        rows = [row for row in rows if str(row.get("language")) in languages]
    if args.limit > 0:
        rows = rows[: args.limit]

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{args.model}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    model_name, ckpt_path = resolve_checkpoint(args.model, args.vocoder_name, args.ckpt_file)
    print(f"[f5] rows={len(rows)} model={model_name} ckpt={ckpt_path} device={device}")
    vocoder = load_vocoder(vocoder_name=args.vocoder_name, is_local=False, local_path="", device=device)
    ema_model = load_model(
        model_cls,
        model_arc,
        ckpt_path,
        mel_spec_type=args.vocoder_name,
        vocab_file=args.vocab_file,
        device=device,
    )

    ok = 0
    failed = 0
    for idx, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"]).replace("moss_tts_", "f5_tts_")
        out_wav = build_output_path(output_root, row)
        result = {
            **row,
            "sample_id": sample_id,
            "tts_backend": "f5_tts",
            "style_carrier_wav": str(out_wav),
            "style_generation_backend": "f5_tts",
            "style_generation_status": "pending",
            "style_generation_error": "",
        }
        print(f"[f5] {idx}/{len(rows)} {sample_id}")
        try:
            if args.skip_existing and out_wav.exists() and out_wav.stat().st_size > 0:
                result["style_generation_status"] = "skipped_existing"
            else:
                ref_audio, ref_text = preprocess_ref_audio_text(
                    str(row["source_style_wav"]),
                    str(row.get("source_style_text") or row.get("source_text") or ""),
                )
                wave, sample_rate, spect = infer_process(
                    ref_audio,
                    ref_text,
                    str(row["input_text"]),
                    ema_model,
                    vocoder,
                    mel_spec_type=args.vocoder_name,
                    target_rms=target_rms,
                    cross_fade_duration=cross_fade_duration,
                    nfe_step=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    speed=speed,
                    fix_duration=fix_duration,
                    device=device,
                )
                out_wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out_wav, wave, sample_rate)
                result["style_generation_status"] = "ok"
                result["style_carrier_sample_rate"] = int(sample_rate)
                result["style_carrier_num_samples"] = int(len(wave))
                result["style_carrier_duration"] = float(len(wave) / sample_rate)
                result["style_generation_spect_shape"] = list(getattr(spect, "shape", []))
            ok += 1
        except Exception as exc:
            failed += 1
            result["style_generation_status"] = "failed"
            result["style_generation_error"] = f"{type(exc).__name__}: {exc}"
            if args.print_traceback:
                result["style_generation_traceback"] = traceback.format_exc()
            print(f"[f5][failed] {sample_id}: {result['style_generation_error']}")
        append_jsonl(output_path, result)
    print(json.dumps({"rows": len(rows), "ok_or_skipped": ok, "failed": failed, "output_jsonl": str(output_path)}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
