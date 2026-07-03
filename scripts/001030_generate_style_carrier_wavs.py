#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


DEFAULT_MOSS_CODE_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-TTS"
)
DEFAULT_MOSS_MODEL_PATH = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-TTS"
)
DEFAULT_CODEC_PATH = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-Audio-Tokenizer"
)

DEFAULT_STYLE_INSTRUCTION = (
    "Generate speech saying the provided text. Use [S1] as the reference for speaker voice, "
    "speaking style, rhythm, pauses, speaking rate, stress and duration hints."
)


def patch_torchaudio_load_with_soundfile_fallback() -> None:
    import torch
    import torchaudio

    original_load = torchaudio.load
    original_save = torchaudio.save

    def safe_load(path, *args, **kwargs):
        try:
            return original_load(path, *args, **kwargs)
        except RuntimeError as exc:
            msg = str(exc)
            if "torchcodec" not in msg and "FFmpeg" not in msg and "libtorchcodec" not in msg:
                raise
            import numpy as np
            import soundfile as sf

            wav, sr = sf.read(path, always_2d=True, dtype="float32")
            wav = torch.from_numpy(np.asarray(wav).T.copy())
            return wav, int(sr)

    def safe_save(path, src, sample_rate, *args, **kwargs):
        try:
            return original_save(path, src, sample_rate, *args, **kwargs)
        except RuntimeError as exc:
            msg = str(exc)
            if "torchcodec" not in msg and "FFmpeg" not in msg and "libtorchcodec" not in msg:
                raise
            import numpy as np
            import soundfile as sf

            wav = src.detach().cpu()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            wav_np = np.asarray(wav.transpose(0, 1).contiguous(), dtype="float32")
            sf.write(path, wav_np, int(sample_rate))

    torchaudio.load = safe_load
    torchaudio.save = safe_save


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def init_moss(args: argparse.Namespace):
    import torch

    code_root = Path(args.moss_code_root).expanduser().resolve()
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
    from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor

    device_arg = args.device
    if device_arg == "cuda":
        device_arg = "cuda:0"
    device = torch.device(device_arg if torch.cuda.is_available() or not device_arg.startswith("cuda") else "cpu")
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype.lower(), torch.bfloat16)
    if device.type != "cuda":
        dtype = torch.float32

    processor = MossTTSDelayProcessor.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        codec_path=str(Path(args.codec_path).expanduser().resolve()),
        trust_remote_code=True,
    )
    processor.audio_tokenizer.to(device)
    model = MossTTSDelayModel.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return processor, model, device


def generate_one(
    *,
    row: dict[str, Any],
    processor: Any,
    model: Any,
    device: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch
    import torchaudio

    out_path = Path(row["style_carrier_wav"]).expanduser().resolve()
    source_wav = Path(row["source_style_wav"]).expanduser().resolve()
    result = dict(row)
    result.update(
        {
            "style_carrier_wav": str(out_path),
            "style_generation_backend": "moss_tts",
            "style_generation_status": "pending",
            "style_generation_error": "",
        }
    )
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        result["style_generation_status"] = "skipped_existing"
        return result
    if not source_wav.exists():
        result["style_generation_status"] = "failed"
        result["style_generation_error"] = f"missing source_style_wav: {source_wav}"
        return result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    source_codes = processor.encode_audios_from_path([str(source_wav)], n_vq=args.n_vq)[0]
    tokens = int(source_codes.shape[0]) if args.tokens_from_source else None
    user_message = processor.build_user_message(
        text=row["input_text"],
        reference=[source_codes],
        instruction=args.instruction,
        tokens=tokens,
        language=row.get("language"),
        quality=args.quality,
    )
    inputs = processor([[user_message]], mode="generation", n_vq=args.n_vq)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "text_temperature": float(args.text_temperature),
        "text_top_p": float(args.text_top_p),
        "text_top_k": int(args.text_top_k),
        "audio_temperature": float(args.audio_temperature),
        "audio_top_p": float(args.audio_top_p),
        "audio_top_k": int(args.audio_top_k),
        "audio_repetition_penalty": float(args.audio_repetition_penalty),
    }
    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)
    messages = processor.decode(output)
    wavs = []
    for message in messages:
        if message is None:
            continue
        for cur_wav in message.to_dict().get("audio_codes_list", []):
            if torch.is_tensor(cur_wav):
                wavs.append(cur_wav)
    if not wavs:
        raise RuntimeError("No waveform decoded from MOSS-TTS output.")
    wav = torch.cat([item.reshape(-1).detach().cpu() for item in wavs], dim=0)
    torchaudio.save(str(out_path), wav.view(1, -1), int(processor.model_config.sampling_rate))
    result["style_generation_status"] = "ok"
    result["style_carrier_num_samples"] = int(wav.numel())
    result["style_carrier_sample_rate"] = int(processor.model_config.sampling_rate)
    result["style_carrier_duration"] = float(wav.numel() / int(processor.model_config.sampling_rate))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate text-mode style carrier wavs from a style-clone plan.")
    ap.add_argument("--plan-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--backend", default="moss_tts")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--moss-code-root", default=str(DEFAULT_MOSS_CODE_ROOT))
    ap.add_argument("--model-path", default=str(DEFAULT_MOSS_MODEL_PATH))
    ap.add_argument("--codec-path", default=str(DEFAULT_CODEC_PATH))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--n-vq", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--text-temperature", type=float, default=0.8)
    ap.add_argument("--text-top-p", type=float, default=0.9)
    ap.add_argument("--text-top-k", type=int, default=50)
    ap.add_argument("--audio-temperature", type=float, default=1.2)
    ap.add_argument("--audio-top-p", type=float, default=0.8)
    ap.add_argument("--audio-top-k", type=int, default=25)
    ap.add_argument("--audio-repetition-penalty", type=float, default=1.0)
    ap.add_argument("--instruction", default=DEFAULT_STYLE_INSTRUCTION)
    ap.add_argument("--quality", default="high")
    ap.add_argument("--tokens-from-source", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overwrite-report", action="store_true")
    ap.add_argument("--print-traceback", action="store_true")
    args = ap.parse_args()

    if args.backend != "moss_tts":
        raise ValueError("This runner currently implements --backend moss_tts only.")

    patch_torchaudio_load_with_soundfile_fallback()
    plan_path = Path(args.plan_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    if args.overwrite_report and output_path.exists():
        output_path.unlink()
    rows = [row for row in load_jsonl(plan_path) if row.get("tts_backend") == args.backend]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[style_carrier] backend={args.backend} rows={len(rows)} output={output_path}")
    processor, model, device = init_moss(args)
    ok = 0
    failed = 0
    for idx, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id", f"row_{idx}")
        print(f"[style_carrier] {idx}/{len(rows)} {sample_id}")
        try:
            result = generate_one(row=row, processor=processor, model=model, device=device, args=args)
            if result.get("style_generation_status") in {"ok", "skipped_existing"}:
                ok += 1
            else:
                failed += 1
        except Exception as exc:  # keep batch generation resumable
            failed += 1
            result = dict(row)
            result["style_generation_backend"] = args.backend
            result["style_generation_status"] = "failed"
            result["style_generation_error"] = f"{type(exc).__name__}: {exc}"
            if args.print_traceback:
                result["style_generation_traceback"] = traceback.format_exc()
            print(f"[style_carrier][failed] {sample_id}: {result['style_generation_error']}", file=sys.stderr)
        append_jsonl(output_path, result)
    print(json.dumps({"rows": len(rows), "ok_or_skipped": ok, "failed": failed, "output_jsonl": str(output_path)}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
