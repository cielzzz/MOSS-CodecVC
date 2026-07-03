#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any


def patch_torch_load_weights_only() -> None:
    import torch

    original_load = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load


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


def build_output_path(output_root: Path, row: dict[str, Any]) -> Path:
    sample_id = str(row["sample_id"]).replace("moss_tts_", "styletts2_")
    return output_root / "style_carrier" / "styletts2" / str(row.get("language") or "unknown") / f"{sample_id}.wav"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate StyleTTS2 style carrier wavs from text style-clone plan.")
    parser.add_argument("--plan-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--languages", default="", help="Optional comma-separated language filter.")
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-report", action="store_true")
    parser.add_argument("--print-traceback", action="store_true")
    args = parser.parse_args()

    patch_torch_load_weights_only()
    import nltk
    import soundfile as sf
    from styletts2 import tts

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

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

    print(f"[styletts2] rows={len(rows)}")
    model = None
    ok = 0
    failed = 0
    unsupported = 0
    for idx, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"]).replace("moss_tts_", "styletts2_")
        out_wav = build_output_path(output_root, row)
        result = {
            **row,
            "sample_id": sample_id,
            "tts_backend": "styletts2",
            "style_carrier_wav": str(out_wav),
            "style_generation_backend": "styletts2",
            "style_generation_status": "pending",
            "style_generation_error": "",
        }
        print(f"[styletts2] {idx}/{len(rows)} {sample_id}")
        if str(row.get("language")) != "en":
            unsupported += 1
            result["style_generation_status"] = "unsupported_language"
            result["style_generation_error"] = (
                "StyleTTS2 package default model is LibriTTS English with gruut-lang-en; "
                "Chinese text is not supported for fair style-clone generation."
            )
            append_jsonl(output_path, result)
            continue
        try:
            if args.skip_existing and out_wav.exists() and out_wav.stat().st_size > 0:
                result["style_generation_status"] = "skipped_existing"
                info = sf.info(str(out_wav))
                result["style_carrier_sample_rate"] = int(info.samplerate)
                result["style_carrier_num_samples"] = int(info.frames)
                result["style_carrier_duration"] = float(info.duration)
            else:
                if model is None:
                    model = tts.StyleTTS2()
                out_wav.parent.mkdir(parents=True, exist_ok=True)
                wav = model.inference(
                    str(row["input_text"]),
                    target_voice_path=str(row["source_style_wav"]),
                    output_wav_file=str(out_wav),
                    diffusion_steps=int(args.diffusion_steps),
                )
                info = sf.info(str(out_wav))
                result["style_generation_status"] = "ok"
                result["style_carrier_sample_rate"] = int(info.samplerate)
                result["style_carrier_num_samples"] = int(info.frames)
                result["style_carrier_duration"] = float(info.duration)
                if hasattr(wav, "shape"):
                    result["style_generation_wave_shape"] = list(wav.shape)
            ok += 1
        except Exception as exc:
            failed += 1
            result["style_generation_status"] = "failed"
            result["style_generation_error"] = f"{type(exc).__name__}: {exc}"
            if args.print_traceback:
                result["style_generation_traceback"] = traceback.format_exc()
            print(f"[styletts2][failed] {sample_id}: {result['style_generation_error']}")
        append_jsonl(output_path, result)
    print(json.dumps({"rows": len(rows), "ok_or_skipped": ok, "unsupported": unsupported, "failed": failed, "output_jsonl": str(output_path)}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
