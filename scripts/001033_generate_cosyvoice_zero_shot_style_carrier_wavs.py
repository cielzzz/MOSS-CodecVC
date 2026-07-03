#!/usr/bin/env python3
"""Generate style-carrier wavs with CosyVoice zero-shot cloning.

This runner is intentionally separate from vc_edit_framework's CosyVoice backend:
that backend uses inference_instruct2(), while this benchmark needs the simpler
zero-shot path for source_style_wav + source_style_text -> input_text.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


DEFAULT_COSYVOICE_REPO = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_edit/external/CosyVoice"
)
DEFAULT_COSYVOICE2_MODEL = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_edit/models/CosyVoice2-0.5B"
)
DEFAULT_COSYVOICE3_MODEL = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_edit/models/Fun-CosyVoice3-0.5B-2512"
)


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


def init_cosyvoice(repo_dir: Path, model_dir: Path, fp16: bool):
    repo_dir = repo_dir.expanduser().resolve()
    for path in (repo_dir, repo_dir / "third_party" / "Matcha-TTS"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore

    return AutoModel(model_dir=str(model_dir.expanduser().resolve()), fp16=fp16)


def normalize_source_row(row: dict[str, Any], backend_label: str, output_root: Path) -> dict[str, Any]:
    language = row.get("language") or "unknown"
    utt_id = row.get("utt_id") or row.get("sample_id") or "sample"
    suffix = str(utt_id)
    if suffix.startswith(f"{language}_"):
        sample_id = f"{backend_label}_{suffix}"
    else:
        sample_id = f"{backend_label}_{language}_{suffix}"
    style_carrier_wav = output_root / "style_carrier" / backend_label / language / f"{sample_id}.wav"
    return {
        "backend": backend_label,
        "sample_id": sample_id,
        "language": language,
        "source_style_wav": row.get("source_style_wav") or row.get("source_audio") or "",
        "source_style_text": row.get("source_style_text") or row.get("source_text") or "",
        "input_text": row.get("input_text") or row.get("target_text") or "",
        "timbre_ref_wav": row.get("timbre_ref_wav") or row.get("timbre_ref_audio") or "",
        "style_carrier_wav": str(style_carrier_wav),
        "source_plan_sample_id": row.get("sample_id") or "",
        "source_plan_utt_id": row.get("utt_id") or "",
    }


def prompt_text_for_model(model: Any, source_style_text: str, args: argparse.Namespace) -> str:
    model_name = model.__class__.__name__.lower()
    if "cosyvoice3" in model_name and args.c3_prompt_prefix:
        return f"{args.c3_prompt_prefix}{source_style_text}"
    return source_style_text


def generate_one(model: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import soundfile as sf

    out_path = Path(row["style_carrier_wav"]).expanduser().resolve()
    result = dict(row)
    result.update(
        {
            "style_generation_backend": args.backend_label,
            "style_generation_method": "cosyvoice_zero_shot",
            "style_generation_status": "pending",
            "style_generation_error": "",
            "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        }
    )
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        result["style_generation_status"] = "skipped_existing"
        return result

    source_wav = Path(row["source_style_wav"]).expanduser()
    if not source_wav.exists():
        result["style_generation_status"] = "failed"
        result["style_generation_error"] = f"missing source_style_wav: {source_wav}"
        return result
    if not row["input_text"]:
        result["style_generation_status"] = "failed"
        result["style_generation_error"] = "empty input_text"
        return result
    if not row["source_style_text"]:
        result["style_generation_status"] = "failed"
        result["style_generation_error"] = "empty source_style_text"
        return result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_text = prompt_text_for_model(model, row["source_style_text"], args)
    outputs = list(
        model.inference_zero_shot(
            row["input_text"],
            prompt_text,
            str(source_wav),
            stream=args.stream,
            text_frontend=not args.disable_text_frontend,
        )
    )
    if not outputs:
        raise RuntimeError("CosyVoice returned no output.")
    speech = outputs[-1]["tts_speech"]
    if hasattr(speech, "detach"):
        speech = speech.detach().cpu().numpy()
    sf.write(out_path, speech.squeeze(), int(model.sample_rate))
    result["style_carrier_wav"] = str(out_path)
    result["style_generation_status"] = "ok"
    result["style_carrier_sample_rate"] = int(model.sample_rate)
    result["style_carrier_num_chunks"] = len(outputs)
    result["style_carrier_num_samples"] = int(speech.squeeze().shape[-1])
    result["style_carrier_duration"] = float(result["style_carrier_num_samples"] / int(model.sample_rate))
    result["prompt_text_used"] = prompt_text
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cosyvoice-repo", default=str(DEFAULT_COSYVOICE_REPO))
    parser.add_argument("--model-dir", default=str(DEFAULT_COSYVOICE2_MODEL))
    parser.add_argument("--backend-label", default="cosyvoice2_zero_shot")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--languages", default="", help="Optional comma-separated language filter.")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-text-frontend", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-report", action="store_true")
    parser.add_argument(
        "--c3-prompt-prefix",
        default="You are a helpful assistant.<|endofprompt|>",
        help="CosyVoice3 zero-shot expects this prefix before prompt transcript.",
    )
    parser.add_argument("--print-traceback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = Path(args.plan_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if args.overwrite_report and output_path.exists():
        output_path.unlink()

    languages = {item.strip() for item in args.languages.split(",") if item.strip()}
    source_rows = load_jsonl(plan_path)
    if languages:
        source_rows = [row for row in source_rows if row.get("language") in languages]
    rows = [normalize_source_row(row, args.backend_label, output_root) for row in source_rows]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"[cosyvoice_zero_shot] backend={args.backend_label} rows={len(rows)}")
    print(f"[cosyvoice_zero_shot] model_dir={Path(args.model_dir).expanduser().resolve()}")
    print(f"[cosyvoice_zero_shot] output_jsonl={output_path}")
    model = init_cosyvoice(Path(args.cosyvoice_repo), Path(args.model_dir), args.fp16)
    ok = 0
    failed = 0
    for idx, row in enumerate(rows, start=1):
        print(f"[cosyvoice_zero_shot] {idx}/{len(rows)} {row['sample_id']}")
        try:
            result = generate_one(model, row, args)
            if result.get("style_generation_status") in {"ok", "skipped_existing"}:
                ok += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            result = dict(row)
            result["style_generation_backend"] = args.backend_label
            result["style_generation_method"] = "cosyvoice_zero_shot"
            result["style_generation_status"] = "failed"
            result["style_generation_error"] = f"{type(exc).__name__}: {exc}"
            if args.print_traceback:
                result["style_generation_traceback"] = traceback.format_exc()
        append_jsonl(output_path, result)
        print(
            "[cosyvoice_zero_shot] status="
            f"{result.get('style_generation_status')} ok={ok} failed={failed}"
        )
    print(f"[cosyvoice_zero_shot] done ok={ok} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
