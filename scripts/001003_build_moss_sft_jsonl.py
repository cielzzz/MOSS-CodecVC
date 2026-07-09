#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl, load_torch_file
from moss_codecvc.modes import (
    VC_MODE_NO_TEXT,
    VC_MODE_TEXT,
    apply_vc_mode_token,
    mode_tag_suffix,
    parse_emit_modes,
)


def load_codes(path: str) -> list[list[int]]:
    payload = load_torch_file(path)
    codes = payload["codes"] if isinstance(payload, dict) else payload
    codes = torch.as_tensor(codes, dtype=torch.long)
    if codes.ndim != 2:
        raise ValueError(f"codes must be (T, NQ), got {tuple(codes.shape)} from {path}")
    return codes.tolist()


def nested_get(row: dict[str, Any], key: str) -> Any | None:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_content_text(text: Any) -> str:
    out: list[str] = []
    for ch in str(text or "").lower():
        code = ord(ch)
        if ch.isalnum() or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            out.append(ch)
    return "".join(out)


def is_v2_real_no_text_ref_content_leak(row: dict[str, Any]) -> bool:
    pair_type = str(nested_get(row, "pair_type") or "")
    marker = nested_get(row, "v2_real_target")
    if "v2_real_target" not in pair_type and not marker:
        return False
    ref_text = normalize_content_text(nested_get(row, "timbre_ref_text"))
    target_text = normalize_content_text(nested_get(row, "target_text"))
    return bool(ref_text and target_text and ref_text == target_text)


def build_instruction(row: dict[str, Any], *, vc_mode: str, enable_mode_token: bool) -> str:
    if vc_mode == VC_MODE_TEXT:
        base = row.get("text_prosody_instruction") or (
            "Text-guided voice conversion task. [S1] is a prosody/style reference carrying rhythm, pauses, "
            "speaking rate, stress and duration hints. [S2] is the target timbre reference. Generate speech "
            "whose lexical content follows the provided text and whose speaker identity follows [S2]."
        )
        instruction = (
            f"{base}\n"
            "Role binding: TEXT=lexical content target; [S1]=prosody/style reference only; "
            "[S2]=target timbre reference. Do not copy [S1] speaker identity or [S1] words."
        )
    elif vc_mode == VC_MODE_NO_TEXT:
        base = row.get("instruction") or (
            "Voice conversion task. [S1] is the source speech carrying content, pauses, duration and prosody. "
            "[S2] is the target timbre reference. Generate the same content as S1 with S2 timbre while preserving S1 timing and prosody."
        )
        instruction = (
            f"{base}\n"
            "Role binding: [S1]=source/content/prosody carrier; [S2]=target timbre reference. "
            "Do not copy [S1] speaker identity unless it is also [S2]."
        )
        instruction += "\nDo not rely on an explicit transcript. Preserve source content, pauses, duration and prosody from [S1]."
    else:
        raise ValueError(f"unsupported vc mode: {vc_mode}")
    return apply_vc_mode_token(instruction, vc_mode, enabled=enable_mode_token)


def resolve_text(row: dict[str, Any], mode: str, placeholder: str) -> str | None:
    if mode == "target":
        return row.get("target_text") or row.get("source_text")
    if mode == "source":
        return row.get("source_text") or row.get("target_text")
    if mode == "empty":
        return ""
    if mode == "placeholder":
        return placeholder
    raise ValueError(f"unsupported text mode: {mode}")


def resolve_text_for_vc_mode(
    row: dict[str, Any],
    *,
    vc_mode: str,
    text_mode: str,
    text_placeholder: str,
    no_text_text_mode: str,
    no_text_placeholder: str,
) -> str | None:
    if vc_mode == VC_MODE_TEXT:
        return resolve_text(row, text_mode, text_placeholder)
    if vc_mode == VC_MODE_NO_TEXT:
        return resolve_text(row, no_text_text_mode, no_text_placeholder)
    raise ValueError(f"unsupported vc mode: {vc_mode}")


def temp_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".tmp")


def progress_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".progress.json")


def done_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".done.json")


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def truncate_jsonl_to_lines(path: Path, expected_lines: int) -> None:
    if expected_lines <= 0:
        path.write_text("", encoding="utf-8")
        return
    tmp = path.with_name(path.name + ".truncate")
    kept = 0
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if kept >= expected_lines:
                break
            dst.write(line)
            kept += 1
    if kept != expected_lines:
        raise RuntimeError(f"Cannot resume {path}: expected {expected_lines} rows, found {kept}")
    tmp.replace(path)


def build_sft_row(
    row: dict[str, Any],
    *,
    source_codes: list[list[int]],
    timbre_codes: list[list[int]],
    target_codes: list[list[int]],
    vc_mode: str,
    emit_modes: list[str],
    text_mode: str,
    text_placeholder: str,
    no_text_text_mode: str,
    no_text_placeholder: str,
    disable_mode_token: bool,
) -> dict[str, Any]:
    target_text = resolve_text_for_vc_mode(
        row,
        vc_mode=vc_mode,
        text_mode=text_mode,
        text_placeholder=text_placeholder,
        no_text_text_mode=no_text_text_mode,
        no_text_placeholder=no_text_placeholder,
    )
    sample_id = row.get("sample_id")
    if len(emit_modes) > 1:
        sample_id = f"{sample_id}:{mode_tag_suffix(vc_mode)}"
    out = {
        "sample_id": sample_id,
        "text": target_text,
        "instruction": build_instruction(
            row,
            vc_mode=vc_mode,
            enable_mode_token=not disable_mode_token,
        ),
        "language": row.get("language"),
        "quality": "high",
        "tokens": int(row.get("target_codec_frames") or len(target_codes)),
        "reference_audio_codes": [source_codes, timbre_codes],
        "audio_codes": target_codes,
        "moss_codecvc_mode": vc_mode,
        "moss_codecvc_text_semantics": "text_prosody" if vc_mode == VC_MODE_TEXT else None,
        "moss_codecvc_mode_token": None if disable_mode_token else f"<vc_{vc_mode}>",
        "source_speaker_embedding_path": row.get("source_speaker_embedding_path"),
        "timbre_ref_speaker_embedding_path": row.get("timbre_ref_speaker_embedding_path"),
        "target_speaker_embedding_path": row.get("target_speaker_embedding_path"),
        "moss_codecvc_meta": {
            "pair_type": row.get("pair_type"),
            "source_audio": row.get("source_audio"),
            "timbre_ref_audio": row.get("timbre_ref_audio"),
            "target_audio": row.get("target_audio"),
            "source_codec_frames": row.get("source_codec_frames"),
            "timbre_ref_codec_frames": row.get("timbre_ref_codec_frames"),
            "target_codec_frames": row.get("target_codec_frames"),
        },
    }
    for key in (
        "source_text",
        "target_text",
        "timbre_ref_text",
        "source_speaker_id",
        "timbre_ref_speaker_id",
        "target_speaker_id",
        "source_gender",
        "timbre_ref_gender",
        "target_gender",
        "validation_set",
        "ref_channel_treatment",
        "ref_channel_profile",
        "ref_channel_seed",
        "ref_channel_severity",
        "ref_channel_target_sample_rate",
        "timbre_ref_audio_original",
        "timbre_ref_channel_augmented",
        "timbre_ref_channel_augmentation",
        "channel_shortcut_risk",
        "v2_real_target",
        "text_policy",
    ):
        if key in row:
            out[key] = row.get(key)
    meta = out["moss_codecvc_meta"]
    for key in (
        "source_text",
        "target_text",
        "timbre_ref_text",
        "validation_set",
        "ref_channel_treatment",
        "ref_channel_profile",
    ):
        if key in row:
            meta[key] = row.get(key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--require-target", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument(
        "--text-mode",
        choices=("target", "source", "empty", "placeholder"),
        default="target",
        help="How to fill the MOSS text field. Use empty/placeholder for no-text VC ablations.",
    )
    ap.add_argument("--text-placeholder", default="<NO_TEXT>")
    ap.add_argument(
        "--emit-modes",
        default="text",
        help="Comma-separated VC modes to emit: text, no_text, or text,no_text for mixed training.",
    )
    ap.add_argument(
        "--no-text-text-mode",
        choices=("empty", "placeholder"),
        default="placeholder",
        help="How to fill the text field for no-text VC rows.",
    )
    ap.add_argument("--no-text-placeholder", default="<NO_TEXT>")
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from output-jsonl.tmp plus output-jsonl.progress.json when available.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Flush output, write progress, and print heartbeat every N input rows.",
    )
    ap.add_argument(
        "--respect-preferred-emit-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If manifest rows contain preferred_emit_mode, only emit that mode for that row.",
    )
    ap.add_argument(
        "--filter-v2-real-no-text-ref-content-leak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For v2 real-target no-text triples, skip no_text rows whose timbre_ref_text "
            "normalizes exactly to target_text. Those rows let S2 carry lexical content."
        ),
    )
    ap.add_argument("--disable-mode-token", action="store_true")
    args = ap.parse_args()
    emit_modes = parse_emit_modes(args.emit_modes)

    output_jsonl = Path(args.output_jsonl).expanduser()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_jsonl = temp_path(output_jsonl)
    progress_json = progress_path(output_jsonl)
    done_json = done_path(output_jsonl)

    input_resolved = str(Path(args.input_jsonl).expanduser().resolve())
    output_resolved = str(output_jsonl.resolve())
    resume_payload: dict[str, Any] | None = None
    if args.resume and tmp_jsonl.exists() and progress_json.exists():
        try:
            payload = json.loads(progress_json.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ignore invalid progress file {progress_json}: {type(exc).__name__}: {exc}", flush=True)
            payload = {}
        if (
            payload.get("input_jsonl") == input_resolved
            and payload.get("output_jsonl") == output_resolved
            and payload.get("emit_modes") == emit_modes
            and payload.get("max_rows") == args.max_rows
        ):
            resume_payload = payload
        else:
            print(f"ignore stale progress for {output_jsonl}", flush=True)

    if resume_payload is None:
        if tmp_jsonl.exists():
            tmp_jsonl.unlink()
        if progress_json.exists():
            progress_json.unlink()
        next_input_index = 0
        written = 0
        skipped = 0
        filtered_ref_content_leak = 0
        mode = "w"
    else:
        next_input_index = int(resume_payload.get("next_input_index") or 0)
        written = int(resume_payload.get("written") or 0)
        skipped = int(resume_payload.get("skipped") or 0)
        filtered_ref_content_leak = int(resume_payload.get("filtered_ref_content_leak") or 0)
        truncate_jsonl_to_lines(tmp_jsonl, written)
        mode = "a"
        print(
            f"resume SFT build from input_row={next_input_index} written={written} skipped={skipped} tmp={tmp_jsonl}",
            flush=True,
        )

    progress_every = max(1, int(args.progress_every))

    def save_progress(status: str, current_next_input_index: int) -> None:
        write_progress(
            progress_json,
            {
                "status": status,
                "input_jsonl": input_resolved,
                "output_jsonl": output_resolved,
                "tmp_jsonl": str(tmp_jsonl.resolve()),
                "emit_modes": emit_modes,
                "max_rows": args.max_rows,
                "next_input_index": current_next_input_index,
                "written": written,
                "skipped": skipped,
                "filtered_ref_content_leak": filtered_ref_content_leak,
            },
        )

    last_next_input_index = next_input_index
    with tmp_jsonl.open(mode, encoding="utf-8") as handle:
        for idx, row in enumerate(iter_jsonl(args.input_jsonl)):
            if idx < next_input_index:
                continue
            if args.max_rows > 0 and written >= args.max_rows:
                break

            source_codes_path = row.get("source_audio_codes_path")
            timbre_codes_path = row.get("timbre_ref_audio_codes_path")
            target_codes_path = row.get("target_audio_codes_path")
            if not source_codes_path or not timbre_codes_path or not target_codes_path:
                skipped += 1
                if args.require_target:
                    last_next_input_index = idx + 1
                    continue
            try:
                source_codes = load_codes(source_codes_path)
                timbre_codes = load_codes(timbre_codes_path)
                target_codes = load_codes(target_codes_path)
            except Exception as exc:
                skipped += 1
                print(f"skip row={idx} sample_id={row.get('sample_id')} err={type(exc).__name__}: {exc}", flush=True)
                last_next_input_index = idx + 1
                continue
            preferred_emit_mode = row.get("preferred_emit_mode")
            row_emit_modes = []
            for vc_mode in emit_modes:
                if args.respect_preferred_emit_mode and preferred_emit_mode and vc_mode != preferred_emit_mode:
                    continue
                if (
                    bool(args.filter_v2_real_no_text_ref_content_leak)
                    and vc_mode == VC_MODE_NO_TEXT
                    and is_v2_real_no_text_ref_content_leak(row)
                ):
                    filtered_ref_content_leak += 1
                    continue
                row_emit_modes.append(vc_mode)
            if not row_emit_modes:
                skipped += 1
                last_next_input_index = idx + 1
                continue
            for vc_mode in row_emit_modes:
                out_row = build_sft_row(
                    row,
                    source_codes=source_codes,
                    timbre_codes=timbre_codes,
                    target_codes=target_codes,
                    vc_mode=vc_mode,
                    emit_modes=emit_modes,
                    text_mode=args.text_mode,
                    text_placeholder=args.text_placeholder,
                    no_text_text_mode=args.no_text_text_mode,
                    no_text_placeholder=args.no_text_placeholder,
                    disable_mode_token=args.disable_mode_token,
                )
                handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1
            last_next_input_index = idx + 1
            if last_next_input_index % progress_every == 0:
                handle.flush()
                save_progress("running", last_next_input_index)
                print(
                    f"processed input_rows={last_next_input_index} written={written} skipped={skipped}",
                    flush=True,
                )
        handle.flush()
        save_progress("complete", last_next_input_index)

    tmp_jsonl.replace(output_jsonl)
    write_progress(
        done_json,
        {
            "status": "complete",
            "input_jsonl": input_resolved,
            "output_jsonl": output_resolved,
            "emit_modes": emit_modes,
            "max_rows": args.max_rows,
            "input_rows_processed": last_next_input_index,
            "written": written,
            "skipped": skipped,
            "filtered_ref_content_leak": filtered_ref_content_leak,
            "filter_v2_real_no_text_ref_content_leak": bool(args.filter_v2_real_no_text_ref_content_leak),
        },
    )
    if progress_json.exists():
        progress_json.unlink()
    print(
        f"wrote {written} MOSS SFT rows -> {output_jsonl.resolve()} "
        f"skipped={skipped} filtered_ref_content_leak={filtered_ref_content_leak}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
