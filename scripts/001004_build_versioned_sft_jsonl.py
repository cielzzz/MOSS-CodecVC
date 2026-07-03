#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl, load_torch_file, stable_id, write_jsonl
from moss_codecvc.modes import (
    VC_MODE_NO_TEXT,
    VC_MODE_TEXT,
    apply_vc_mode_token,
    mode_tag_suffix,
    parse_emit_modes,
)
from moss_codecvc.versions import build_version_instruction, get_version_spec, list_versions


COUNTERFACTUAL_CODE_KEYS = (
    "counterfactual_source_audio_codes_path",
    "source_counterfactual_audio_codes_path",
    "source_cf_audio_codes_path",
)


def load_codes(path: str) -> list[list[int]]:
    payload = load_torch_file(path)
    codes = payload["codes"] if isinstance(payload, dict) else payload
    codes = torch.as_tensor(codes, dtype=torch.long)
    if codes.ndim != 2:
        raise ValueError(f"codes must be (T, NQ), got {tuple(codes.shape)} from {path}")
    return codes.tolist()


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return None


def build_aux(row: dict[str, Any], version: str, condition_view: str, has_counterfactual: bool) -> dict[str, Any]:
    spec = get_version_spec(version)
    group_id = row.get("counterfactual_group_id") or stable_id(
        row.get("source_audio"),
        row.get("timbre_ref_audio"),
        row.get("target_audio"),
        length=20,
    )
    aux = {
        "version": version,
        "version_name": spec["name"],
        "objectives": spec["objectives"],
        "role_references": [
            {"index": 0, "name": "S1", "role": "source_content_prosody"},
            {"index": 1, "name": "S2", "role": "target_timbre"},
        ],
        "condition_view": condition_view,
        "counterfactual_group_id": group_id,
        "has_counterfactual_source": has_counterfactual,
        "role_routing": spec["aux"]["role_routing"],
        "counterfactual_invariance": spec["aux"]["counterfactual_invariance"],
        "adversarial_mi": spec["aux"]["adversarial_mi"],
        "speaker_embeddings": {
            "source": row.get("source_speaker_embedding_path"),
            "timbre_ref": row.get("timbre_ref_speaker_embedding_path"),
            "target": row.get("target_speaker_embedding_path"),
        },
    }
    return aux


def build_record(
    row: dict[str, Any],
    *,
    version: str,
    vc_mode: str,
    no_text_placeholder: str,
    enable_mode_token: bool,
    source_codes: list[list[int]],
    timbre_codes: list[list[int]],
    target_codes: list[list[int]],
    condition_view: str,
    has_counterfactual: bool,
    idx: int,
) -> dict[str, Any]:
    sample_id = row.get("sample_id") or stable_id(row.get("source_audio"), row.get("timbre_ref_audio"), idx)
    sample_id = f"{sample_id}:{mode_tag_suffix(vc_mode)}"
    if condition_view != "original_source":
        sample_id = f"{sample_id}:{condition_view}"
    if vc_mode == VC_MODE_TEXT:
        base_instruction = row.get("text_prosody_instruction") or (
            "Text-guided voice conversion task. [S1] is a prosody/style reference carrying rhythm, pauses, "
            "speaking rate, stress and duration hints. [S2] is the target timbre reference. Generate speech "
            "whose lexical content follows the provided text and whose speaker identity follows [S2]."
        )
        instruction = build_version_instruction(base_instruction, version)
        instruction += "\nUse the provided text as the lexical content target. Do not copy [S1] words."
        text = row.get("target_text") or row.get("source_text")
    elif vc_mode == VC_MODE_NO_TEXT:
        base_instruction = row.get("instruction") or (
            "Voice conversion task. [S1] is the source speech carrying content, pauses, duration and prosody. "
            "[S2] is the target timbre reference. Generate the same content as S1 with S2 timbre while preserving S1 timing and prosody."
        )
        instruction = build_version_instruction(base_instruction, version)
        instruction += "\nDo not rely on an explicit transcript. Preserve source content, pauses, duration and prosody from [S1]."
        text = no_text_placeholder
    else:
        raise ValueError(f"unsupported vc mode: {vc_mode}")
    return {
        "sample_id": sample_id,
        "text": text,
        "instruction": apply_vc_mode_token(instruction, vc_mode, enabled=enable_mode_token),
        "language": row.get("language"),
        "quality": row.get("quality") or "high",
        "tokens": int(row.get("target_codec_frames") or len(target_codes)),
        "reference_audio_codes": [source_codes, timbre_codes],
        "audio_codes": target_codes,
        "moss_codecvc_version": version,
        "moss_codecvc_mode": vc_mode,
        "moss_codecvc_text_semantics": "text_prosody" if vc_mode == VC_MODE_TEXT else None,
        "moss_codecvc_mode_token": None if not enable_mode_token else f"<vc_{vc_mode}>",
        "source_speaker_embedding_path": row.get("source_speaker_embedding_path"),
        "timbre_ref_speaker_embedding_path": row.get("timbre_ref_speaker_embedding_path"),
        "target_speaker_embedding_path": row.get("target_speaker_embedding_path"),
        "moss_codecvc_aux": build_aux(row, version, condition_view, has_counterfactual),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True, help="Encoded VC manifest from 001002_encode_codec_tokens.py")
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--version", choices=list_versions(), default="ver1")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--require-counterfactual", action="store_true")
    ap.add_argument("--require-speaker-embeddings", action="store_true")
    ap.add_argument("--emit-modes", default="text")
    ap.add_argument("--no-text-placeholder", default="<NO_TEXT>")
    ap.add_argument("--disable-mode-token", action="store_true")
    args = ap.parse_args()
    emit_modes = parse_emit_modes(args.emit_modes)

    spec = get_version_spec(args.version)
    require_counterfactual = args.require_counterfactual or bool(spec["requires_counterfactual"])
    require_speaker_embeddings = args.require_speaker_embeddings or bool(spec["requires_speaker_embeddings"])

    out = []
    skipped = 0
    missing_counterfactual = 0
    missing_speaker_embeddings = 0
    for idx, row in enumerate(iter_jsonl(args.input_jsonl)):
        if args.max_rows > 0 and len(out) >= args.max_rows:
            break
        source_path = row.get("source_audio_codes_path")
        timbre_path = row.get("timbre_ref_audio_codes_path")
        target_path = row.get("target_audio_codes_path")
        if not source_path or not timbre_path or not target_path:
            skipped += 1
            continue
        speaker_emb_ok = bool(row.get("source_speaker_embedding_path") and row.get("timbre_ref_speaker_embedding_path"))
        if require_speaker_embeddings and not speaker_emb_ok:
            missing_speaker_embeddings += 1
            continue
        cf_path = first_present(row, COUNTERFACTUAL_CODE_KEYS)
        if require_counterfactual and not cf_path:
            missing_counterfactual += 1
            continue
        try:
            source_codes = load_codes(str(source_path))
            timbre_codes = load_codes(str(timbre_path))
            target_codes = load_codes(str(target_path))
        except Exception as exc:
            skipped += 1
            print(f"skip row={idx} sample_id={row.get('sample_id')} err={type(exc).__name__}: {exc}", flush=True)
            continue
        for vc_mode in emit_modes:
            out.append(
                build_record(
                    row,
                    version=args.version,
                    vc_mode=vc_mode,
                    no_text_placeholder=args.no_text_placeholder,
                    enable_mode_token=not args.disable_mode_token,
                    source_codes=source_codes,
                    timbre_codes=timbre_codes,
                    target_codes=target_codes,
                    condition_view=row.get("condition_view") or "original_source",
                    has_counterfactual=bool(cf_path),
                    idx=idx,
                )
            )
        if cf_path:
            try:
                cf_codes = load_codes(cf_path)
            except Exception as exc:
                skipped += 1
                print(f"skip counterfactual row={idx} sample_id={row.get('sample_id')} err={type(exc).__name__}: {exc}", flush=True)
                continue
            for vc_mode in emit_modes:
                out.append(
                    build_record(
                        row,
                        version=args.version,
                        vc_mode=vc_mode,
                        no_text_placeholder=args.no_text_placeholder,
                        enable_mode_token=not args.disable_mode_token,
                        source_codes=cf_codes,
                        timbre_codes=timbre_codes,
                        target_codes=target_codes,
                        condition_view="counterfactual_source",
                        has_counterfactual=True,
                        idx=idx,
                    )
                )
    n = write_jsonl(args.output_jsonl, out)
    print(
        f"wrote {n} rows -> {Path(args.output_jsonl).resolve()} "
        f"version={args.version} skipped={skipped} "
        f"missing_counterfactual={missing_counterfactual} "
        f"missing_speaker_embeddings={missing_speaker_embeddings}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
