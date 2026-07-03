#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl


ROLE_TO_OUTPUT_KEY = {
    "source": "source_speaker_embedding_path",
    "timbre_ref": "timbre_ref_speaker_embedding_path",
    "target": "target_speaker_embedding_path",
}


def get_audio_path(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value:
        return str(value)
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict) and meta.get(key):
        return str(meta[key])
    return None


def temp_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".tmp")


def done_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".done.json")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True, help="VC/encoded manifest or mixed-mode MOSS SFT JSONL")
    ap.add_argument("--embedding-plan-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument(
        "--require-embedding-exists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only attach paths whose embedding files already exist. Use --no-require-embedding-exists for mapping smoke tests.",
    )
    ap.add_argument("--max-rows", type=int, default=0, help="Limit input rows for smoke tests.")
    ap.add_argument("--progress-every", type=int, default=10000)
    args = ap.parse_args()

    by_audio_role: dict[tuple[str, str], str] = {}
    by_audio: dict[str, str] = {}
    plan_rows = 0
    skipped_missing_file = 0
    for item in iter_jsonl(args.embedding_plan_jsonl):
        plan_rows += 1
        audio = item.get("audio")
        role = item.get("role")
        emb = item.get("speaker_embedding_path")
        if not audio or not role or not emb:
            continue
        if args.require_embedding_exists and not Path(emb).exists():
            skipped_missing_file += 1
            continue
        by_audio_role[(str(audio), str(role))] = str(emb)
        by_audio.setdefault(str(audio), str(emb))

    output = Path(args.output_jsonl).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = temp_path(output)
    done_output = done_path(output)
    if tmp_output.exists():
        tmp_output.unlink()

    written = 0
    attached = 0
    attached_by_role = {role: 0 for role in ROLE_TO_OUTPUT_KEY}
    missing_by_role = {role: 0 for role in ROLE_TO_OUTPUT_KEY}
    input_rows = 0
    progress_every = max(1, int(args.progress_every))
    with tmp_output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(args.input_jsonl):
            input_rows += 1
            if args.max_rows > 0 and input_rows > args.max_rows:
                input_rows -= 1
                break
            new_row = dict(row)
            role_audio = {
                "source": get_audio_path(row, "source_audio"),
                "timbre_ref": get_audio_path(row, "timbre_ref_audio"),
                "target": get_audio_path(row, "target_audio"),
            }
            for role, audio in role_audio.items():
                if not audio:
                    continue
                emb = by_audio_role.get((str(audio), role)) or by_audio.get(str(audio))
                if not emb:
                    missing_by_role[role] += 1
                    continue
                new_row[ROLE_TO_OUTPUT_KEY[role]] = emb
                attached += 1
                attached_by_role[role] += 1
            handle.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            written += 1
            if input_rows % progress_every == 0:
                handle.flush()
                print(
                    f"processed input_rows={input_rows} written={written} attached={attached} "
                    f"attached_by_role={attached_by_role} missing_by_role={missing_by_role}",
                    flush=True,
                )
        handle.flush()
    tmp_output.replace(output)
    summary = {
        "status": "complete",
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "embedding_plan_jsonl": str(Path(args.embedding_plan_jsonl).expanduser().resolve()),
        "output_jsonl": str(output.resolve()),
        "written": written,
        "attached": attached,
        "input_rows": input_rows,
        "plan_rows": plan_rows,
        "skipped_missing_file": skipped_missing_file,
        "attached_by_role": attached_by_role,
        "missing_by_role": missing_by_role,
    }
    write_json_atomic(done_output, summary)
    print(
        f"wrote {written} rows -> {output.resolve()} attached={attached} "
        f"input_rows={input_rows} "
        f"plan_rows={plan_rows} skipped_missing_file={skipped_missing_file} "
        f"attached_by_role={attached_by_role} missing_by_role={missing_by_role}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
