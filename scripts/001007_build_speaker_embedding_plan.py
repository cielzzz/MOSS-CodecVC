#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl, stable_id


AUDIO_ROLES = (
    ("source", "source_audio"),
    ("timbre_ref", "timbre_ref_audio"),
    ("target", "target_audio"),
)


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
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--embedding-root", required=True)
    ap.add_argument("--model-name", default="ecapa_tdnn_speechbrain_voxceleb")
    ap.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-rows", type=int, default=0, help="Limit input rows for smoke tests.")
    ap.add_argument("--progress-every", type=int, default=10000)
    args = ap.parse_args()

    seen: set[str] = set()
    root = Path(args.embedding_root)
    output = Path(args.output_jsonl).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = temp_path(output)
    done_output = done_path(output)
    if tmp_output.exists():
        tmp_output.unlink()

    input_rows = 0
    written = 0
    missing_by_role = {role: 0 for role, _ in AUDIO_ROLES}
    planned_by_role = {role: 0 for role, _ in AUDIO_ROLES}
    progress_every = max(1, int(args.progress_every))
    with tmp_output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(args.input_jsonl):
            input_rows += 1
            if args.max_rows > 0 and input_rows > args.max_rows:
                input_rows -= 1
                break
            sample_id = row.get("sample_id")
            for role, key in AUDIO_ROLES:
                audio = get_audio_path(row, key)
                if not audio:
                    missing_by_role[role] += 1
                    continue
                audio_key = stable_id(audio, args.model_name, length=24)
                if args.dedupe and audio_key in seen:
                    continue
                seen.add(audio_key)
                planned_by_role[role] += 1
                item = {
                    "embedding_id": audio_key,
                    "sample_id": sample_id,
                    "role": role,
                    "audio": audio,
                    "language": row.get("language"),
                    "speaker_embedding_path": str(root / f"{audio_key}.pt"),
                    "model_name": args.model_name,
                    "meta": {
                        "builder": "001007_build_speaker_embedding_plan",
                        "source_key": key,
                    },
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
            if input_rows % progress_every == 0:
                handle.flush()
                print(
                    f"processed input_rows={input_rows} written={written} "
                    f"planned_by_role={planned_by_role} missing_by_role={missing_by_role}",
                    flush=True,
                )
        handle.flush()
    tmp_output.replace(output)
    summary = {
        "status": "complete",
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "output_jsonl": str(output.resolve()),
        "written": written,
        "input_rows": input_rows,
        "planned_by_role": planned_by_role,
        "missing_by_role": missing_by_role,
        "dedupe": bool(args.dedupe),
        "model_name": args.model_name,
    }
    write_json_atomic(done_output, summary)
    print(
        f"wrote {written} speaker embedding plan rows -> {output.resolve()} "
        f"input_rows={input_rows} "
        f"planned_by_role={planned_by_role} missing_by_role={missing_by_role}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
