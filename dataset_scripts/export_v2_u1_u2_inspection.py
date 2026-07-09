#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable


SEGMENT_RE = re.compile(r"segment_(?P<start>\d+(?:_\d+)?)-(?P<end>\d+(?:_\d+)?)\.flac$")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def segment_duration(path: str) -> float | None:
    match = SEGMENT_RE.search(path)
    if not match:
        return None
    start = float(match.group("start").replace("_", "."))
    end = float(match.group("end").replace("_", "."))
    duration = end - start
    return duration if duration >= 0 else None


def link_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(source)


def path_digest(path: Path, length: int = 10) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:length]


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def short_text(value: Any, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build(args: argparse.Namespace) -> int:
    input_jsonl = Path(args.input_jsonl).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(input_jsonl), start=1):
        if args.limit > 0 and len(rows) >= args.limit:
            break
        u1 = Path(str(row.get("u1_target_audio_path") or row.get("target_audio") or "")).resolve(strict=False)
        u2 = Path(str(row.get("u2_timbre_ref_audio_path") or row.get("timbre_ref_audio") or "")).resolve(strict=False)
        if args.require_existing_audio and (not u1.exists() or not u2.exists()):
            continue
        case = f"case_{idx:02d}"
        case_dir = output_dir / case
        if args.unique_audio_names:
            u1_link = case_dir / f"{case}_u1_target_{path_digest(u1)}{u1.suffix or '.wav'}"
            u2_link = case_dir / f"{case}_u2_timbre_ref_{path_digest(u2)}{u2.suffix or '.wav'}"
        else:
            u1_link = case_dir / f"{case}_u1_target{u1.suffix or '.wav'}"
            u2_link = case_dir / f"{case}_u2_timbre_ref{u2.suffix or '.wav'}"
        link_file(u1, u1_link)
        link_file(u2, u2_link)
        item = {
            "case": case,
            "sample_id": row.get("sample_id") or "",
            "dataset_name": row.get("dataset_name") or "",
            "language": row.get("language") or "",
            "similarity": row.get("similarity"),
            "same_long_audio_channel_risk": row.get("same_long_audio_channel_risk"),
            "u1_episode_id": row.get("u1_episode_id") or "",
            "u2_episode_id": row.get("u2_episode_id") or "",
            "u1_audio": str(u1),
            "u2_audio": str(u2),
            "u1_audio_link": str(u1_link),
            "u2_audio_link": str(u2_link),
            "u1_audio_rel": rel(u1_link, output_dir),
            "u2_audio_rel": rel(u2_link, output_dir),
            "u1_duration_sec": segment_duration(str(row.get("label") or u1)),
            "u2_duration_sec": segment_duration(str(row.get("reference") or u2)),
            "u1_text": row.get("u1_text") or row.get("target_text") or "",
            "u2_text": row.get("u2_text") or row.get("timbre_ref_text") or "",
            "label": row.get("label") or "",
            "reference": row.get("reference") or "",
            "segs": row.get("segs"),
            "label_idx": row.get("label_idx"),
            "ref_idx": row.get("ref_idx"),
            "source_seedvc_job_id": row.get("source_seedvc_job_id") or "",
            "source_prepare_jsonl": row.get("source_prepare_jsonl") or "",
        }
        rows.append(item)

    write_jsonl(output_dir / "sample_manifest.jsonl", rows)

    readme: list[str] = [
        f"# {args.title}",
        "",
        f"Source JSONL: `{input_jsonl}`",
        "",
    ]
    for item in rows:
        readme.extend(
            [
                f"## {item['case']} {item['dataset_name']} {item['language']}",
                f"- similarity: `{item['similarity']}`",
                f"- same_long_audio_channel_risk: `{item['same_long_audio_channel_risk']}`",
                f"- u1_episode_id: `{item['u1_episode_id']}`",
                f"- u2_episode_id: `{item['u2_episode_id']}`",
                f"- u1_duration_sec: `{item['u1_duration_sec']}`",
                f"- u2_duration_sec: `{item['u2_duration_sec']}`",
                f"- u1 audio: `{item['u1_audio_link']}`",
                f"- u2 audio: `{item['u2_audio_link']}`",
                f"- u1 text: {item['u1_text']}",
                f"- u2 text: {item['u2_text']}",
                f"- label: `{item['label']}`",
                f"- reference: `{item['reference']}`",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    cards: list[str] = []
    for item in rows:
        cards.append(
            f"""
<section class="case">
  <h2>{html.escape(item['case'])} · {html.escape(str(item['dataset_name']))} · {html.escape(str(item['language']))}</h2>
  <div class="meta">
    <span>similarity {html.escape(str(item['similarity']))}</span>
    <span>u1 {html.escape(str(item['u1_duration_sec']))}s</span>
    <span>u2 {html.escape(str(item['u2_duration_sec']))}s</span>
    <span>same episode {html.escape(str(item['same_long_audio_channel_risk']))}</span>
  </div>
  <div class="audio-grid">
    <div>
      <h3>u1 target</h3>
      <audio controls preload="metadata" src="{html.escape(item['u1_audio_rel'])}"></audio>
      <p>{html.escape(short_text(item['u1_text']))}</p>
    </div>
    <div>
      <h3>u2 timbre ref</h3>
      <audio controls preload="metadata" src="{html.escape(item['u2_audio_rel'])}"></audio>
      <p>{html.escape(short_text(item['u2_text']) or '(no ref text loaded)')}</p>
    </div>
  </div>
  <details>
    <summary>paths and metadata</summary>
    <pre>{html.escape(json.dumps(item, ensure_ascii=False, indent=2))}</pre>
  </details>
</section>
"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(args.title)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f7f5; color: #202124; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    .intro {{ color: #5f6368; margin: 0 0 24px; }}
    .case {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 18px; margin: 0 0 16px; }}
    h2 {{ font-size: 18px; margin: 0 0 10px; }}
    h3 {{ font-size: 14px; margin: 0 0 8px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
    .meta span {{ border: 1px solid #d6d6d6; border-radius: 999px; padding: 4px 8px; font-size: 12px; color: #444; }}
    .audio-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    audio {{ width: 100%; height: 40px; }}
    p {{ line-height: 1.45; margin: 8px 0 0; }}
    details {{ margin-top: 12px; }}
    pre {{ overflow: auto; background: #f2f2ef; padding: 12px; border-radius: 6px; }}
    @media (max-width: 760px) {{ .audio-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(args.title)}</h1>
  <p class="intro">u1/u2 listening page for pending Seed-VC no-text construction. Seed-VC outputs are not generated here.</p>
  {''.join(cards)}
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {len(rows)} samples to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="V2 u1/u2 Seed-VC inspection")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--require-existing-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unique-audio-names", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
