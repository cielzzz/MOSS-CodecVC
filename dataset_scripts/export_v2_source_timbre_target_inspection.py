#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_id(*values: Any, length: int = 12) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def short_text(value: Any, limit: int = 260) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def row_audio(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    meta = row.get("meta")
    if isinstance(meta, dict):
        if "source_audio" in keys:
            u1_prime = meta.get("u1_prime")
            if isinstance(u1_prime, dict) and u1_prime.get("audio"):
                return str(u1_prime["audio"])
        if "timbre_ref_audio" in keys:
            u2 = meta.get("u2")
            if isinstance(u2, dict) and u2.get("audio"):
                return str(u2["audio"])
        if "target_audio" in keys:
            u1 = meta.get("u1")
            if isinstance(u1, dict) and u1.get("audio"):
                return str(u1["audio"])
    return ""


def row_text(row: dict[str, Any], role: str) -> str:
    if role == "source":
        return str(row.get("source_text") or row.get("target_text") or row.get("u1_text") or "")
    if role == "timbre":
        return str(row.get("timbre_ref_text") or row.get("u2_text") or "")
    return str(row.get("target_text") or row.get("u1_text") or "")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def language_sample(path: Path, *, seed: int, en_count: int, zh_count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    targets = {"en": en_count, "zh": zh_count}
    selected: dict[str, list[dict[str, Any]]] = {"en": [], "zh": []}
    seen: dict[str, int] = {"en": 0, "zh": 0}
    for row in iter_jsonl(path):
        language = str(row.get("language") or "")
        if language not in selected:
            continue
        seen[language] += 1
        bucket = selected[language]
        target = targets[language]
        if len(bucket) < target:
            bucket.append(row)
        else:
            idx = rng.randrange(seen[language])
            if idx < target:
                bucket[idx] = row
    rows = selected["en"] + selected["zh"]
    if len(rows) < en_count + zh_count:
        raise SystemExit(f"sampled {len(rows)} rows; wanted {en_count + zh_count}")
    rng.shuffle(rows)
    return rows


def parse_profile_plan(value: str) -> dict[str, int]:
    plan: dict[str, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise SystemExit(f"bad profile plan item: {item}")
        key, count = item.split(":", 1)
        plan[key.strip()] = int(count)
    return plan


def profile_balanced_sample(path: Path, *, seed: int, plan: dict[str, int]) -> list[dict[str, Any]]:
    ranked: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in iter_jsonl(path):
        profile = str(row.get("ref_channel_profile") or "")
        if profile not in plan:
            continue
        rank = stable_id(seed, profile, row.get("sample_id") or "", length=24)
        ranked[profile].append((rank, row))
    rows: list[dict[str, Any]] = []
    for profile, count in plan.items():
        items = ranked.get(profile, [])
        items.sort(key=lambda item: item[0])
        if len(items) < count:
            raise SystemExit(f"not enough rows for profile {profile}: {len(items)} < {count}")
        rows.extend(row for _, row in items[:count])
    rows.sort(key=lambda row: stable_id(seed, row.get("sample_id") or "", length=24))
    return rows


def build_page(rows: list[dict[str, Any]], output_dir: Path, *, title: str, link_mode: str) -> None:
    page_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        case = f"case_{idx:02d}"
        case_dir = output_dir / case
        source = Path(row_audio(row, "source_audio", "u1_prime_source_audio_path")).resolve(strict=False)
        timbre = Path(row_audio(row, "timbre_ref_audio", "u2_timbre_ref_audio_path")).resolve(strict=False)
        target = Path(row_audio(row, "target_audio", "u1_target_audio_path")).resolve(strict=False)
        for label, src in [("source", source), ("timbre", timbre), ("target", target)]:
            if not src.exists():
                raise SystemExit(f"missing {label}: {src}")
        source_link = case_dir / f"{case}_source_{stable_id(source)}{source.suffix or '.wav'}"
        timbre_link = case_dir / f"{case}_timbre_{stable_id(timbre)}{timbre.suffix or '.wav'}"
        target_link = case_dir / f"{case}_target_{stable_id(target)}{target.suffix or '.wav'}"
        link_or_copy(source, source_link, link_mode)
        link_or_copy(timbre, timbre_link, link_mode)
        link_or_copy(target, target_link, link_mode)
        page_rows.append(
            {
                "case": case,
                "sample_id": row.get("sample_id") or "",
                "dataset_name": row.get("dataset_name") or "",
                "language": row.get("language") or "",
                "profile": row.get("ref_channel_profile") or "",
                "treatment": row.get("ref_channel_treatment") or "",
                "source_rel": rel(source_link, output_dir),
                "timbre_rel": rel(timbre_link, output_dir),
                "target_rel": rel(target_link, output_dir),
                "source_path": str(source),
                "timbre_path": str(timbre),
                "target_path": str(target),
                "source_text": row_text(row, "source"),
                "timbre_text": row_text(row, "timbre"),
                "target_text": row_text(row, "target"),
                "u1_episode_id": row.get("u1_episode_id")
                or ((row.get("channel_shortcut_risk") or {}).get("u1_episode_id") if isinstance(row.get("channel_shortcut_risk"), dict) else ""),
                "u2_episode_id": row.get("u2_episode_id")
                or ((row.get("channel_shortcut_risk") or {}).get("u2_episode_id") if isinstance(row.get("channel_shortcut_risk"), dict) else ""),
            }
        )
    write_jsonl(output_dir / "sample_manifest.jsonl", page_rows)

    cards: list[str] = []
    for item in page_rows:
        cards.append(
            f"""
    <section class="case">
      <div class="case-head">
        <h2>{html.escape(item['case'])} · {html.escape(item['language'])} · {html.escape(item['dataset_name'])}</h2>
        <div class="meta">profile={html.escape(str(item['profile']))} · treatment={html.escape(str(item['treatment']))}</div>
      </div>
      <div class="grid">
        <div class="panel source">
          <h3>source</h3>
          <audio controls preload="metadata" src="{html.escape(item['source_rel'])}"></audio>
          <p>{html.escape(short_text(item['source_text']))}</p>
        </div>
        <div class="panel timbre">
          <h3>timbre ref</h3>
          <audio controls preload="metadata" src="{html.escape(item['timbre_rel'])}"></audio>
          <p>{html.escape(short_text(item['timbre_text']) or '(no ref text)')}</p>
        </div>
        <div class="panel target">
          <h3>target</h3>
          <audio controls preload="metadata" src="{html.escape(item['target_rel'])}"></audio>
          <p>{html.escape(short_text(item['target_text']))}</p>
        </div>
      </div>
      <details>
        <summary>paths</summary>
        <pre>sample_id: {html.escape(item['sample_id'])}
u1_episode_id: {html.escape(str(item['u1_episode_id']))}
u2_episode_id: {html.escape(str(item['u2_episode_id']))}
source: {html.escape(item['source_path'])}
timbre_ref: {html.escape(item['timbre_path'])}
target: {html.escape(item['target_path'])}</pre>
      </details>
    </section>
"""
        )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ --ink:#172026; --muted:#5f6b76; --line:#d8dee4; --bg:#f6f7f9; --panel:#fff; --source:#2563eb; --timbre:#9a3412; --target:#0f766e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:28px 32px 18px; background:var(--panel); border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:24px; letter-spacing:0; }}
    main {{ max-width:1240px; margin:0 auto; padding:22px 24px 48px; display:grid; gap:16px; }}
    .intro, .meta {{ color:var(--muted); line-height:1.55; font-size:14px; }}
    .case {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }}
    .case-head {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline; flex-wrap:wrap; margin-bottom:12px; }}
    h2 {{ margin:0; font-size:18px; letter-spacing:0; }}
    .grid {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; }}
    .panel {{ border:1px solid var(--line); border-radius:8px; padding:12px; min-width:0; }}
    .source {{ border-top:4px solid var(--source); }}
    .timbre {{ border-top:4px solid var(--timbre); }}
    .target {{ border-top:4px solid var(--target); }}
    h3 {{ margin:0 0 10px; font-size:15px; letter-spacing:0; }}
    audio {{ width:100%; display:block; margin-bottom:10px; }}
    p {{ margin:0; color:#28323c; font-size:14px; line-height:1.5; overflow-wrap:anywhere; }}
    details {{ margin-top:12px; }}
    summary {{ cursor:pointer; color:var(--muted); font-size:14px; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#f1f3f5; border:1px solid var(--line); border-radius:6px; padding:10px; font-size:12px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns:1fr; }} header {{ padding:22px 20px 14px; }} main {{ padding:18px 14px 36px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="intro">每条按 source / timbre ref / target 顺序试听，用于检查当前 no-text 三元组是否成对合理。</div>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-jsonl", default="")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--en-count", type=int, default=10)
    parser.add_argument("--zh-count", type=int, default=10)
    parser.add_argument("--profile-balanced", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-plan", default="near_flat:5,mild_eq:5,room_eq:4,codec_eq:4,phone_band:2")
    parser.add_argument("--title", default="V2 source-timbre-target triple inspection")
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.profile_balanced:
        rows = profile_balanced_sample(input_jsonl, seed=args.seed, plan=parse_profile_plan(args.profile_plan))
    else:
        rows = language_sample(input_jsonl, seed=args.seed, en_count=args.en_count, zh_count=args.zh_count)
    if args.sample_jsonl:
        write_jsonl(Path(args.sample_jsonl).expanduser().resolve(strict=False), rows)
    build_page(rows, output_dir, title=args.title, link_mode=args.link_mode)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": len(rows),
                "profile_counts": dict(Counter(str(row.get("ref_channel_profile") or "") for row in rows)),
                "language_counts": dict(Counter(str(row.get("language") or "") for row in rows)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
