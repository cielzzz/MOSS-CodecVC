#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import shutil
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


def sample_rows(path: Path, *, seed: int, en_count: int, zh_count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {"en": [], "zh": []}
    seen: dict[str, int] = {"en": 0, "zh": 0}
    targets = {"en": en_count, "zh": zh_count}
    for row in iter_jsonl(path):
        language = str(row.get("language") or "")
        if language not in selected:
            continue
        seen[language] += 1
        bucket = selected[language]
        if len(bucket) < targets[language]:
            bucket.append(row)
        else:
            idx = rng.randrange(seen[language])
            if idx < targets[language]:
                bucket[idx] = row
    out = selected["en"] + selected["zh"]
    if len(out) < en_count + zh_count:
        raise SystemExit(f"only sampled {len(out)} rows from {path}; wanted {en_count + zh_count}")
    rng.shuffle(out)
    return out


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def short_text(value: Any, limit: int = 300) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def row_audio(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def build_page(rows: list[dict[str, Any]], output_dir: Path, *, title: str, link_mode: str) -> None:
    page_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        case = f"case_{idx:02d}"
        case_dir = output_dir / case
        target = Path(row_audio(row, "u1_target_audio_path", "target_audio")).resolve(strict=False)
        ref_aug = Path(row_audio(row, "u2_timbre_ref_audio_path", "timbre_ref_audio")).resolve(strict=False)
        ref_orig = Path(row_audio(row, "u2_timbre_ref_audio_path_original", "timbre_ref_audio_original")).resolve(strict=False)
        if not ref_orig.exists():
            ref_orig = ref_aug
        for label, src in [("target_u1", target), ("ref_u2_original", ref_orig), ("ref_u2_aug", ref_aug)]:
            if not src.exists():
                raise SystemExit(f"missing {label}: {src}")
        target_link = case_dir / f"{case}_target_u1_{stable_id(target)}{target.suffix or '.wav'}"
        ref_orig_link = case_dir / f"{case}_ref_u2_original_{stable_id(ref_orig)}{ref_orig.suffix or '.wav'}"
        ref_aug_link = case_dir / f"{case}_ref_u2_aug_{stable_id(ref_aug)}{ref_aug.suffix or '.wav'}"
        link_or_copy(target, target_link, link_mode)
        link_or_copy(ref_orig, ref_orig_link, link_mode)
        link_or_copy(ref_aug, ref_aug_link, link_mode)
        page_rows.append(
            {
                "case": case,
                "sample_id": row.get("sample_id") or "",
                "dataset_name": row.get("dataset_name") or "",
                "language": row.get("language") or "",
                "similarity": row.get("similarity"),
                "u1_episode_id": row.get("u1_episode_id") or "",
                "u2_episode_id": row.get("u2_episode_id") or "",
                "u1_text": row.get("u1_text") or row.get("target_text") or "",
                "u2_text": row.get("u2_text") or row.get("timbre_ref_text") or "",
                "target_rel": rel(target_link, output_dir),
                "ref_orig_rel": rel(ref_orig_link, output_dir),
                "ref_aug_rel": rel(ref_aug_link, output_dir),
                "target_path": str(target),
                "ref_orig_path": str(ref_orig),
                "ref_aug_path": str(ref_aug),
                "aug": row.get("timbre_ref_channel_augmentation") or {},
            }
        )
    write_jsonl(output_dir / "sample_manifest.jsonl", page_rows)

    cards = []
    for item in page_rows:
        aug = item["aug"] if isinstance(item["aug"], dict) else {}
        profile = aug.get("profile") or ""
        seed = aug.get("seed") or ""
        loudness = aug.get("loudness_match") if isinstance(aug.get("loudness_match"), dict) else {}
        gain = loudness.get("gain_db", "")
        original_mean = loudness.get("original_mean_volume_db", "")
        post_mean = loudness.get("post_match_augmented_mean_volume_db", "")
        cards.append(
            f"""
    <section class="case">
      <div class="case-head">
        <h2>{html.escape(item['case'])} · {html.escape(item['language'])} · {html.escape(item['dataset_name'])}</h2>
        <div class="meta">sim={html.escape(str(item['similarity']))} · aug={html.escape(str(profile))} · gain={html.escape(str(gain))}dB · seed={html.escape(str(seed))}</div>
      </div>
      <div class="grid">
        <div class="panel target">
          <h3>target u1</h3>
          <audio controls preload="metadata" src="{html.escape(item['target_rel'])}"></audio>
          <p>{html.escape(short_text(item['u1_text']))}</p>
        </div>
        <div class="panel original">
          <h3>original timbre ref u2</h3>
          <audio controls preload="metadata" src="{html.escape(item['ref_orig_rel'])}"></audio>
          <p>{html.escape(short_text(item['u2_text']))}</p>
        </div>
        <div class="panel augmented">
          <h3>ref-channel-aug u2</h3>
          <audio controls preload="metadata" src="{html.escape(item['ref_aug_rel'])}"></audio>
          <p>{html.escape(short_text(item['u2_text']))}</p>
        </div>
      </div>
      <details>
        <summary>paths</summary>
        <pre>sample_id: {html.escape(item['sample_id'])}
u1_episode_id: {html.escape(item['u1_episode_id'])}
u2_episode_id: {html.escape(item['u2_episode_id'])}
target: {html.escape(item['target_path'])}
ref_original: {html.escape(item['ref_orig_path'])}
ref_aug: {html.escape(item['ref_aug_path'])}
original_mean_volume_db: {html.escape(str(original_mean))}
post_match_augmented_mean_volume_db: {html.escape(str(post_mean))}</pre>
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
    :root {{ --ink:#172026; --muted:#5f6b76; --line:#d8dee4; --bg:#f6f7f9; --panel:#fff; --target:#0f766e; --orig:#334155; --aug:#9a3412; }}
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
    .target {{ border-top:4px solid var(--target); }}
    .original {{ border-top:4px solid var(--orig); }}
    .augmented {{ border-top:4px solid var(--aug); }}
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
    <div class="intro">每条依次听 target u1、原始 u2、ref-channel-aug u2。这个页面只检查 ref 侧扰动是否过强，不包含 Seed-VC 输出。</div>
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
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--en-count", type=int, default=3)
    parser.add_argument("--zh-count", type=int, default=2)
    parser.add_argument("--title", default="V2 ref-channel augmentation inspection")
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sample_rows(input_jsonl, seed=args.seed, en_count=args.en_count, zh_count=args.zh_count)
    if args.sample_jsonl:
        write_jsonl(Path(args.sample_jsonl).expanduser().resolve(strict=False), rows)
    build_page(rows, output_dir, title=args.title, link_mode=args.link_mode)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
