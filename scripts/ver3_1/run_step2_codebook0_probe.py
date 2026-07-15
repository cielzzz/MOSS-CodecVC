#!/usr/bin/env python
"""Batch-45 Step 2: official Seed-TTS-Eval LFQ codebook-0 probe.

The original Batch-45 note asks for 20 speakers x 5 clips per language.  The
distributed Seed-TTS-Eval package does not contain that contract: each
recoverable group has at most one prompt plus two target clips, and English
does not ship Common Voice client/speaker metadata.  This runner therefore
materializes an auditable *reduced* protocol:

* Chinese prefix groups are true speaker labels (20 groups x 3 clips) and are
  the only language used for the A/B path gate.
* English groups are deterministic prompt-clip proxies and are exploratory.
* Exact transcript matches are used for same-content/cross-group pairs; the
  available English set may yield fewer than 20 such pairs.

The tokenizer currently uses MossAudioTokenizerResidualLFQ, not a classical
RVQ.  We call this ``LFQ codebook-0`` throughout the outputs to avoid making
an incorrect architectural claim.  Only the first codebook indices and its
8-D embedding table are probed; this is not a 512-D semantic condition.

The script has three phases:

1. ``--prepare-only`` parses the official meta files and writes a frozen
   manifest plus pair lists.
2. ``--encode`` (optionally sharded with ``--worker-id/--num-workers``)
   encodes clips and writes per-worker JSONL results.
3. default aggregation validates every worker, computes pair metrics,
   bootstrap intervals, plots and writes the conservative A/B decision.

All paths and hashes are recorded in the output, so a QZ wrapper can run the
same code offline without silently substituting another dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


DEFAULT_DATA_ROOT = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/"
    "batch42/datasets/seed-tts-eval/seedtts_testset"
)
# Step 2 is an evaluation/probe artifact rather than a training target
# dataset; keep it under testset/outputs and leave ``prepared/`` for zq and
# future semantic tensors.
DEFAULT_OUTPUT_ROOT = ROOT / "testset/outputs/ver3_1_step2_codebook0_probe_20260715"
DEFAULT_CODEC_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "vcdata_construction/MOSS-Audio-Tokenizer"
)
DEFAULT_CONFIG = ROOT / "configs/remote_full.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    # Keep letters/numbers from all scripts; remove punctuation and spacing.
    return "".join(ch for ch in text if ch.isalnum())


def split_target_stem(uid: str, prompt_stem: str) -> str:
    prefix = prompt_stem + "_"
    if not uid.startswith(prefix):
        raise ValueError(f"meta uid does not start with prompt stem: uid={uid!r} prompt={prompt_stem!r}")
    target = uid[len(prefix) :]
    if not target:
        raise ValueError(f"empty target stem in {uid!r}")
    return target


def group_id(language: str, target_stem: str) -> str:
    if language == "zh":
        return target_stem.split("-", 1)[0]
    marker = "-common_voice_en_"
    if marker not in target_stem:
        raise ValueError(f"English target stem has no pair marker: {target_stem!r}")
    return target_stem.split(marker, 1)[0]


@dataclass(frozen=True)
class Clip:
    clip_id: str
    language: str
    group_id: str
    label_source: str
    role: str
    text: str
    content_key: str
    audio_path: str
    source_meta: str


def parse_meta(data_root: Path, language: str) -> tuple[list[Clip], dict[str, Any]]:
    meta_path = data_root / language / "meta.lst"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    rows: list[tuple[str, str, str, str, str, str]] = []
    # uid|prompt text|prompt relative path|target text
    for line_no, line in enumerate(meta_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 4:
            raise ValueError(f"{meta_path}:{line_no}: expected 4 fields, got {len(parts)}")
        uid, prompt_text, prompt_rel, target_text = parts
        prompt_path = data_root / language / prompt_rel
        prompt_stem = Path(prompt_rel).stem
        # In the official ``meta.lst`` the first field is already the target
        # clip stem (the prompt stem is carried separately in field 3).  The
        # ``non_para_reconstruct_meta.lst`` uses a combined prompt/target UID;
        # Step 2 intentionally uses ``meta.lst`` because it is the canonical
        # prompt+target group definition.
        target_stem = uid
        target_path = data_root / language / "wavs" / f"{uid}.wav"
        # The target filename is the complete meta UID in the official set.
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        if not prompt_path.is_file():
            raise FileNotFoundError(prompt_path)
        rows.append((uid, prompt_stem, prompt_text, target_text, str(prompt_path), str(target_path)))

    # A group is keyed by the target speaker prefix/base clip.  The prompt
    # clip for that group occurs in another row; target rows contribute up to
    # two target clips.  This is the documented 3-clip closed loop.
    prompt_by_group: dict[str, tuple[str, str, str]] = {}
    for _, prompt_stem, prompt_text, _, prompt_path, _ in rows:
        pg = prompt_stem.split("-", 1)[0] if language == "zh" else prompt_stem
        prompt_by_group.setdefault(pg, (prompt_stem, prompt_text, prompt_path))

    clips: list[Clip] = []
    target_seen: set[str] = set()
    group_targets: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for uid, prompt_stem, _, target_text, _, target_path in rows:
        target_stem = uid
        gid = group_id(language, target_stem)
        group_targets[gid].append((target_stem, target_text, target_path, uid))

    # Only retain groups with the intended 1 prompt + 2 distinct targets.
    complete_groups = 0
    for gid in sorted(group_targets):
        targets = []
        for target_stem, text, target_path, uid in group_targets[gid]:
            if target_path in target_seen:
                continue
            target_seen.add(target_path)
            targets.append((target_stem, text, target_path, uid))
        prompt = prompt_by_group.get(gid)
        if prompt is None or len(targets) < 2:
            continue
        complete_groups += 1
        p_stem, p_text, p_path = prompt
        clips.append(
            Clip(
                clip_id=f"{language}:{gid}:prompt:{p_stem}",
                language=language,
                group_id=gid,
                label_source="zh_prefix" if language == "zh" else "en_prompt_proxy",
                role="prompt",
                text=p_text,
                content_key=norm_text(p_text),
                audio_path=str(Path(p_path).resolve()),
                source_meta=str(meta_path.resolve()),
            )
        )
        for target_stem, text, target_path, uid in sorted(targets[:2]):
            clips.append(
                Clip(
                    clip_id=f"{language}:{gid}:target:{target_stem}",
                    language=language,
                    group_id=gid,
                    label_source="zh_prefix" if language == "zh" else "en_prompt_proxy",
                    role="target",
                    text=text,
                    content_key=norm_text(text),
                    audio_path=str(Path(target_path).resolve()),
                    source_meta=str(meta_path.resolve()),
                )
            )
    if not clips:
        raise RuntimeError(f"no complete groups parsed from {meta_path}")
    return clips, {
        "language": language,
        "meta_path": str(meta_path.resolve()),
        "meta_sha256": sha256_file(meta_path),
        "rows": len(rows),
        "complete_groups": complete_groups,
        "clip_count": len(clips),
        "label_source": "zh_prefix" if language == "zh" else "en_prompt_proxy",
    }


def choose_pairs(clips: list[Clip], language: str, max_groups: int = 20, max_content_pairs: int = 20) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    by_group: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        by_group[clip.group_id].append(clip)
    complete = [g for g in sorted(by_group) if len(by_group[g]) >= 3]
    # Do not take the lexicographically first IDs: in the Chinese archive
    # those IDs cluster by recording source.  A fixed seed keeps the sample
    # reproducible while spreading it over the available group population.
    seed = 20260715 + (0 if language == "zh" else 1)
    rng = random.Random(seed)
    selected_groups = sorted(rng.sample(complete, min(max_groups, len(complete))))
    selected_set = set(selected_groups)
    selected_clips = [c for c in clips if c.group_id in selected_set]
    pairs: list[dict[str, Any]] = []
    used_clip_ids: set[str] = set()

    # One same-speaker/cross-content relation per selected group.
    for gid in selected_groups:
        group = sorted(by_group[gid], key=lambda c: (c.role != "prompt", c.clip_id))
        chosen = None
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.content_key and b.content_key and a.content_key != b.content_key:
                    chosen = (a, b)
                    break
            if chosen:
                break
        if not chosen:
            continue
        a, b = chosen
        pairs.append(
            {
                "pair_id": f"{language}:same_speaker_cross_content:{len(pairs):03d}",
                "relation": "same_speaker_cross_content",
                "language": language,
                "speaker_a": gid,
                "speaker_b": gid,
                "clip_a": a.clip_id,
                "clip_b": b.clip_id,
                "content_key_a": a.content_key,
                "content_key_b": b.content_key,
                "label_source": a.label_source,
            }
        )
        used_clip_ids.update((a.clip_id, b.clip_id))

    # Same-content/cross-group: exact normalized transcript, deterministic
    # greedy pairing. Prefer selected groups, then expand if fewer than 20.
    by_content: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        if clip.role == "target" and clip.content_key:
            by_content[clip.content_key].append(clip)
    content_pairs: list[dict[str, Any]] = []
    for key in sorted(by_content):
        candidates = sorted(by_content[key], key=lambda c: (c.group_id, c.clip_id))
        # Distinct groups only; one pair per text key avoids overweighting a
        # repeated sentence.
        for i, a in enumerate(candidates):
            b = next((x for x in candidates[i + 1 :] if x.group_id != a.group_id), None)
            if b is None:
                continue
            content_pairs.append(
                {
                    "pair_id": f"{language}:same_content_cross_speaker:{len(content_pairs):03d}",
                    "relation": "same_content_cross_speaker",
                    "language": language,
                    "speaker_a": a.group_id,
                    "speaker_b": b.group_id,
                    "clip_a": a.clip_id,
                    "clip_b": b.clip_id,
                    "content_key_a": key,
                    "content_key_b": key,
                    "label_source": a.label_source,
                }
            )
            break
    # For the strict Chinese gate, use up to 20. English may have only 15.
    content_pairs = content_pairs[:max_content_pairs]
    pairs.extend(content_pairs)
    for p in content_pairs:
        used_clip_ids.update((p["clip_a"], p["clip_b"]))

    # The union is enough for encoding and includes the 20 selected groups
    # plus any extra groups needed by content pairs.
    needed_groups = set(selected_groups)
    for p in content_pairs:
        needed_groups.update((p["speaker_a"], p["speaker_b"]))
    selected_clips = [c for c in clips if c.group_id in needed_groups]
    return pairs, [c.clip_id for c in selected_clips], selected_groups


def materialize_manifest(data_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty Step 2 output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    all_clips: dict[str, Clip] = {}
    language_stats: dict[str, Any] = {}
    pairs: list[dict[str, Any]] = []
    required_by_lang: dict[str, set[str]] = {}
    for lang in ("zh", "en"):
        clips, stats = parse_meta(data_root, lang)
        language_stats[lang] = stats
        for c in clips:
            all_clips[c.clip_id] = c
        lang_pairs, required, selected_groups = choose_pairs(clips, lang)
        stats["selected_primary_groups"] = selected_groups
        stats["selection_seed"] = 20260715 + (0 if lang == "zh" else 1)
        pairs.extend(lang_pairs)
        required_by_lang[lang] = set(required)

    # Keep only clips used by either relation.  Deterministic order makes the
    # worker sharding and hashes reproducible.
    required = set().union(*required_by_lang.values())
    clips_out = [all_clips[k] for k in sorted(required)]
    manifest = {
        "schema": "ver3_1_step2_codebook0_probe_v1",
        "generated_at_utc": utc_now(),
        "protocol": {
            "requested": "20 speakers x 5 clips per language",
            "actual": "official reduced: 20 primary groups x up to 3 clips plus an explicitly listed content-pair pool; English group label is proxy",
            "gate_language": "zh",
            "english_role": "exploratory_only",
            "max_groups_per_language": 20,
            "max_same_content_pairs_per_language": 20,
        },
        "data_root": str(data_root.resolve()),
        "source_dataset": "Seed-TTS-Eval official meta.lst TTS clips (not non_para_reconstruct_meta.lst VC pairs)",
        "archive": (
            {
                "path": str((data_root.parent / "seedtts_testset.tar").resolve()),
                "sha256": sha256_file(data_root.parent / "seedtts_testset.tar"),
            }
            if (data_root.parent / "seedtts_testset.tar").is_file()
            else None
        ),
        "languages": language_stats,
        "clips": [asdict(c) for c in clips_out],
        "pairs": pairs,
    }
    write_json(output_root / "probe_manifest.json", manifest)
    with (output_root / "probe_manifest.jsonl").open("w", encoding="utf-8") as f:
        for c in clips_out:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    with (output_root / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return manifest


def load_codec(codec_root: Path, device: str, dtype: str = "float32"):
    from moss_codecvc.moss_codec import MossCodec

    # The custom tokenizer code is beside the checkpoint under
    # ``vcdata_construction/MOSS-TTS``.  Deriving it from the CodecVC repo
    # parent would point at the wrong project directory.
    moss_root = codec_root.parent / "MOSS-TTS"
    if not moss_root.is_dir():
        raise FileNotFoundError(f"MOSS-TTS remote-code root is missing: {moss_root}")
    return MossCodec(codec_root, moss_root=moss_root, device=device, dtype=dtype)


def encode_worker(manifest: dict[str, Any], output_root: Path, codec_root: Path, worker_id: int, num_workers: int, device: str, max_clips: int | None) -> dict[str, Any]:
    import torch

    clips = manifest["clips"]
    if max_clips is not None:
        clips = clips[:max_clips]
    assigned = [c for i, c in enumerate(clips) if i % num_workers == worker_id]
    worker_dir = output_root / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    result_path = worker_dir / f"worker-{worker_id:02d}-of-{num_workers:02d}.jsonl"
    tmp = result_path.with_suffix(".tmp")
    codec = load_codec(codec_root, device)
    # Residual LFQ exposes one 8-D embedding table per quantizer.
    weight = codec.model.quantizer.quantizers[0].codebook.weight.detach().to(torch.float32)
    rows: list[dict[str, Any]] = []
    with tmp.open("w", encoding="utf-8") as f:
        for clip in assigned:
            encoded = codec.encode_path(clip["audio_path"], n_vq=32)
            codes = encoded["codes"][:, 0].to(torch.long)
            if codes.numel() == 0:
                raise RuntimeError(f"empty codebook-0 sequence: {clip['clip_id']}")
            if int(encoded["n_vq"]) != 32:
                raise RuntimeError(f"expected 32 quantizers, got {encoded['n_vq']} for {clip['clip_id']}")
            if int(codes.min().item()) < 0 or int(codes.max().item()) >= 1024:
                raise RuntimeError(f"codebook-0 token outside [0,1023]: {clip['clip_id']}")
            emb = weight[codes]
            mean = emb.mean(dim=0)
            mean = mean / mean.norm().clamp_min(1e-12)
            row = {
                "clip_id": clip["clip_id"],
                "language": clip["language"],
                "group_id": clip["group_id"],
                "role": clip["role"],
                "audio_path": clip["audio_path"],
                "num_frames": int(codes.numel()),
                "code0": codes.tolist(),
                "mean_embedding": mean.cpu().tolist(),
                "embedding_dim": int(weight.shape[1]),
                "codebook_size": int(weight.shape[0]),
                "num_quantizers": int(encoded["n_vq"]),
                "sample_rate": int(encoded["sample_rate"]),
                "frame_rate_hz": 12.5,
                "device": str(codec.device),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
    tmp.replace(result_path)
    return {"worker_id": worker_id, "num_workers": num_workers, "assigned": len(assigned), "path": str(result_path), "status": "completed"}


def levenshtein(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def pair_metric(a: dict[str, Any], b: dict[str, Any], weight: np.ndarray) -> dict[str, float]:
    ca = np.asarray(a["code0"], dtype=np.int64)
    cb = np.asarray(b["code0"], dtype=np.int64)
    ea = weight[ca]
    eb = weight[cb]
    ma = ea.mean(axis=0)
    mb = eb.mean(axis=0)
    mean_cos = float(np.dot(ma, mb) / max(np.linalg.norm(ma) * np.linalg.norm(mb), 1e-12))
    n = min(len(ea), len(eb))
    na = ea[:n] / np.maximum(np.linalg.norm(ea[:n], axis=1, keepdims=True), 1e-12)
    nb = eb[:n] / np.maximum(np.linalg.norm(eb[:n], axis=1, keepdims=True), 1e-12)
    # Deterministic prefix alignment is only a diagnostic.  It is not DTW and
    # must not be described as time-warp aligned for unequal-length clips.
    prefix_cos = float((na * nb).sum(axis=1).mean()) if n else float("nan")
    edit = levenshtein(ca.tolist(), cb.tolist())
    edit_sim = float(1.0 - edit / max(len(ca), len(cb), 1))
    hist_a = np.bincount(ca, minlength=weight.shape[0]).astype(np.float64)
    hist_b = np.bincount(cb, minlength=weight.shape[0]).astype(np.float64)
    hist_cos = float(np.dot(hist_a, hist_b) / max(np.linalg.norm(hist_a) * np.linalg.norm(hist_b), 1e-12))
    return {
        "mean_embedding_cosine": mean_cos,
        "prefix_frame_cosine": prefix_cos,
        "normalized_edit_similarity": edit_sim,
        "token_histogram_cosine": hist_cos,
        "frames_a": float(len(ca)),
        "frames_b": float(len(cb)),
    }


def bootstrap_ci(values: list[float], seed: int = 1234, draws: int = 2000) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = rng.choice(vals, size=(draws, vals.size), replace=True).mean(axis=1)
    return float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def aggregate(manifest: dict[str, Any], output_root: Path, codec_root: Path) -> dict[str, Any]:
    import torch

    result_rows: dict[str, dict[str, Any]] = {}
    worker_files = sorted((output_root / "workers").glob("worker-*-of-*.jsonl"))
    if not worker_files:
        raise RuntimeError("no worker result files found")
    for path in worker_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["clip_id"] in result_rows:
                raise RuntimeError(f"duplicate encoded clip: {row['clip_id']}")
            result_rows[row["clip_id"]] = row
    expected = {c["clip_id"] for c in manifest["clips"]}
    missing = sorted(expected - set(result_rows))
    extra = sorted(set(result_rows) - expected)
    if missing or extra:
        raise RuntimeError(f"worker coverage mismatch: missing={missing[:5]} extra={extra[:5]}")

    # Load the exact codebook table used by workers for metric reproducibility.
    codec = load_codec(codec_root, "cpu")
    weight = codec.model.quantizer.quantizers[0].codebook.weight.detach().to(torch.float32).cpu().numpy()
    clip_map = {c["clip_id"]: c for c in manifest["clips"]}
    metrics: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        a = result_rows[pair["clip_a"]]
        b = result_rows[pair["clip_b"]]
        m = pair_metric(a, b, weight)
        metrics.append({**pair, **m})
    write_json(output_root / "pair_metrics.json", {"schema": "ver3_1_step2_pair_metrics_v1", "rows": metrics})
    with (output_root / "pair_metrics.csv").open("w", encoding="utf-8") as f:
        cols = ["pair_id", "language", "relation", "speaker_a", "speaker_b", "mean_embedding_cosine", "prefix_frame_cosine", "normalized_edit_similarity", "token_histogram_cosine", "frames_a", "frames_b"]
        f.write(",".join(cols) + "\n")
        for row in metrics:
            f.write(",".join(json.dumps(row.get(k, ""), ensure_ascii=False) for k in cols) + "\n")

    summary: dict[str, Any] = {"schema": "ver3_1_step2_summary_v1", "generated_at_utc": utc_now(), "protocol": manifest["protocol"], "languages": {}}
    for lang in ("zh", "en"):
        summary["languages"][lang] = {}
        for relation in ("same_content_cross_speaker", "same_speaker_cross_content"):
            rows = [r for r in metrics if r["language"] == lang and r["relation"] == relation]
            entry: dict[str, Any] = {"n": len(rows), "metrics": {}}
            for key in ("mean_embedding_cosine", "prefix_frame_cosine", "normalized_edit_similarity", "token_histogram_cosine"):
                vals = [float(r[key]) for r in rows]
                ci = bootstrap_ci(vals)
                entry["metrics"][key] = {"mean": float(np.mean(vals)) if vals else float("nan"), "ci95": [ci[0], ci[1]]}
            summary["languages"][lang][relation] = entry

    zh_same = summary["languages"]["zh"]["same_content_cross_speaker"]["metrics"]["mean_embedding_cosine"]["mean"]
    zh_cross = summary["languages"]["zh"]["same_speaker_cross_content"]["metrics"]["mean_embedding_cosine"]["mean"]
    zh_n_same = summary["languages"]["zh"]["same_content_cross_speaker"]["n"]
    zh_n_cross = summary["languages"]["zh"]["same_speaker_cross_content"]["n"]
    if zh_n_same < 5 or zh_n_cross < 5 or not (math.isfinite(zh_same) and math.isfinite(zh_cross)):
        path = "A"
        reason = "insufficient Chinese primary pairs; conservative WavLM BNF path"
    elif zh_same > 0.7 and abs(zh_same - zh_cross) < 0.05:
        path = "B"
        reason = "Chinese LFQ codebook-0 passes speaker-invariance gate"
    else:
        path = "A"
        reason = "Chinese LFQ codebook-0 fails the preregistered invariance gate"
    decision = {
        "path": path,
        "gate_language": "zh",
        "same_content_cross_speaker_mean_cosine": zh_same,
        "same_speaker_cross_content_mean_cosine": zh_cross,
        "absolute_difference": abs(zh_same - zh_cross) if math.isfinite(zh_same) and math.isfinite(zh_cross) else None,
        "thresholds": {"same_content_cosine_gt": 0.7, "absolute_difference_lt": 0.05},
        "reason": reason,
        "english_participates_in_gate": False,
        "generated_at_utc": utc_now(),
    }
    summary["decision"] = decision
    write_json(output_root / "SUMMARY.json", summary)
    write_json(output_root / "PATH_DECISION.json", decision)

    plot_path = output_root / "codebook0_probe_distributions.png"
    plot_backend = "matplotlib"
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, lang in zip(axes, ("zh", "en")):
            for relation, color in (("same_content_cross_speaker", "tab:blue"), ("same_speaker_cross_content", "tab:orange")):
                vals = [r["mean_embedding_cosine"] for r in metrics if r["language"] == lang and r["relation"] == relation]
                if vals:
                    ax.hist(vals, bins=min(12, max(4, len(vals))), alpha=0.6, label=relation, color=color)
            ax.set_title(f"{lang} codebook-0 mean embedding cosine")
            ax.set_xlabel("cosine")
            ax.set_ylabel("pairs")
            ax.legend(fontsize=8)
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except Exception:
        # The tokenizer environment may have a NumPy-1.x matplotlib build.
        # Render a dependency-light PNG with Pillow so the required
        # distribution artifact is still produced deterministically.
        from PIL import Image, ImageDraw

        plot_backend = "pillow"
        width, height = 1200, 520
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        colors = {"same_content_cross_speaker": (52, 101, 164), "same_speaker_cross_content": (230, 126, 34)}
        titles = {"zh": "zh (primary gate)", "en": "en (proxy exploratory)"}
        for panel, lang in enumerate(("zh", "en")):
            left = 55 + panel * 570
            top, right, bottom = 55, left + 520, 445
            draw.rectangle((left, top, right, bottom), outline=(40, 40, 40), width=2)
            draw.text((left + 10, 18), f"{titles[lang]} LFQ codebook-0 mean cosine", fill=(0, 0, 0))
            for tick in range(0, 11, 2):
                x = left + int((right - left) * tick / 10)
                draw.line((x, bottom, x, bottom + 6), fill=(40, 40, 40), width=1)
                draw.text((x - 8, bottom + 10), f"{tick / 10:.1f}", fill=(0, 0, 0))
            for idx, relation in enumerate(("same_content_cross_speaker", "same_speaker_cross_content")):
                vals = [r["mean_embedding_cosine"] for r in metrics if r["language"] == lang and r["relation"] == relation]
                if not vals:
                    continue
                counts = np.histogram(vals, bins=10, range=(0.0, 1.0))[0]
                max_count = max(int(counts.max()), 1)
                bar_w = max(5, (right - left) // 10 // 2 - 2)
                for b, count in enumerate(counts):
                    x0 = left + int((right - left) * b / 10) + 2 + idx * bar_w
                    x1 = x0 + bar_w
                    y1 = bottom - int((bottom - top - 15) * int(count) / max_count)
                    draw.rectangle((x0, y1, x1, bottom), fill=colors[relation])
                legend_x = left + 15 + idx * 245
                draw.rectangle((legend_x, top + 15, legend_x + 14, top + 29), fill=colors[relation])
                draw.text((legend_x + 20, top + 13), relation, fill=(0, 0, 0))
        image.save(plot_path)
    summary["plot_backend"] = plot_backend
    summary.pop("plot_warning", None)
    write_json(output_root / "SUMMARY.json", summary)

    report = [
        "# Batch-45 Step 2 — LFQ codebook-0 speaker-invariance probe",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        "## Protocol caveat",
        "",
        "The official package does not contain 20 speakers × 5 clips. "
        "This run uses 20 complete groups with up to 3 clips (prompt + 2 targets). "
        "Chinese prefix groups are the primary true-speaker gate; English groups "
        "are prompt-clip proxies and are exploratory only.",
        "",
        "The tokenizer is `MossAudioTokenizerResidualLFQ`; metrics below refer to "
        "LFQ codebook 0 (8-D embedding), not a classical RVQ claim.",
        "",
        "## Aggregate metrics",
        "",
        "| Language | Relation | n | Mean embedding cosine | 95% CI | Prefix frame cosine | Edit similarity | |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for lang in ("zh", "en"):
        for relation in ("same_content_cross_speaker", "same_speaker_cross_content"):
            e = summary["languages"][lang][relation]
            m = e["metrics"]
            report.append(
                f"| {lang} | {relation} | {e['n']} | {m['mean_embedding_cosine']['mean']:.4f} | "
                f"[{m['mean_embedding_cosine']['ci95'][0]:.4f}, {m['mean_embedding_cosine']['ci95'][1]:.4f}] | "
                f"{m['prefix_frame_cosine']['mean']:.4f} | {m['normalized_edit_similarity']['mean']:.4f} |"
            )
    report.extend(
        [
            "",
            "## Path decision",
            "",
            f"- **Path {decision['path']}**: {decision['reason']}",
            f"- Chinese same-content/cross-speaker mean: `{zh_same:.6f}`",
            f"- Chinese same-speaker/cross-content mean: `{zh_cross:.6f}`",
            f"- Absolute difference: `{decision['absolute_difference']}`",
            "- English does not participate in the A/B gate because the archive lacks speaker metadata.",
            f"- Distribution plot: `{plot_path.name}` (backend: `{plot_backend}`).",
            "",
        ]
    )
    (output_root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    completed = {
        "schema": "ver3_1_step2_completion_v1",
        "status": "completed",
        "generated_at_utc": utc_now(),
        "output_root": str(output_root.resolve()),
        "clip_count": len(manifest["clips"]),
        "pair_count": len(metrics),
        "decision": decision,
        "artifacts": {
            name: {"path": str((output_root / name).resolve()), "sha256": sha256_file(output_root / name)}
            for name in ("probe_manifest.json", "pairs.jsonl", "pair_metrics.json", "pair_metrics.csv", "SUMMARY.json", "PATH_DECISION.json", "REPORT.md", "codebook0_probe_distributions.png")
        },
    }
    write_json(output_root / "COMPLETED.json", completed)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--codec-root", type=Path, default=DEFAULT_CODEC_ROOT)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--encode", action="store_true")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    p.add_argument("--max-clips", type=int, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_workers < 1 or not (0 <= args.worker_id < args.num_workers):
        raise SystemExit("invalid worker-id/num-workers")
    output_root = args.output_root.resolve()
    manifest_path = output_root / "probe_manifest.json"
    if not manifest_path.is_file():
        manifest = materialize_manifest(args.data_root.resolve(), output_root)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "output_root": str(output_root), "clips": len(manifest["clips"]), "pairs": len(manifest["pairs"])}, ensure_ascii=False))
        return
    if args.encode:
        result = encode_worker(manifest, output_root, args.codec_root.resolve(), args.worker_id, args.num_workers, args.device, args.max_clips)
        print(json.dumps(result, ensure_ascii=False))
        return
    summary = aggregate(manifest, output_root, args.codec_root.resolve())
    print(json.dumps({"status": "completed", "path": summary["decision"]["path"], "output_root": str(output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
