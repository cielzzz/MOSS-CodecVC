from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.ver3_1.run_step2_codebook0_probe import (
    choose_pairs,
    levenshtein,
    norm_text,
    pair_metric,
    parse_meta,
)


def _make_official_like_tree(root: Path, language: str = "zh") -> None:
    lang = root / language
    (lang / "prompt-wavs").mkdir(parents=True)
    (lang / "wavs").mkdir(parents=True)
    rows = []
    # Two groups with one prompt and two targets each.  Empty files are enough
    # for parser tests because no codec encoding is performed here.
    for gid, prompt_idx in (("100", "00000001"), ("200", "00000001"), ("300", "00000001")):
        prompt_stem = f"{gid}-{prompt_idx}"
        prompt_rel = f"prompt-wavs/{prompt_stem}.wav"
        (lang / prompt_rel).touch()
        for j, text in enumerate(("same content", f"different {gid}"), 2):
            target_stem = f"{gid}-{j:08d}"
            (lang / "wavs" / f"{target_stem}.wav").touch()
            rows.append(f"{target_stem}|prompt {gid}|{prompt_rel}|{text}")
    (lang / "meta.lst").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_norm_text_and_levenshtein() -> None:
    assert norm_text(" Héllo, 世界! ") == "héllo世界"
    assert levenshtein([1, 2, 3], [1, 4]) == 2


def test_parse_meta_and_choose_pairs(tmp_path: Path) -> None:
    _make_official_like_tree(tmp_path)
    clips, stats = parse_meta(tmp_path, "zh")
    assert stats["rows"] == 6
    assert stats["complete_groups"] == 3
    assert len(clips) == 9
    pairs, required, selected = choose_pairs(clips, "zh", max_groups=3, max_content_pairs=3)
    assert len(required) == 9
    assert selected == ["100", "200", "300"]
    assert any(p["relation"] == "same_speaker_cross_content" for p in pairs)


def test_pair_metric_is_finite() -> None:
    weight = np.eye(8, dtype=np.float64)
    a = {"code0": [0, 1, 2]}
    b = {"code0": [0, 1, 3]}
    out = pair_metric(a, b, weight)
    assert 0.0 <= out["normalized_edit_similarity"] <= 1.0
    assert np.isfinite(out["mean_embedding_cosine"])
    assert np.isfinite(out["prefix_frame_cosine"])
