#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch


PALETTE = {
    "source": "#4C78A8",
    "ref": "#F58518",
    "target": "#54A24B",
    "generated": "#E45756",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize MOSS-CodecVC codec-token intermediate states.")
    ap.add_argument("--jsonl", required=True, help="SFT JSONL containing reference_audio_codes and audio_codes.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--row-index", type=int, default=0)
    ap.add_argument("--sample-id", default="", help="Optional sample_id lookup. Overrides --row-index when set.")
    ap.add_argument("--generated-codec", default="", help="Optional generated codec tensor/list: .pt/.pth/.npy/.json.")
    ap.add_argument("--max-frames", type=int, default=800, help="Subsample long heatmaps for readable figures.")
    ap.add_argument(
        "--codec-frame-rate",
        type=float,
        default=50.0,
        help="Codec frames per second used for duration annotations. MOSS tokenizer is typically around 50 Hz.",
    )
    ap.add_argument("--dpi", type=int, default=160)
    return ap.parse_args()


def has_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def read_record(path: Path, *, row_index: int, sample_id: str = "") -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            record = json.loads(line)
            if sample_id:
                if str(record.get("sample_id") or "") == sample_id:
                    return record
                continue
            if idx == row_index:
                return record
    if sample_id:
        raise ValueError(f"sample_id not found: {sample_id}")
    raise IndexError(f"row_index {row_index} out of range for {path}")


def codec_tensor(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be [T, n_vq], got {tuple(tensor.shape)}")
    if tensor.shape[1] != 32:
        raise ValueError(f"{name} expected n_vq=32, got shape {tuple(tensor.shape)}")
    return tensor.contiguous()


def load_generated_codec(path: str) -> torch.Tensor | None:
    if not path:
        return None
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix in {".pt", ".pth"}:
        try:
            obj = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(p, map_location="cpu")
    elif suffix == ".npy":
        import numpy as np

        obj = np.load(p)
    else:
        obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        for key in ("generated_codec", "generated_codes", "audio_codes", "codes", "codec"):
            if key in obj:
                obj = obj[key]
                break
    return codec_tensor(obj, name=str(p))


def extract_triplet(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    refs = record.get("reference_audio_codes") or record.get("ref_audio_codes")
    if not isinstance(refs, list) or len(refs) < 2:
        raise ValueError("record must contain reference_audio_codes=[source_codes,timbre_ref_codes]")
    return {
        "source": codec_tensor(refs[0], name="source"),
        "ref": codec_tensor(refs[1], name="ref"),
        "target": codec_tensor(record.get("audio_codes"), name="target"),
    }


def downsample_time(codes: torch.Tensor, max_frames: int) -> torch.Tensor:
    if max_frames <= 0 or codes.shape[0] <= max_frames:
        return codes
    index = torch.linspace(0, codes.shape[0] - 1, steps=max_frames).round().long()
    return codes[index]


def normalized(codes: torch.Tensor) -> torch.Tensor:
    return codes.float() / max(1.0, float(codes.max().item()))


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def save_heatmap(codes: torch.Tensor, path: Path, *, title: str, max_frames: int, dpi: int) -> None:
    shown = downsample_time(codes, max_frames)
    if has_matplotlib():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_w = max(8.0, min(18.0, shown.shape[0] / 60.0))
        fig, ax = plt.subplots(figsize=(fig_w, 4.0))
        image = ax.imshow(normalized(shown).transpose(0, 1).numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(f"{title} codec tokens [T={codes.shape[0]}, n_vq={codes.shape[1]}]")
        ax.set_xlabel("time frame")
        ax.set_ylabel("RVQ codebook")
        ax.set_yticks([0, 7, 15, 23, 31])
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return
    from PIL import Image

    data = (normalized(shown).transpose(0, 1) * 255).byte().numpy()
    image = Image.fromarray(data, mode="L").resize((shown.shape[0] * 2, shown.shape[1] * 12))
    image.save(path)


def save_overview(codecs: dict[str, torch.Tensor], path: Path, *, max_frames: int, dpi: int) -> None:
    if not has_matplotlib():
        from PIL import Image, ImageDraw, ImageFont

        names = list(codecs)
        panel_w = 1200
        panel_h = 120
        label_w = 130
        title_h = 34
        image = Image.new("RGB", (label_w + panel_w + 20, title_h + panel_h * len(names) + 20), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((12, 10), "Codec triplet overview, subsampled when long", fill="black", font=font)
        for row, name in enumerate(names):
            codes = downsample_time(codecs[name], max_frames)
            data = (normalized(codes).transpose(0, 1) * 255).byte().numpy()
            panel = Image.fromarray(data, mode="L").resize((panel_w, panel_h))
            y0 = title_h + row * panel_h
            image.paste(panel.convert("RGB"), (label_w, y0))
            draw.text((10, y0 + 42), f"{name}\nT={codecs[name].shape[0]}", fill="black", font=font)
        image.save(path)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(codecs)
    fig, axes = plt.subplots(len(names), 1, figsize=(14, max(3.0, 2.3 * len(names))), sharex=False)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        codes = codecs[name]
        shown = downsample_time(codes, max_frames)
        ax.imshow(normalized(shown).transpose(0, 1).numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(f"{name}: T={codes.shape[0]}")
        ax.set_ylabel("RVQ")
        ax.set_yticks([0, 15, 31])
    axes[-1].set_xlabel("time frame, subsampled when long")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_length_alignment(codecs: dict[str, torch.Tensor], path: Path, *, dpi: int) -> None:
    if not has_matplotlib():
        from PIL import Image, ImageDraw, ImageFont

        names = list(codecs)
        lengths = [int(codecs[name].shape[0]) for name in names]
        source_len = max(1, lengths[names.index("source")] if "source" in names else lengths[0])
        ratios = [value / source_len for value in lengths]
        image = Image.new("RGB", (900, 330), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((24, 18), "Codec frame length alignment", fill="black", font=font)
        base_y = 250
        max_ratio = max(1.1, max(ratios) * 1.15)
        scale = 180 / max_ratio
        bar_w = max(40, min(110, int(620 / max(1, len(names)))))
        for idx, (name, length, ratio) in enumerate(zip(names, lengths, ratios)):
            x0 = 110 + idx * int(700 / max(1, len(names)))
            x1 = x0 + bar_w
            y0 = int(base_y - ratio * scale)
            draw.rectangle((x0, y0, x1, base_y), fill=hex_rgb(PALETTE.get(name, "#999999")))
            draw.text((x0 - 8, base_y + 10), name, fill="black", font=font)
            draw.text((x0 - 12, y0 - 32), f"T={length}\n{ratio:.2f}x", fill="black", font=font)
        one_y = int(base_y - 1.0 * scale)
        draw.line((70, one_y, 840, one_y), fill="black", width=1)
        draw.text((30, one_y - 8), "1.0x", fill="black", font=font)
        image.save(path)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(codecs)
    lengths = [int(codecs[name].shape[0]) for name in names]
    source_len = max(1, lengths[names.index("source")] if "source" in names else lengths[0])
    ratios = [value / source_len for value in lengths]
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    bars = ax.bar(names, ratios, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(names)])
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("length / source length")
    ax.set_title("Codec frame length alignment")
    for bar, length, ratio in zip(bars, lengths, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"T={length}\n{ratio:.2f}x", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def synthetic_role_segments(record: dict[str, Any], codecs: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    """Build a schematic role timeline from SFT codec lengths.

    This is intentionally independent from the MOSS-TTS processor. It is meant
    for paper/report figures that show the role layout. The exact packed token
    indices can differ because text, instruction, and special tokens are handled
    by the upstream processor at train time.
    """
    instruction = str(record.get("instruction") or "")
    text = str(record.get("text") or "")
    text_like_chars = len(instruction) + (0 if text == "<NO_TEXT>" else len(text))
    text_len = max(8, min(160, math.ceil(text_like_chars / 6)))
    source_len = int(codecs["source"].shape[0])
    ref_len = int(codecs["ref"].shape[0])
    target_len = int(codecs.get("target", codecs["source"]).shape[0])
    generated_len = int(codecs["generated"].shape[0]) if "generated" in codecs else 0
    segments: list[dict[str, Any]] = [
        {"name": "TEXT_OR_OTHER", "role_id": 0, "start": 0, "end": text_len, "target_mask": False},
        {
            "name": "SOURCE_CODEC",
            "role_id": 1,
            "start": text_len,
            "end": text_len + source_len,
            "target_mask": False,
        },
        {
            "name": "REF_CODEC",
            "role_id": 2,
            "start": text_len + source_len,
            "end": text_len + source_len + ref_len,
            "target_mask": False,
        },
        {
            "name": "TARGET_CODEC",
            "role_id": 3,
            "start": text_len + source_len + ref_len,
            "end": text_len + source_len + ref_len + target_len,
            "target_mask": True,
        },
    ]
    if generated_len > 0:
        segments.append(
            {
                "name": "GENERATED_CODEC",
                "role_id": 3,
                "start": text_len + source_len + ref_len,
                "end": text_len + source_len + ref_len + generated_len,
                "target_mask": True,
                "alternate": True,
            }
        )
    return segments


def save_role_span_target_mask(
    record: dict[str, Any],
    codecs: dict[str, torch.Tensor],
    path: Path,
    *,
    dpi: int,
) -> list[dict[str, Any]]:
    segments = synthetic_role_segments(record, codecs)
    primary_segments = [seg for seg in segments if not seg.get("alternate")]
    total = max(seg["end"] for seg in primary_segments)
    colors = {
        "TEXT_OR_OTHER": "#B8B8B8",
        "SOURCE_CODEC": "#4C78A8",
        "REF_CODEC": "#F58518",
        "TARGET_CODEC": "#54A24B",
        "GENERATED_CODEC": "#E45756",
    }
    if not has_matplotlib():
        from PIL import Image, ImageDraw, ImageFont

        width = 1400
        height = 260
        margin_x = 90
        scale = (width - 2 * margin_x) / max(1, total)
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((margin_x, 18), "Role spans and target mask, schematic from SFT codec lengths", fill="black", font=font)
        draw.text((10, 105), "role_ids", fill="black", font=font)
        draw.text((10, 180), "target_mask", fill="black", font=font)
        for seg in primary_segments:
            x0 = int(margin_x + seg["start"] * scale)
            x1 = int(margin_x + seg["end"] * scale)
            draw.rectangle((x0, 80, x1, 135), fill=colors.get(seg["name"], "#999999"), outline="white")
            label = f"{seg['name']} T={seg['end'] - seg['start']}"
            draw.text((x0 + 4, 100), label, fill="white", font=font)
            if seg["target_mask"]:
                draw.rectangle((x0, 168, x1, 205), fill="#111111")
        generated = next((seg for seg in segments if seg.get("alternate")), None)
        if generated is not None:
            target_start = primary_segments[-1]["start"]
            x0 = int(margin_x + target_start * scale)
            x1 = int(margin_x + generated["end"] * scale)
            draw.rectangle((x0, 218, x1, 242), fill=colors["GENERATED_CODEC"])
            draw.text((x0 + 4, 222), f"GENERATED T={generated['end'] - generated['start']}", fill="white", font=font)
        image.save(path)
        return segments
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.5, 2.9))
    for seg in primary_segments:
        width = seg["end"] - seg["start"]
        ax.add_patch(
            patches.Rectangle(
                (seg["start"], 1.05),
                width,
                0.55,
                facecolor=colors.get(seg["name"], "#999999"),
                edgecolor="white",
                linewidth=0.8,
            )
        )
        label = f"{seg['name']}\nT={width}"
        ax.text(seg["start"] + width / 2, 1.325, label, ha="center", va="center", fontsize=8, color="white")
        if seg["target_mask"]:
            ax.add_patch(
                patches.Rectangle((seg["start"], 0.25), width, 0.35, facecolor="#111111", edgecolor="none", alpha=0.85)
            )
    generated = next((seg for seg in segments if seg.get("alternate")), None)
    if generated is not None:
        gen_width = generated["end"] - generated["start"]
        target_start = primary_segments[-1]["start"]
        ax.add_patch(
            patches.Rectangle(
                (target_start, -0.30),
                gen_width,
                0.28,
                facecolor=colors["GENERATED_CODEC"],
                edgecolor="none",
                alpha=0.9,
            )
        )
        ax.text(
            target_start + gen_width / 2,
            -0.16,
            f"GENERATED T={gen_width}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
    ax.set_xlim(0, max(total, generated["end"] if generated else total))
    ax.set_ylim(-0.45, 1.8)
    ax.set_yticks([0.425, 1.325])
    ax.set_yticklabels(["target_mask", "role_ids"])
    ax.set_xlabel("schematic packed-token time axis")
    ax.set_title("Role spans and target mask, schematic from SFT codec lengths")
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return segments


def save_generated_source_relation(
    codecs: dict[str, torch.Tensor],
    path: Path,
    *,
    codec_frame_rate: float,
    dpi: int,
) -> dict[str, Any]:
    lengths = {name: int(codes.shape[0]) for name, codes in codecs.items()}
    source_len = max(1, lengths.get("source", 1))
    generated_or_target = "generated" if "generated" in lengths else "target"
    compare_len = max(1, lengths.get(generated_or_target, source_len))
    names = ["source", generated_or_target]
    values = [source_len, compare_len]
    stats = {
        "codec_frame_rate": float(codec_frame_rate),
        "source_frames": source_len,
        f"{generated_or_target}_frames": compare_len,
        f"{generated_or_target}_to_source_ratio": compare_len / source_len,
        "source_seconds": source_len / max(codec_frame_rate, 1e-6),
        f"{generated_or_target}_seconds": compare_len / max(codec_frame_rate, 1e-6),
    }
    if not has_matplotlib():
        from PIL import Image, ImageDraw, ImageFont

        width = 760
        height = 300
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((24, 18), f"{generated_or_target} codec length vs source duration", fill="black", font=font)
        max_value = max(values)
        bar_width = 120
        base_y = 235
        scale = 155 / max(1, max_value)
        for idx, (name, value) in enumerate(zip(names, values)):
            x0 = 170 + idx * 230
            x1 = x0 + bar_width
            y0 = int(base_y - value * scale)
            color = "#4C78A8" if name == "source" else ("#E45756" if name == "generated" else "#54A24B")
            draw.rectangle((x0, y0, x1, base_y), fill=color)
            draw.text((x0, base_y + 10), name, fill="black", font=font)
            draw.text((x0 - 5, y0 - 30), f"T={value}  {value / max(codec_frame_rate, 1e-6):.2f}s", fill="black", font=font)
        draw.line((90, base_y, 690, base_y), fill="black", width=1)
        draw.text((300, 52), f"ratio={compare_len / source_len:.3f}x", fill="black", font=font)
        image.save(path)
        return stats
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    bars = ax.bar(names, values, color=["#4C78A8", "#E45756" if generated_or_target == "generated" else "#54A24B"])
    ax.set_ylabel("codec frames")
    ax.set_title(f"{generated_or_target} codec length vs source duration")
    for bar, value in zip(bars, values):
        seconds = value / max(codec_frame_rate, 1e-6)
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"T={value}\n{seconds:.2f}s", ha="center", va="bottom", fontsize=8)
    ax.text(
        0.5,
        0.94,
        f"ratio={compare_len / source_len:.3f}x",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return stats


def token_entropy(values: torch.Tensor) -> float:
    counts = torch.bincount(values.long().flatten(), minlength=1024).float()
    probs = counts[counts > 0] / counts.sum().clamp(min=1.0)
    return float(-(probs * torch.log2(probs)).sum().item())


def codebook_stats(codes: torch.Tensor) -> dict[str, list[float]]:
    unique = []
    entropy = []
    for idx in range(codes.shape[1]):
        col = codes[:, idx]
        unique.append(float(torch.unique(col).numel()))
        entropy.append(token_entropy(col))
    return {"unique_tokens": unique, "entropy_bits": entropy}


def save_codebook_distribution(codecs: dict[str, torch.Tensor], path: Path, *, dpi: int) -> None:
    if not has_matplotlib():
        from PIL import Image, ImageDraw, ImageFont

        image = Image.new("RGB", (1100, 620), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((24, 18), "Per-codebook token distribution", fill="black", font=font)
        panels = [
            ("unique token count", 70, 280, 1024.0),
            ("entropy bits", 350, 560, 10.0),
        ]
        left, right = 80, 1040
        for title, top, bottom, nominal_max in panels:
            draw.rectangle((left, top, right, bottom), outline="#777777")
            draw.text((left, top - 22), title, fill="black", font=font)
            draw.line((left, bottom, right, bottom), fill="#777777")
            draw.line((left, top, left, bottom), fill="#777777")
            draw.text((left - 45, top), f"{nominal_max:g}", fill="black", font=font)
            draw.text((left - 20, bottom - 8), "0", fill="black", font=font)
            for tick in (0, 7, 15, 23, 31):
                x = left + int(tick / 31 * (right - left))
                draw.line((x, bottom, x, bottom + 4), fill="#777777")
                draw.text((x - 7, bottom + 8), str(tick), fill="black", font=font)
        for name, codes in codecs.items():
            stats = codebook_stats(codes)
            color = hex_rgb(PALETTE.get(name, "#999999"))
            for values, (_, top, bottom, nominal_max) in zip(
                (stats["unique_tokens"], stats["entropy_bits"]),
                panels,
            ):
                max_value = max(nominal_max, max(values) if values else nominal_max)
                points = []
                for idx, value in enumerate(values):
                    x = left + int(idx / 31 * (right - left))
                    y = bottom - int(float(value) / max_value * (bottom - top))
                    points.append((x, y))
                if len(points) >= 2:
                    draw.line(points, fill=color, width=2)
                for x, y in points[::4]:
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        legend_x = 820
        legend_y = 22
        for idx, name in enumerate(codecs):
            color = hex_rgb(PALETTE.get(name, "#999999"))
            y = legend_y + idx * 18
            draw.rectangle((legend_x, y, legend_x + 12, y + 10), fill=color)
            draw.text((legend_x + 18, y - 1), name, fill="black", font=font)
        image.save(path)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    x = list(range(32))
    for name, codes in codecs.items():
        stats = codebook_stats(codes)
        axes[0].plot(x, stats["unique_tokens"], marker=".", label=name)
        axes[1].plot(x, stats["entropy_bits"], marker=".", label=name)
    axes[0].set_ylabel("unique token count")
    axes[0].set_title("Per-codebook token distribution")
    axes[1].set_ylabel("entropy bits")
    axes[1].set_xlabel("RVQ codebook")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def repeated_ngram_ratio(codes: torch.Tensor, n: int) -> float:
    if codes.shape[0] < n * 2:
        return 0.0
    grams = [tuple(codes[idx : idx + n].flatten().tolist()) for idx in range(codes.shape[0] - n + 1)]
    if not grams:
        return 0.0
    repeats = sum(1 for idx in range(1, len(grams)) if grams[idx] == grams[idx - 1])
    return repeats / max(1, len(grams) - 1)


def adjacent_frame_repeat_ratio(codes: torch.Tensor) -> float:
    if codes.shape[0] < 2:
        return 0.0
    same = (codes[1:] == codes[:-1]).all(dim=-1).float().mean()
    return float(same.item())


def build_report(
    record: dict[str, Any],
    codecs: dict[str, torch.Tensor],
    *,
    role_segments: list[dict[str, Any]],
    duration_relation: dict[str, Any],
) -> dict[str, Any]:
    lengths = {name: int(codes.shape[0]) for name, codes in codecs.items()}
    source_len = max(1, lengths.get("source", next(iter(lengths.values()))))
    return {
        "sample_id": record.get("sample_id"),
        "mode": record.get("moss_codecvc_mode") or record.get("mode"),
        "text": record.get("text"),
        "lengths": lengths,
        "length_ratio_to_source": {name: value / source_len for name, value in lengths.items()},
        "role_span_target_mask": role_segments,
        "duration_relation": duration_relation,
        "repetition": {
            name: {
                "adjacent_frame_repeat_ratio": adjacent_frame_repeat_ratio(codes),
                "repeated_bigram_ratio": repeated_ngram_ratio(codes, 2),
                "repeated_trigram_ratio": repeated_ngram_ratio(codes, 3),
            }
            for name, codes in codecs.items()
        },
        "codebook_stats": {name: codebook_stats(codes) for name, codes in codecs.items()},
    }


def main() -> int:
    args = parse_args()
    jsonl_path = Path(args.jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    record = read_record(jsonl_path, row_index=args.row_index, sample_id=args.sample_id)
    codecs = extract_triplet(record)
    generated = load_generated_codec(args.generated_codec)
    if generated is not None:
        codecs["generated"] = generated

    for name, codes in codecs.items():
        save_heatmap(codes, output_dir / f"{name}_codec_heatmap.png", title=name, max_frames=args.max_frames, dpi=args.dpi)
    save_overview(codecs, output_dir / "codec_triplet_overview.png", max_frames=args.max_frames, dpi=args.dpi)
    save_length_alignment(codecs, output_dir / "codec_length_alignment.png", dpi=args.dpi)
    save_codebook_distribution(codecs, output_dir / "codec_codebook_distribution.png", dpi=args.dpi)
    role_segments = save_role_span_target_mask(
        record,
        codecs,
        output_dir / "role_span_target_mask_schematic.png",
        dpi=args.dpi,
    )
    duration_relation = save_generated_source_relation(
        codecs,
        output_dir / "generated_vs_source_duration.png",
        codec_frame_rate=args.codec_frame_rate,
        dpi=args.dpi,
    )

    report = build_report(record, codecs, role_segments=role_segments, duration_relation=duration_relation)
    report["jsonl"] = str(jsonl_path.resolve(strict=False))
    report["row_index"] = int(args.row_index)
    report["generated_codec"] = args.generated_codec or None
    (output_dir / "codec_intermediate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote codec intermediate visualizations -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
