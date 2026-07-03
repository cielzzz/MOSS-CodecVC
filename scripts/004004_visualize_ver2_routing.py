#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.roles import ROLE_NAMES


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize Ver2 role routing and target-head gates.")
    ap.add_argument("--adapter-dir", required=True, help="Directory containing timbre_memory_adapter.pt")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dpi", type=int, default=160)
    return ap.parse_args()


def load_state(adapter_dir: Path) -> dict:
    path = adapter_dir / "timbre_memory_adapter.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing adapter state: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sigmoid_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(value.detach().float().cpu())


def has_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def color_cell(value: float, scheme: str = "viridis") -> tuple[int, int, int]:
    value = max(0.0, min(1.0, float(value)))
    if scheme == "magma":
        return (
            int(30 + 225 * value),
            int(8 + 120 * (value ** 1.4)),
            int(60 + 80 * (1.0 - value)),
        )
    return (
        int(68 + 185 * value),
        int(20 + 205 * value),
        int(85 + 70 * (1.0 - value)),
    )


def save_heatmap_pil(
    data: torch.Tensor,
    output_path: Path,
    *,
    row_labels: list[str],
    title: str,
    scheme: str = "viridis",
) -> None:
    from PIL import Image, ImageDraw

    cell_w, cell_h = 28, 28
    left, top, right, bottom = 150, 36, 20, 42
    width = left + data.shape[1] * cell_w + right
    height = top + data.shape[0] * cell_h + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), title, fill="black")
    for row in range(data.shape[0]):
        draw.text((8, top + row * cell_h + 8), row_labels[row], fill="black")
        for col in range(data.shape[1]):
            x0 = left + col * cell_w
            y0 = top + row * cell_h
            draw.rectangle(
                [x0, y0, x0 + cell_w - 1, y0 + cell_h - 1],
                fill=color_cell(float(data[row, col]), scheme=scheme),
            )
    for col in range(data.shape[1]):
        draw.text((left + col * cell_w + 8, top + data.shape[0] * cell_h + 6), str(col), fill="black")
    image.save(output_path)


def save_bar_pil(prosody: torch.Tensor, timbre: torch.Tensor, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1100, 360
    left, top, bottom = 50, 38, 42
    plot_h = height - top - bottom
    plot_w = width - left - 20
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "Per-codebook target head gates", fill="black")
    n = int(prosody.numel())
    group_w = plot_w / max(1, n)
    bar_w = max(2, int(group_w * 0.35))
    for idx in range(n):
        x = int(left + idx * group_w + group_w * 0.15)
        p_h = int(float(prosody[idx]) * plot_h)
        t_h = int(float(timbre[idx]) * plot_h)
        draw.rectangle([x, top + plot_h - p_h, x + bar_w, top + plot_h], fill=(55, 126, 184))
        draw.rectangle([x + bar_w + 2, top + plot_h - t_h, x + 2 * bar_w + 2, top + plot_h], fill=(228, 26, 28))
        if idx % 2 == 0:
            draw.text((int(left + idx * group_w), top + plot_h + 5), str(idx), fill="black")
    draw.line([left, top, left, top + plot_h, left + plot_w, top + plot_h], fill="black")
    draw.text((left + 10, height - 20), "blue=prosody  red=timbre", fill="black")
    image.save(output_path)


def save_role_heatmap(role_gates: torch.Tensor, output_dir: Path, dpi: int) -> None:
    if not has_matplotlib():
        save_heatmap_pil(
            role_gates,
            output_dir / "role_gate_heatmap.png",
            row_labels=[ROLE_NAMES.get(idx, str(idx)) for idx in range(role_gates.shape[0])],
            title="RoleCodecRouter gates",
            scheme="viridis",
        )
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.2))
    image = ax.imshow(role_gates.numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(range(role_gates.shape[0]))
    ax.set_yticklabels([ROLE_NAMES.get(idx, str(idx)) for idx in range(role_gates.shape[0])])
    ax.set_xticks(range(role_gates.shape[1]))
    ax.set_xticklabels([str(idx) for idx in range(role_gates.shape[1])], fontsize=7)
    ax.set_xlabel("RVQ codebook")
    ax.set_title("RoleCodecRouter gates")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / "role_gate_heatmap.png", dpi=dpi)
    plt.close(fig)


def save_head_bar(prosody: torch.Tensor, timbre: torch.Tensor, output_dir: Path, dpi: int) -> None:
    if not has_matplotlib():
        save_bar_pil(prosody, timbre, output_dir / "target_head_gate_bar.png")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = torch.arange(prosody.numel()).numpy()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    width = 0.42
    ax.bar(x - width / 2, prosody.numpy(), width=width, label="prosody")
    ax.bar(x + width / 2, timbre.numpy(), width=width, label="timbre")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(idx) for idx in x], fontsize=7)
    ax.set_xlabel("RVQ audio head")
    ax.set_ylabel("gate")
    ax.set_title("Per-codebook target head gates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "target_head_gate_bar.png", dpi=dpi)
    plt.close(fig)


def save_head_heatmap(prosody: torch.Tensor, timbre: torch.Tensor, output_dir: Path, dpi: int) -> None:
    if not has_matplotlib():
        save_heatmap_pil(
            torch.stack([prosody, timbre], dim=0),
            output_dir / "target_head_gate_heatmap.png",
            row_labels=["prosody", "timbre"],
            title="Target-head routing gates",
            scheme="magma",
        )
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = torch.stack([prosody, timbre], dim=0)
    fig, ax = plt.subplots(figsize=(12, 2.4))
    image = ax.imshow(data.numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["prosody", "timbre"])
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([str(idx) for idx in range(data.shape[1])], fontsize=7)
    ax.set_xlabel("RVQ audio head")
    ax.set_title("Target-head routing gates")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / "target_head_gate_heatmap.png", dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(adapter_dir)

    report: dict[str, object] = {"adapter_dir": str(adapter_dir.resolve(strict=False))}
    role_state = state.get("role_router")
    if role_state and "gate_logits" in role_state:
        role_gates = sigmoid_tensor(role_state["gate_logits"])
        save_role_heatmap(role_gates, output_dir, args.dpi)
        report["role_gates"] = role_gates.tolist()
        report["role_gate_mean"] = float(role_gates.mean().item())
    else:
        report["role_gates"] = None

    head_state = state.get("target_head_router")
    if head_state and "prosody_gate_logits" in head_state and "timbre_gate_logits" in head_state:
        prosody = sigmoid_tensor(head_state["prosody_gate_logits"])
        timbre = sigmoid_tensor(head_state["timbre_gate_logits"])
        save_head_bar(prosody, timbre, output_dir, args.dpi)
        save_head_heatmap(prosody, timbre, output_dir, args.dpi)
        report["prosody_head_gates"] = prosody.tolist()
        report["timbre_head_gates"] = timbre.tolist()
        report["prosody_head_gate_mean"] = float(prosody.mean().item())
        report["timbre_head_gate_mean"] = float(timbre.mean().item())
        report["head_gate_gap"] = float((prosody - timbre).abs().mean().item())
    else:
        report["target_head_gates"] = None

    report_path = output_dir / "routing_gates.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote routing visualizations -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
