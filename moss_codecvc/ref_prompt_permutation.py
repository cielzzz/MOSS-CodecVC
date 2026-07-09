from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass(frozen=True)
class RefPromptCodecPermutationStats:
    enabled: int
    source_frames: int
    prompt_frames: int
    start: int
    shuffled: int
    mode: str = "shuffle"
    block_frames: int = 0
    requested_frames: int = 0
    bootstrap: str = "off"
    bootstrap_used: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _make_cpu_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def normalize_ref_prompt_permutation_mode(mode: str | None) -> str:
    value = str(mode or "shuffle").strip().lower().replace("-", "_")
    aliases = {
        "permute": "shuffle",
        "permutation": "shuffle",
        "frame_shuffle": "shuffle",
        "full_shuffle": "shuffle",
        "ordered": "contiguous",
        "continuous": "contiguous",
        "no_shuffle": "contiguous",
        "block": "block_shuffle",
        "block_permute": "block_shuffle",
        "block_permutation": "block_shuffle",
    }
    value = aliases.get(value, value)
    if value not in {"shuffle", "contiguous", "block_shuffle"}:
        raise ValueError(f"unsupported ref prompt permutation mode: {mode!r}")
    return value


def normalize_ref_prompt_bootstrap(bootstrap: str | bool | None) -> str:
    if isinstance(bootstrap, bool):
        return "block" if bootstrap else "off"
    value = str(bootstrap or "off").strip().lower().replace("-", "_")
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "none": "off",
        "disabled": "off",
        "1": "block",
        "true": "block",
        "yes": "block",
        "block_bootstrap": "block",
    }
    value = aliases.get(value, value)
    if value not in {"off", "block"}:
        raise ValueError(f"unsupported ref prompt bootstrap mode: {bootstrap!r}")
    return value


def _sample_span(
    timbre_ref_codes: torch.Tensor,
    *,
    min_seconds: float,
    max_seconds: float,
    frame_rate: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, int, int]:
    total_frames = int(timbre_ref_codes.shape[0])
    min_frames = max(1, int(round(max(0.0, float(min_seconds)) * max(1.0e-6, float(frame_rate)))))
    max_frames = max(min_frames, int(round(max(float(min_seconds), float(max_seconds)) * max(1.0e-6, float(frame_rate)))))
    min_frames = min(min_frames, total_frames)
    max_frames = min(max_frames, total_frames)
    if max_frames <= min_frames:
        take = min_frames
    else:
        take = int(torch.randint(min_frames, max_frames + 1, (1,), generator=generator).item())
    if total_frames == take:
        start = 0
    else:
        start = int(torch.randint(0, total_frames - take + 1, (1,), generator=generator).item())
    return timbre_ref_codes[start : start + take].clone(), start, take


def _requested_frame_count(
    *,
    total_frames: int,
    min_seconds: float,
    max_seconds: float,
    frame_rate: float,
    generator: torch.Generator | None,
) -> int:
    min_frames = max(1, int(round(max(0.0, float(min_seconds)) * max(1.0e-6, float(frame_rate)))))
    max_frames = max(min_frames, int(round(max(float(min_seconds), float(max_seconds)) * max(1.0e-6, float(frame_rate)))))
    if max_frames <= min_frames:
        return min_frames
    return int(torch.randint(min_frames, max_frames + 1, (1,), generator=generator).item())


def _block_bootstrap_span(
    timbre_ref_codes: torch.Tensor,
    *,
    take: int,
    block_frames: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    total_frames = int(timbre_ref_codes.shape[0])
    block_frames = max(1, int(block_frames))
    blocks = [
        timbre_ref_codes[start : min(start + block_frames, total_frames)]
        for start in range(0, total_frames, block_frames)
    ]
    if not blocks:
        return timbre_ref_codes[:0].clone()
    pieces = []
    produced = 0
    while produced < int(take):
        idx = int(torch.randint(0, len(blocks), (1,), generator=generator).item())
        block = blocks[idx]
        remaining = int(take) - produced
        piece = block[:remaining]
        pieces.append(piece)
        produced += int(piece.shape[0])
    return torch.cat(pieces, dim=0).clone()


def _shuffle_order(length: int, *, generator: torch.Generator | None) -> tuple[torch.Tensor, int]:
    order = torch.randperm(length, generator=generator)
    shuffled = int(length > 1 and not bool(torch.equal(order, torch.arange(length))))
    if length > 1 and not shuffled:
        order = torch.roll(order, shifts=1)
        shuffled = 1
    return order, shuffled


def _block_shuffle_order(
    length: int,
    *,
    block_frames: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, int]:
    block_frames = max(1, int(block_frames))
    blocks = [torch.arange(start, min(start + block_frames, length)) for start in range(0, length, block_frames)]
    block_count = len(blocks)
    if block_count <= 1:
        return torch.arange(length), 0
    block_order, shuffled = _shuffle_order(block_count, generator=generator)
    order = torch.cat([blocks[int(idx)] for idx in block_order.tolist()], dim=0)
    return order, int(shuffled)


def permute_ref_prompt_codes(
    timbre_ref_codes: torch.Tensor,
    *,
    enabled: bool,
    min_seconds: float = 2.0,
    max_seconds: float = 4.0,
    frame_rate: float = 12.5,
    seed: int | None = None,
    mode: str = "shuffle",
    block_seconds: float = 0.4,
    bootstrap: str | bool | None = None,
) -> tuple[torch.Tensor, RefPromptCodecPermutationStats]:
    """Sample a C_ref span and optionally reorder it for the AR prompt.

    The input and output shapes are [T, C]. A fixed seed is intended for
    train-vs-infer consistency checks; normal training should pass seed=None.
    """

    if timbre_ref_codes.dim() != 2:
        raise ValueError(f"timbre_ref_codes must be [T, C], got {tuple(timbre_ref_codes.shape)}")
    total_frames = int(timbre_ref_codes.shape[0])
    if not bool(enabled):
        return timbre_ref_codes, RefPromptCodecPermutationStats(
            enabled=0,
            source_frames=total_frames,
            prompt_frames=total_frames,
            start=0,
            shuffled=0,
            mode="off",
            block_frames=0,
            requested_frames=total_frames,
            bootstrap="off",
            bootstrap_used=0,
        )
    if total_frames <= 1:
        return timbre_ref_codes.clone(), RefPromptCodecPermutationStats(
            enabled=1,
            source_frames=total_frames,
            prompt_frames=total_frames,
            start=0,
            shuffled=0,
            mode=normalize_ref_prompt_permutation_mode(mode),
            block_frames=0,
            requested_frames=total_frames,
            bootstrap=normalize_ref_prompt_bootstrap(bootstrap),
            bootstrap_used=0,
        )

    normalized_mode = normalize_ref_prompt_permutation_mode(mode)
    generator = _make_cpu_generator(seed)
    bootstrap_mode = normalize_ref_prompt_bootstrap(bootstrap)
    requested_take = _requested_frame_count(
        total_frames=total_frames,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        frame_rate=frame_rate,
        generator=generator,
    )
    block_frames = max(1, int(round(max(0.0, float(block_seconds)) * max(1.0e-6, float(frame_rate)))))
    bootstrap_used = int(bootstrap_mode == "block" and requested_take > total_frames)
    if bootstrap_used:
        sliced = _block_bootstrap_span(
            timbre_ref_codes,
            take=requested_take,
            block_frames=block_frames,
            generator=generator,
        )
        start = 0
        take = int(requested_take)
    else:
        sliced, start, take = _sample_span(
            timbre_ref_codes,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            frame_rate=frame_rate,
            generator=generator,
        )
    if normalized_mode == "contiguous":
        order = torch.arange(take)
        shuffled = 0
    elif normalized_mode == "block_shuffle":
        order, shuffled = _block_shuffle_order(take, block_frames=block_frames, generator=generator)
    else:
        order, shuffled = _shuffle_order(take, generator=generator)
    order = order.to(device=sliced.device)
    prompt = sliced.index_select(0, order)
    return prompt, RefPromptCodecPermutationStats(
        enabled=1,
        source_frames=total_frames,
        prompt_frames=take,
        start=start,
        shuffled=shuffled,
        mode=normalized_mode,
        block_frames=block_frames,
        requested_frames=int(requested_take),
        bootstrap=bootstrap_mode,
        bootstrap_used=bootstrap_used,
    )
