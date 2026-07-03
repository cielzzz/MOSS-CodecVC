from __future__ import annotations

from dataclasses import dataclass

import torch


TEXT_OR_OTHER = 0
SOURCE_CODEC = 1
REF_CODEC = 2
TARGET_CODEC = 3

ROLE_NAMES = {
    TEXT_OR_OTHER: "TEXT_OR_OTHER",
    SOURCE_CODEC: "SOURCE_CODEC",
    REF_CODEC: "REF_CODEC",
    TARGET_CODEC: "TARGET_CODEC",
}

NUM_ROLES = len(ROLE_NAMES)


@dataclass(frozen=True)
class RoleCounts:
    text_or_other: int
    source_codec: int
    ref_codec: int
    target_codec: int

    def as_dict(self) -> dict[str, int]:
        return {
            "TEXT_OR_OTHER": self.text_or_other,
            "SOURCE_CODEC": self.source_codec,
            "REF_CODEC": self.ref_codec,
            "TARGET_CODEC": self.target_codec,
        }


def build_role_ids(
    shape: tuple[int, int] | torch.Size,
    *,
    source_positions: torch.Tensor | None = None,
    ref_positions: torch.Tensor | None = None,
    target_positions: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build [B, T] role ids from span masks.

    Later assignments intentionally override earlier ones. In training,
    TARGET_CODEC should win over prompt roles because labels define the
    supervised assistant span.
    """

    if len(shape) != 2:
        raise ValueError(f"role_ids shape must be [B, T], got {tuple(shape)}")
    role_ids = torch.full(tuple(shape), TEXT_OR_OTHER, dtype=torch.long, device=device)
    for positions, role in (
        (source_positions, SOURCE_CODEC),
        (ref_positions, REF_CODEC),
        (target_positions, TARGET_CODEC),
    ):
        if positions is None:
            continue
        if tuple(positions.shape) != tuple(shape):
            raise ValueError(f"position mask shape {tuple(positions.shape)} does not match role shape {tuple(shape)}")
        role_ids[positions.to(device=role_ids.device).bool()] = role
    return role_ids


def count_roles(role_ids: torch.Tensor) -> RoleCounts:
    if role_ids.dim() != 2:
        raise ValueError(f"role_ids must be [B, T], got {tuple(role_ids.shape)}")
    values = role_ids.detach()
    return RoleCounts(
        text_or_other=int((values == TEXT_OR_OTHER).sum().item()),
        source_codec=int((values == SOURCE_CODEC).sum().item()),
        ref_codec=int((values == REF_CODEC).sum().item()),
        target_codec=int((values == TARGET_CODEC).sum().item()),
    )


def infer_prompt_role_ids_from_audio_spans(input_ids: torch.Tensor, *, audio_pad_code: int) -> torch.Tensor:
    """Infer SOURCE/REF prompt roles from the first two contiguous audio spans.

    This is a project-local compatibility shim for upstream processors that do
    not expose exact span metadata. It marks only prompt audio spans; target
    positions should be applied separately from labels or generation state.
    """

    if input_ids.dim() != 3:
        raise ValueError(f"input_ids must be [B, T, C], got {tuple(input_ids.shape)}")
    active = (input_ids[..., 1:] != int(audio_pad_code)).any(dim=-1)
    role_ids = torch.full(input_ids.shape[:2], TEXT_OR_OTHER, dtype=torch.long, device=input_ids.device)
    for batch_idx in range(input_ids.shape[0]):
        spans: list[tuple[int, int]] = []
        start: int | None = None
        row = active[batch_idx].detach().cpu().bool().tolist()
        for idx, is_active in enumerate(row):
            if is_active and start is None:
                start = idx
            if start is not None and (not is_active or idx == len(row) - 1):
                end = idx if not is_active else idx + 1
                spans.append((start, end))
                start = None
        if spans:
            start, end = spans[0]
            role_ids[batch_idx, start:end] = SOURCE_CODEC
        if len(spans) > 1:
            start, end = spans[1]
            role_ids[batch_idx, start:end] = REF_CODEC
    return role_ids
