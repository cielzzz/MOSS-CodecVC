from __future__ import annotations

from collections.abc import Iterable
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


def ref_speaker_prompt_slot_positions(
    input_ids: torch.Tensor,
    *,
    audio_start_token_id: int,
    audio_end_token_id: int,
    audio_gen_slot_token_id: int | Iterable[int],
    token_count: int,
    occurrence: int = 2,
) -> torch.Tensor:
    """Locate K prompt-slot positions in the Nth audio block.

    The slot uses a fake S2 audio placeholder whose audio channels may all be
    pad codes, so it cannot be found from audio activity. We instead use the
    text channel audio block markers and select the first K gen-slot tokens.
    """

    if input_ids.dim() != 3:
        raise ValueError(f"input_ids must be [B, T, C], got {tuple(input_ids.shape)}")
    positions = torch.zeros(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)
    k = int(token_count)
    if k <= 0:
        return positions
    occurrence = max(1, int(occurrence))
    if isinstance(audio_gen_slot_token_id, Iterable) and not isinstance(audio_gen_slot_token_id, (str, bytes)):
        slot_token_ids = [int(value) for value in audio_gen_slot_token_id]
    else:
        slot_token_ids = [int(audio_gen_slot_token_id)]
    text_tokens = input_ids[..., 0]
    for batch_idx in range(int(text_tokens.shape[0])):
        row = text_tokens[batch_idx]
        starts = torch.nonzero(row == int(audio_start_token_id), as_tuple=False).flatten()
        if int(starts.numel()) < occurrence:
            continue
        start = int(starts[occurrence - 1].item())
        end_candidates = torch.nonzero(row[start + 1 :] == int(audio_end_token_id), as_tuple=False).flatten()
        end = int(row.shape[0]) if int(end_candidates.numel()) == 0 else start + 1 + int(end_candidates[0].item())
        block = row[start : end + 1]
        slot_mask = torch.zeros_like(block, dtype=torch.bool)
        for slot_token_id in slot_token_ids:
            slot_mask |= block == int(slot_token_id)
        gen_idxs = torch.nonzero(slot_mask, as_tuple=False).flatten()
        if int(gen_idxs.numel()) == 0:
            continue
        selected = gen_idxs[:k] + start
        positions[batch_idx, selected.to(device=positions.device)] = True
    return positions
