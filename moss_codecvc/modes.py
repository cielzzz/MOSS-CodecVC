from __future__ import annotations

from typing import Iterable


VC_MODE_TEXT = "text"
VC_MODE_NO_TEXT = "no_text"
VC_MODE_TEXT_SEMANTICS = "text_prosody"
VC_NO_TEXT_PLACEHOLDER = "<NO_TEXT>"
VC_MODES = (VC_MODE_TEXT, VC_MODE_NO_TEXT)

VC_MODE_TOKENS = {
    VC_MODE_TEXT: "<vc_text>",
    VC_MODE_NO_TEXT: "<vc_no_text>",
}


def normalize_vc_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in VC_MODES:
        valid = ", ".join(VC_MODES)
        raise ValueError(f"unsupported vc mode: {mode!r}; expected one of: {valid}")
    return value


def parse_emit_modes(spec: str) -> list[str]:
    values = [normalize_vc_mode(item) for item in str(spec).split(",") if item.strip()]
    if not values:
        raise ValueError("emit modes cannot be empty")
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def vc_mode_token(mode: str) -> str:
    return VC_MODE_TOKENS[normalize_vc_mode(mode)]


def apply_vc_mode_token(base_instruction: str, mode: str, *, enabled: bool = True) -> str:
    normalized_mode = normalize_vc_mode(mode)
    if not enabled:
        return base_instruction
    return f"{vc_mode_token(normalized_mode)}\n{base_instruction}"


def mode_tag_suffix(mode: str) -> str:
    normalized_mode = normalize_vc_mode(mode)
    return VC_MODE_TOKENS[normalized_mode].strip("<>")


def has_both_modes(modes: Iterable[str]) -> bool:
    values = {normalize_vc_mode(mode) for mode in modes}
    return values == set(VC_MODES)
