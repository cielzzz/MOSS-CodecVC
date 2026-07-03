from __future__ import annotations

from copy import deepcopy
from typing import Any


VERSION_SPECS: dict[str, dict[str, Any]] = {
    "ver1": {
        "name": "ver1_baseline_sft",
        "description": "Baseline codec-token SFT with [S1]=source/prosody and [S2]=target timbre.",
        "requires_counterfactual": False,
        "requires_speaker_embeddings": False,
        "objectives": {
            "codec_ce": 1.0,
        },
        "aux": {
            "role_routing": {"enabled": False},
            "counterfactual_invariance": {"enabled": False},
            "adversarial_mi": {"enabled": False},
        },
    },
    "ver1.6": {
        "name": "ver1_6_reference_timbre_memory",
        "description": (
            "Ver1.5 mixed-mode LoRA plus a project-local Reference Timbre Memory / "
            "Timbre Token Adapter. S2 codec tokens are compressed into target-only "
            "timbre memory tokens and injected at selected decoder layers, with optional "
            "frozen speaker-embedding proxy loss."
        ),
        "requires_counterfactual": False,
        "requires_speaker_embeddings": False,
        "objectives": {
            "codec_ce": 1.0,
            "target_speaker_similarity": 0.1,
            "source_speaker_suppression": 0.05,
        },
        "aux": {
            "role_routing": {
                "enabled": True,
                "source_role": "source_content_prosody_or_prosody_reference",
                "timbre_role": "target_timbre_memory",
                "routing_unit": "target_position_adapter",
                "routing_mode": "gated_residual_cross_attention",
            },
            "timbre_memory": {
                "enabled": True,
                "input": "reference_audio_codes[1]",
                "encoder": "2-layer conformer",
                "optional_speaker_conditioning": True,
                "default_tokens": 16,
                "default_layers": "last_4",
                "target_only": True,
            },
            "speaker_encoder": {
                "enabled": "optional",
                "type": "frozen_ecapa_embedding_or_embedding_loader",
                "online_wav_encoder": "interface_reserved",
            },
            "counterfactual_invariance": {"enabled": False},
            "adversarial_mi": {"enabled": False},
        },
    },
    "ver2": {
        "name": "ver2_role_disentangled_routing",
        "description": "Add role-aware metadata and codebook-routing targets for source/prosody vs target timbre.",
        "requires_counterfactual": False,
        "requires_speaker_embeddings": False,
        "objectives": {
            "codec_ce": 1.0,
            "role_routing": 0.1,
        },
        "aux": {
            "role_routing": {
                "enabled": True,
                "source_role": "source_content_prosody",
                "timbre_role": "target_timbre",
                "routing_unit": "rvq_codebook",
                "routing_mode": "learned_soft_gate",
            },
            "counterfactual_invariance": {"enabled": False},
            "adversarial_mi": {"enabled": False},
        },
    },
    "ver3": {
        "name": "ver3_counterfactual_timbre_invariance",
        "description": "Add counterfactual source views so output is invariant to source speaker timbre.",
        "requires_counterfactual": True,
        "requires_speaker_embeddings": False,
        "objectives": {
            "codec_ce": 1.0,
            "role_routing": 0.1,
            "counterfactual_consistency": 0.2,
        },
        "aux": {
            "role_routing": {
                "enabled": True,
                "source_role": "source_content_prosody",
                "timbre_role": "target_timbre",
                "routing_unit": "rvq_codebook",
                "routing_mode": "learned_soft_gate",
            },
            "counterfactual_invariance": {
                "enabled": True,
                "group_key": "counterfactual_group_id",
                "view_key": "condition_view",
                "same_target_required": True,
            },
            "adversarial_mi": {"enabled": False},
        },
    },
    "ver4": {
        "name": "ver4_adversarial_speaker_suppression",
        "description": "Add adversarial speaker/source-timbre suppression on top of role routing and counterfactual views.",
        "requires_counterfactual": True,
        "requires_speaker_embeddings": True,
        "objectives": {
            "codec_ce": 1.0,
            "role_routing": 0.1,
            "counterfactual_consistency": 0.2,
            "target_speaker_similarity": 0.2,
            "source_speaker_suppression": 0.1,
            "adversarial_mi": 0.05,
        },
        "aux": {
            "role_routing": {
                "enabled": True,
                "source_role": "source_content_prosody",
                "timbre_role": "target_timbre",
                "routing_unit": "rvq_codebook",
                "routing_mode": "learned_soft_gate",
            },
            "counterfactual_invariance": {
                "enabled": True,
                "group_key": "counterfactual_group_id",
                "view_key": "condition_view",
                "same_target_required": True,
            },
            "adversarial_mi": {
                "enabled": True,
                "source_embedding_key": "source_speaker_embedding_path",
                "target_embedding_key": "timbre_ref_speaker_embedding_path",
                "gradient_reversal": True,
            },
        },
    },
    "ver_all": {
        "name": "ver_all_full_stack",
        "description": "Full-stack setting for final comparison; mirrors ver4 but is named explicitly for ablation tables.",
        "requires_counterfactual": True,
        "requires_speaker_embeddings": True,
        "objectives": {
            "codec_ce": 1.0,
            "role_routing": 0.1,
            "counterfactual_consistency": 0.2,
            "target_speaker_similarity": 0.2,
            "source_speaker_suppression": 0.1,
            "adversarial_mi": 0.05,
        },
        "aux": {
            "role_routing": {
                "enabled": True,
                "source_role": "source_content_prosody",
                "timbre_role": "target_timbre",
                "routing_unit": "rvq_codebook",
                "routing_mode": "learned_soft_gate",
            },
            "counterfactual_invariance": {
                "enabled": True,
                "group_key": "counterfactual_group_id",
                "view_key": "condition_view",
                "same_target_required": True,
            },
            "adversarial_mi": {
                "enabled": True,
                "source_embedding_key": "source_speaker_embedding_path",
                "target_embedding_key": "timbre_ref_speaker_embedding_path",
                "gradient_reversal": True,
            },
        },
    },
}


def list_versions() -> list[str]:
    return list(VERSION_SPECS)


def get_version_spec(version: str) -> dict[str, Any]:
    if version not in VERSION_SPECS:
        valid = ", ".join(list_versions())
        raise KeyError(f"Unknown version {version!r}. Valid versions: {valid}")
    return deepcopy(VERSION_SPECS[version])


def build_version_instruction(base_instruction: str, version: str) -> str:
    spec = get_version_spec(version)
    parts = [
        base_instruction.strip(),
        "Role binding: [S1]=source/prosody carrier; [S2]=target timbre reference.",
        "Do not copy [S1] speaker identity unless it is also [S2].",
    ]
    if spec["aux"]["role_routing"]["enabled"]:
        parts.append(
            "Use source-role information according to the current mode: content/prosody in no_text, "
            "prosody/style reference in text; use timbre-role information for speaker identity."
        )
    if spec["aux"]["counterfactual_invariance"]["enabled"]:
        parts.append(
            "Counterfactual source views in the same group should produce the same target voice."
        )
    if spec["aux"]["adversarial_mi"]["enabled"]:
        parts.append(
            "Suppress source-speaker leakage and match the target timbre reference."
        )
    return "\n".join(parts)
