#!/usr/bin/env python3
"""Run the effective-30k Path-X checkpoint under Batch-42 strict protocol.

This is a narrow adapter around ``004093_run_batch42_pathx_strict.py``.  It
keeps the established EN567/ZH1194 preparation and inference contract, but
registers the Batch-44 r3 weights-only warm-start checkpoint:

* system_id: ``path_x_final`` so the existing Batch-42 table builder fills the
  final row requested by the user;
* checkpoint: r3 warm-start local step 20000, nominal effective step 30000;
* frozen inference code: Batch37 evaluation snapshot used for Batch-44 full320.

The output provenance explicitly records that this is the effective-30k
weights-only continuation, not an uninterrupted optimizer-state resume.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


SYSTEM_ID = "path_x_final"
PROJECT_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
)
_SNAPSHOT_BASE = Path(__file__).with_name("004093_run_batch42_pathx_strict_base.py")
UPSTREAM_PATH = (
    _SNAPSHOT_BASE
    if _SNAPSHOT_BASE.is_file()
    else PROJECT_ROOT / "scripts/004093_run_batch42_pathx_strict.py"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k/step-20000"
)
CODE_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
)
ENGINE_SCRIPT = CODE_ROOT / "scripts/004044_run_seedtts_validation_infer_persistent.py"
FULL320_COMPLETION = (
    PROJECT_ROOT
    / "trainset/local_jobs/ver23_batch44_r3_warmstart_full320_step30000_20260713/"
    "COMPLETED.json"
)
FULL320_METRICS = (
    PROJECT_ROOT
    / "testset/outputs/ver23_batch44_r3_warmstart_full320_20260713/"
    "step-30000/aggregate/metrics.json"
)

REQUIRED_MODEL_FILES = {
    "README.md": {
        "size": 179,
        "sha256": "4d45f7d68a88a39671cc0cbc86f1acdfbee5351401eee2a97df253f0d077717f",
    },
    "adapter_config.json": {
        "size": 1179,
        "sha256": "9d5fba10346b9b894b0e1eff6afddd0661f2bb3c5ae038a62d318ab1c40381f1",
    },
    "adapter_model.safetensors": {
        "size": 87366096,
        "sha256": "b0df7ee1b39d4dfa5e513e569df7ef275f30dd043359eaa785572c288f7b0264",
    },
    "timbre_memory_config.json": {
        "size": 5023,
        "sha256": "d8dad69f6523c67ecee9cce5900ae9809099b68655bf2e945713fb39f8271519",
    },
    "timbre_memory_adapter.pt": {
        "size": 1697093491,
        "sha256": "f22ecf7dddd8f7994d4083af6e26afbe819ac56f402e8c490b19e1e4036b02ef",
    },
}

CODE_FILES = {
    "scripts/004044_run_seedtts_validation_infer_persistent.py": (
        "22045797d68d54bc2b72c64773c43464e4164b19b3a29d97537149e15594fa1d"
    ),
    "scripts/003001_infer_moss_codecvc.py": (
        "d9a3426a3668a4bdd95a81fdf86b02e32d774b4893ba0428e5b1c6fba4f5ce73"
    ),
    "moss_codecvc/models/moss_codecvc_wrapper.py": (
        "1d32527ec29fada353dc70b88a11cff972da901c5830dfeafb3bcf9f067d3ae3"
    ),
    "moss_codecvc/models/content_cross_attn.py": (
        "2be7b4cdf24c18df773b215ad3afe8682a65e519dee6ea81515ac4dd8b44ed1a"
    ),
}


def load_upstream() -> ModuleType:
    if not UPSTREAM_PATH.is_file():
        raise FileNotFoundError(f"missing upstream strict runner: {UPSTREAM_PATH}")
    module_name = "moss_codecvc_batch42_pathx_effective30k_004093"
    specification = importlib.util.spec_from_file_location(module_name, UPSTREAM_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {UPSTREAM_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _require_registered_files() -> None:
    for path in (MODEL_PATH, CODE_ROOT, ENGINE_SCRIPT, FULL320_COMPLETION, FULL320_METRICS):
        if not path.exists():
            raise FileNotFoundError(path)
    for name, item in REQUIRED_MODEL_FILES.items():
        path = MODEL_PATH / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(item["size"]):
            raise ValueError(f"{path}: size drift")
    timbre_config = json.loads(
        (MODEL_PATH / "timbre_memory_config.json").read_text(encoding="utf-8")
    )
    required_config = {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_cross_attn_gate_init": -0.5,
        "content_cross_attn_output_scale": 0.3,
        "content_encoder_layers": 2,
        "guided_attn_loss_weight": 0.05,
        "phoneme_classifier_loss_weight": 0.02,
        "content_ctc_weight": 0.0,
        "progress_loss_weight": 0.1,
        "stop_loss_weight": 0.2,
        "target_front_ce_weight": 4.0,
        "target_front_ce_seconds": 0.75,
        "use_role_routing": True,
        "num_memory_tokens": 0,
        "timbre_side_only": False,
        "source_semantic_memory_enabled": False,
        "speaker_side_pathway_enabled": False,
        "speaker_cross_attn_enabled": False,
        "speaker_condition_dropout": 0.0,
    }
    mismatches = {
        key: {"expected": expected, "actual": timbre_config.get(key)}
        for key, expected in required_config.items()
        if timbre_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"effective-30k Path-X config drift: {mismatches}")


def configure_upstream(module: ModuleType) -> None:
    _require_registered_files()
    module.SYSTEM_ID = SYSTEM_ID
    module.REGISTERED_MODEL_PATH = MODEL_PATH
    module.REGISTERED_CODE_ROOT = CODE_ROOT
    module.REGISTERED_ENGINE_SCRIPT = ENGINE_SCRIPT
    module.REGISTERED_CODE_FILES = dict(CODE_FILES)
    module.REGISTERED_MODEL_FILES = dict(REQUIRED_MODEL_FILES)

    original_inference_config = module.inference_config

    def effective30k_inference_config(args: argparse.Namespace) -> dict[str, Any]:
        result = original_inference_config(args)
        result["ref_audio_cfg_scale"] = 1.0
        result["ref_audio_cfg_implementation"] = (
            "available in registered Batch37 engine; fixed to identity scale 1.0"
        )
        result["batch44_effective30k"] = {
            "status": "user_requested_batch42_external_eval",
            "arm": "r3",
            "text_repeat": 3,
            "training_job_id": "job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab",
            "base_training_job_id": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
            "checkpoint_role": "weights-only warm-start continuation",
            "source_checkpoint_effective_step": 10000,
            "local_continuation_step": 20000,
            "nominal_effective_step": 30000,
            "optimizer_scheduler_rng_data_position": "reset at warm-start boundary",
            "strict_full320_completion": str(FULL320_COMPLETION),
            "strict_full320_metrics": str(FULL320_METRICS),
        }
        return result

    module.inference_config = effective30k_inference_config


def main(argv: Sequence[str] | None = None) -> int:
    upstream = load_upstream()
    configure_upstream(upstream)
    return int(upstream.main(list(argv if argv is not None else sys.argv[1:])))


if __name__ == "__main__":
    raise SystemExit(main())
