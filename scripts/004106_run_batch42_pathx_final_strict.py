#!/usr/bin/env python3
"""Run the selected Batch-44 v1 checkpoint under the Batch-42 strict protocol.

This is a narrow, provenance-bearing adapter around 004093.  The established
EN567/ZH1194 preparation, inference contract, frozen evaluation code, WavLM
BNF extraction, and output schema remain unchanged.  Only two registered
identities change:

* ``system_id`` becomes ``path_x_final``;
* the checkpoint path and its five exact file hashes come from a completed
  ``moss_codecvc.batch44_v1_final_selection.v1`` manifest.

Example (prepare remains deterministic and local):

    python scripts/004106_run_batch42_pathx_final_strict.py \
      --final-selection /path/FINAL_SELECTION.json prepare ...

The submit wrapper 004108 supplies the same final-selection manifest to every
worker.  This script does not submit QZ jobs itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


FINAL_SCHEMA = "moss_codecvc.batch44_v1_final_selection.v1"
EXPERIMENT_ID = "batch44_v1"
SYSTEM_ID = "path_x_final"
PROJECT_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
)
_SNAPSHOT_BASE = Path(__file__).with_name("004093_run_batch42_pathx_strict_base.py")
UPSTREAM_PATH = (
    _SNAPSHOT_BASE
    if _SNAPSHOT_BASE.is_file()
    else Path(__file__).with_name("004093_run_batch42_pathx_strict.py")
)
FINAL_CODE_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
)
FINAL_CODE_FILES = {
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
ALLOWED_RUN_DIRS = {
    "r3": "ver2_9_5_final_r3_v1_30k",
    "r5": "ver2_9_5_final_r5_v1_30k",
}
EXPECTED_REPEATS = {"r3": 3, "r5": 5}
EXPECTED_TRAIN_JOBS = {
    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
}
ALLOWED_STEPS = {26000, 28000, 30000}
REQUIRED_FILES = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "timbre_memory_config.json",
    "timbre_memory_adapter.pt",
)
_SNAPSHOT_FINALIZER = Path(__file__).with_name(
    "004107_finalize_batch43_pathx_final.py"
)


def load_final_provenance_validator() -> ModuleType:
    path = (
        _SNAPSHOT_FINALIZER
        if _SNAPSHOT_FINALIZER.is_file()
        else PROJECT_ROOT / "scripts/004107_finalize_batch43_pathx_final.py"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing 004107 final provenance validator: {path}")
    name = "moss_codecvc_batch44_final_provenance_004107"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load 004107 provenance validator from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_final_selection(
    path: Path, *, verify_checkpoint_hashes: bool = True
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"final selection does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_scalars = {
        "schema_version": FINAL_SCHEMA,
        "status": "final",
        "system_id": SYSTEM_ID,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"{path}: {key}={payload.get(key)!r}, expected {expected!r}"
            )
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("data_version") != "v1_20260709"
    ):
        raise ValueError(f"{path}: final selection is not registered Batch-44 v1")
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{path}: candidate must be an object")
    arm = candidate.get("arm")
    step = candidate.get("step")
    if arm not in ALLOWED_RUN_DIRS or step not in ALLOWED_STEPS:
        raise ValueError(f"{path}: invalid final arm/step: {arm!r}/{step!r}")
    candidate_id = f"{arm}_step-{step}"
    if candidate.get("candidate_id") != candidate_id:
        raise ValueError(
            f"{path}: candidate_id={candidate.get('candidate_id')!r}, expected {candidate_id!r}"
        )
    if candidate.get("text_repeat") != EXPECTED_REPEATS[str(arm)]:
        raise ValueError(
            f"{path}: text_repeat={candidate.get('text_repeat')!r}, "
            f"expected {EXPECTED_REPEATS[str(arm)]}"
        )
    if candidate.get("train_job_id") != EXPECTED_TRAIN_JOBS[str(arm)]:
        raise ValueError(
            f"{path}: train_job_id={candidate.get('train_job_id')!r}, "
            f"expected {EXPECTED_TRAIN_JOBS[str(arm)]!r}"
        )
    expected_checkpoint = (
        PROJECT_ROOT
        / "outputs/lora_runs"
        / ALLOWED_RUN_DIRS[str(arm)]
        / f"step-{step}"
    ).resolve()
    checkpoint = Path(str(candidate.get("checkpoint_path") or "")).expanduser().resolve()
    if checkpoint != expected_checkpoint:
        raise ValueError(
            f"{path}: checkpoint path={checkpoint}, expected registered {expected_checkpoint}"
        )
    model_files = candidate.get("model_files")
    if not isinstance(model_files, dict) or set(model_files) != set(REQUIRED_FILES):
        raise ValueError(
            f"{path}: model_files must contain exactly {list(REQUIRED_FILES)}, "
            f"got {sorted(model_files) if isinstance(model_files, dict) else model_files!r}"
        )
    for name in REQUIRED_FILES:
        registration = model_files[name]
        if not isinstance(registration, dict):
            raise ValueError(f"{path}: model_files.{name} must be an object")
        item = checkpoint / name
        if not item.is_file():
            raise FileNotFoundError(item)
        expected_size = registration.get("size")
        expected_sha = registration.get("sha256")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError(f"{path}: invalid size for {name}: {expected_size!r}")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"{path}: invalid SHA256 for {name}: {expected_sha!r}")
        if item.stat().st_size != expected_size:
            raise ValueError(
                f"{path}: {name} size={item.stat().st_size}, expected {expected_size}"
            )
        if verify_checkpoint_hashes:
            actual = sha256_file(item)
            if actual != expected_sha:
                raise ValueError(
                    f"{path}: {name} SHA256={actual}, expected {expected_sha}"
                )
    timbre_config = json.loads(
        (checkpoint / "timbre_memory_config.json").read_text(encoding="utf-8")
    )
    final_specific = {
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
        for key, expected in final_specific.items()
        if timbre_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{path}: final Path-X config drift: {mismatches}")
    validator = load_final_provenance_validator()
    validator.validate_final_selection_provenance(
        path,
        payload=payload,
        project_root=PROJECT_ROOT,
        verify_checkpoint_hashes=verify_checkpoint_hashes,
    )
    payload["_selection_path"] = str(path)
    payload["_selection_sha256"] = sha256_file(path)
    return payload


def load_upstream() -> ModuleType:
    if not UPSTREAM_PATH.is_file():
        raise FileNotFoundError(f"missing 004093 implementation: {UPSTREAM_PATH}")
    module_name = "moss_codecvc_batch42_pathx_strict_004093"
    specification = importlib.util.spec_from_file_location(module_name, UPSTREAM_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load 004093 from {UPSTREAM_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def configure_upstream(module: ModuleType, selection: Mapping[str, Any]) -> None:
    candidate = selection["candidate"]
    checkpoint = Path(str(candidate["checkpoint_path"])).resolve()
    module.SYSTEM_ID = SYSTEM_ID
    module.REGISTERED_MODEL_PATH = checkpoint
    module.REGISTERED_CODE_ROOT = FINAL_CODE_ROOT
    module.REGISTERED_ENGINE_SCRIPT = (
        FINAL_CODE_ROOT / "scripts/004044_run_seedtts_validation_infer_persistent.py"
    )
    module.REGISTERED_CODE_FILES = dict(FINAL_CODE_FILES)
    module.REGISTERED_MODEL_FILES = {
        name: {
            "size": int(candidate["model_files"][name]["size"]),
            "sha256": str(candidate["model_files"][name]["sha256"]),
        }
        for name in REQUIRED_FILES
    }
    original_inference_config = module.inference_config

    def final_inference_config(args: argparse.Namespace) -> dict[str, Any]:
        result = original_inference_config(args)
        result["ref_audio_cfg_scale"] = 1.0
        result["ref_audio_cfg_implementation"] = (
            "available in registered Batch37 engine; fixed to identity scale 1.0"
        )
        result["final_selection"] = selection["_selection_path"]
        result["final_selection_sha256"] = selection["_selection_sha256"]
        return result

    module.inference_config = final_inference_config


def split_adapter_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--final-selection",
        type=Path,
        default=os.environ.get("BATCH44_FINAL_SELECTION"),
    )
    parser.add_argument(
        "--verify-selection-checkpoint-hashes",
        action="store_true",
        help="hash all five checkpoint files before entering 004093",
    )
    known, remainder = parser.parse_known_args(argv)
    if known.final_selection is None:
        parser.error(
            "--final-selection is required (or set BATCH44_FINAL_SELECTION)"
        )
    if not remainder or remainder[0] not in {"prepare", "run"}:
        parser.error("the remaining command must start with prepare or run")
    return known, remainder


def main(argv: Sequence[str] | None = None) -> int:
    known, remainder = split_adapter_args(list(argv if argv is not None else sys.argv[1:]))
    selection = load_final_selection(
        known.final_selection,
        verify_checkpoint_hashes=known.verify_selection_checkpoint_hashes,
    )
    upstream = load_upstream()
    configure_upstream(upstream, selection)
    return int(upstream.main(remainder))


if __name__ == "__main__":
    raise SystemExit(main())
