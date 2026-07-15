#!/usr/bin/env python3
"""Stage and atomically publish one 320-case run to the port-18603 page.

The default mode is a read-only plan: it builds and validates a complete
candidate in a sibling staging directory, prints the publication plan, and
removes the staging directory without touching the served page.  A live
publication additionally requires both ``--apply`` and the exact confirmation
string ``--confirm-apply ATOMIC_REPLACE``.

Publication order is deliberately fail-safe:

1. validate the fixed validation set, manifests, model, outputs and live
   source/reference anchors;
2. copy all new targets into an immutable, content-addressed staging tree;
3. write an immutable backup of the current ``benchmark_data.js``;
4. atomically install the new, content-addressed asset directory;
5. atomically replace ``benchmark_data.js`` on the same filesystem.

The HTTP server is never restarted.  Old asset directories are intentionally
left in place as rollback material; only payload references are replaced.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGE_DIR = ROOT / "outputs/listening_frontend/seedtts_valid_benchmark"
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_VALIDATION_SHA256 = "725ee9d58a7e6066d2a7b79c858cb6ff4dd7292cc167c45dc6b6ebbeaff2fe14"
PAYLOAD_PREFIX = "window.SEEDTTS_BENCHMARK = "
EXPECTED_CASE_COUNT = 320
VER23_ROLE = "ver23_content_latest"
DEFAULT_REPLACE_PREFIXES = ("ver23_content",)
PLAYABLE_STATUSES = {"ok", "ok_after_rerun", "skipped_exists"}
LOCK_NAME = ".benchmark_data.update.lock"


class UpdateError(RuntimeError):
    """Raised when any publication precondition is not proven."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise UpdateError(f"missing or empty file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise UpdateError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise UpdateError(f"JSONL row must be an object at {path}:{line_no}")
            yield row


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem[:160] or "item"


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new_file(path: Path, data: bytes, *, mode: int = 0o444) -> None:
    """Create a new durable file and refuse an accidental overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    fsync_dir(path.parent)


def copy_durable(source: Path, destination: Path) -> None:
    """Copy one file into staging and make the copy durable and read-only."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    destination.chmod(0o444)


def atomic_replace_bytes(path: Path, data: bytes) -> None:
    """Replace one file atomically using a same-directory temporary."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(fd)
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_payload(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8").strip()
    if not text.startswith(PAYLOAD_PREFIX):
        raise UpdateError(f"unexpected benchmark payload prefix: {path}")
    body = text[len(PAYLOAD_PREFIX) :]
    if body.endswith(";"):
        body = body[:-1]
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise UpdateError("benchmark payload must contain a runs list")
    return payload, raw, sha256_bytes(raw)


def serialize_payload(payload: Mapping[str, Any]) -> bytes:
    text = PAYLOAD_PREFIX + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n"
    return text.encode("utf-8")


def require_audio_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise UpdateError(f"missing {label}: {path}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 44:
        raise UpdateError(f"invalid/empty {label}: {resolved}")
    return resolved


def validate_validation_rows(
    path: Path,
    *,
    expected_sha256: str,
    expected_count: int,
) -> tuple[list[dict[str, Any]], str]:
    path = path.expanduser().resolve()
    actual_sha = sha256_file(path)
    if expected_sha256 and actual_sha != expected_sha256:
        raise UpdateError(
            f"validation SHA drift: expected={expected_sha256} actual={actual_sha} path={path}"
        )
    rows = list(iter_jsonl(path))
    if len(rows) != expected_count:
        raise UpdateError(f"fixed validation set must have {expected_count} rows, got {len(rows)}")
    case_ids: list[str] = []
    stems: list[str] = []
    for index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise UpdateError(f"validation row {index} has no case_id")
        mode = str(row.get("mode") or "")
        if mode not in {"no_text", "text"}:
            raise UpdateError(f"validation row {case_id} has unsupported mode={mode!r}")
        require_audio_file(Path(str(row.get("source_audio") or "")), f"source audio for {case_id}")
        require_audio_file(
            Path(str(row.get("timbre_ref_audio") or "")),
            f"timbre reference audio for {case_id}",
        )
        case_ids.append(case_id)
        stems.append(safe_stem(case_id))
    if len(set(case_ids)) != expected_count:
        raise UpdateError("fixed validation case IDs are not unique")
    if len(set(stems)) != expected_count:
        raise UpdateError("fixed validation case IDs collide after filename sanitization")
    return rows, actual_sha


def discover_manifests(output_dir: Path) -> list[Path]:
    shards = sorted(output_dir.glob("manifest.shard*.jsonl"))
    if not shards and (output_dir / "manifest.jsonl").is_file():
        shards = [output_dir / "manifest.jsonl"]
    rerun = output_dir / "manifest.rerun_failed.jsonl"
    if rerun.is_file():
        shards.append(rerun)
    if not shards:
        raise UpdateError(f"no inference manifests found under {output_dir}")
    return [path.resolve() for path in shards]


def load_manifest_rows(
    manifests: Sequence[Path],
    *,
    output_dir: Path,
    validation_case_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    wanted = set(validation_case_ids)
    by_case: dict[str, dict[str, Any]] = {}
    raw_statuses: Counter[str] = Counter()
    manifest_artifacts: list[dict[str, Any]] = []
    unknown: set[str] = set()
    for manifest in manifests:
        manifest = manifest.expanduser().resolve()
        manifest_artifacts.append(artifact(manifest))
        for row in iter_jsonl(manifest):
            case_id = str(row.get("case_id") or "")
            if not case_id:
                raise UpdateError(f"manifest row has no case_id: {manifest}")
            if case_id not in wanted:
                unknown.add(case_id)
                continue
            by_case[case_id] = row
            raw_statuses[str(row.get("status") or "unknown")] += 1
    if unknown:
        raise UpdateError(f"manifest contains cases outside fixed320: {sorted(unknown)[:8]}")
    missing = [case_id for case_id in validation_case_ids if case_id not in by_case]
    if missing:
        raise UpdateError(f"manifest misses {len(missing)} fixed320 cases: {missing[:8]}")

    output_dir = output_dir.resolve()
    for case_id in validation_case_ids:
        row = by_case[case_id]
        status = str(row.get("status") or "")
        if status not in PLAYABLE_STATUSES:
            raise UpdateError(f"non-playable manifest status for {case_id}: {status!r}")
        output_value = row.get("output_wav") or row.get("target_audio")
        output_path = Path(str(output_value or (output_dir / f"{safe_stem(case_id)}.wav")))
        resolved = require_audio_file(output_path, f"generated target for {case_id}")
        try:
            resolved.relative_to(output_dir)
        except ValueError as exc:
            raise UpdateError(
                f"generated target escapes --output-dir for {case_id}: {resolved}"
            ) from exc
        row = dict(row)
        row["_resolved_output_wav"] = str(resolved)
        by_case[case_id] = row
    return by_case, manifest_artifacts, raw_statuses


def validate_existing_payload(
    payload: Mapping[str, Any],
    validation_case_ids: Sequence[str],
) -> None:
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or int(dataset.get("total") or 0) != len(validation_case_ids):
        raise UpdateError("existing page dataset is not the fixed320 dataset")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise UpdateError("existing page has no runs")
    seen: set[str] = set()
    expected = list(validation_case_ids)
    for run in runs:
        if not isinstance(run, dict):
            raise UpdateError("existing run entry is not an object")
        run_id = str(run.get("run_id") or "")
        if not run_id or run_id in seen:
            raise UpdateError(f"existing page has empty/duplicate run_id={run_id!r}")
        seen.add(run_id)
        samples = run.get("samples")
        if not isinstance(samples, list):
            raise UpdateError(f"existing run {run_id} has no samples list")
        actual = [str(sample.get("case_id") or "") for sample in samples if isinstance(sample, dict)]
        if actual != expected:
            raise UpdateError(f"existing run {run_id} case order differs from fixed320")
    if "ver2_3" not in seen:
        raise UpdateError("existing page must contain exactly one ver2_3 anchor run")


def validate_live_anchor_links(page_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        case_id = str(row["case_id"])
        stem = safe_stem(case_id)
        for kind, row_key in (("source", "source_audio"), ("timbre", "timbre_ref_audio")):
            live_link = page_dir / "assets" / kind / f"{stem}.wav"
            if not live_link.is_symlink():
                raise UpdateError(f"live {kind} anchor is not a symlink: {live_link}")
            expected = require_audio_file(Path(str(row[row_key])), f"{kind} input for {case_id}")
            try:
                actual = live_link.resolve(strict=True)
            except FileNotFoundError as exc:
                raise UpdateError(f"broken live {kind} anchor: {live_link}") from exc
            if actual != expected:
                raise UpdateError(
                    f"live {kind} anchor target drift for {case_id}: expected={expected} actual={actual}"
                )


def mode_display_text(row: Mapping[str, Any]) -> str:
    if str(row.get("mode") or "") == "text":
        return str(row.get("text") or row.get("content_ref_text") or "")
    return str(row.get("content_ref_text") or row.get("source_text") or "")


def target_tree_fingerprint(
    *,
    run_id: str,
    run_label: str,
    output_dir: Path,
    model_path: Path,
    validation_sha256: str,
    manifest_artifacts: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    manifest_by_case: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, dict[str, Any]]]:
    target_artifacts: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    header = {
        "run_id": run_id,
        "run_label": run_label,
        "output_dir": str(output_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "validation_sha256": validation_sha256,
        "manifests": list(manifest_artifacts),
    }
    digest.update(json.dumps(header, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for row in validation_rows:
        case_id = str(row["case_id"])
        target = Path(str(manifest_by_case[case_id]["_resolved_output_wav"]))
        spec = artifact(target)
        target_artifacts[case_id] = spec
        digest.update(case_id.encode("utf-8"))
        digest.update(str(spec["size"]).encode("ascii"))
        digest.update(str(spec["sha256"]).encode("ascii"))
    return digest.hexdigest(), target_artifacts


def build_candidate_run(
    *,
    run_id: str,
    run_label: str,
    output_dir: Path,
    model_path: Path,
    validation_jsonl: Path,
    validation_sha256: str,
    validation_rows: Sequence[Mapping[str, Any]],
    manifest_by_case: Mapping[str, Mapping[str, Any]],
    manifest_artifacts: Sequence[Mapping[str, Any]],
    raw_statuses: Counter[str],
    stage_root: Path,
) -> tuple[dict[str, Any], str, Path]:
    fingerprint, target_artifacts = target_tree_fingerprint(
        run_id=run_id,
        run_label=run_label,
        output_dir=output_dir,
        model_path=model_path,
        validation_sha256=validation_sha256,
        manifest_artifacts=manifest_artifacts,
        validation_rows=validation_rows,
        manifest_by_case=manifest_by_case,
    )
    asset_key = f"{safe_stem(run_id)}-{fingerprint[:12]}"
    staged_asset_dir = stage_root / "assets" / "runs" / asset_key
    staged_target_dir = staged_asset_dir / "target"
    staged_target_dir.mkdir(parents=True, exist_ok=False)

    samples: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    for index, row in enumerate(validation_rows, start=1):
        case_id = str(row["case_id"])
        stem = safe_stem(case_id)
        manifest_row = manifest_by_case[case_id]
        target = Path(str(manifest_row["_resolved_output_wav"]))
        staged_link = staged_target_dir / f"{stem}.wav"
        copy_durable(target, staged_link)
        status = str(manifest_row.get("status") or "")
        mode = str(row.get("mode") or "")
        cell = str(row.get("cell") or "")
        sample = {
            "index": index,
            "case_id": case_id,
            "mode": mode,
            "cell": cell,
            "source_lang": row.get("source_lang"),
            "ref_lang": row.get("ref_lang"),
            "source_id": row.get("source_id"),
            "ref_id": row.get("ref_id"),
            "source_audio": f"assets/source/{stem}.wav",
            "timbre_audio": f"assets/timbre/{stem}.wav",
            "target_audio": f"assets/runs/{asset_key}/target/{stem}.wav",
            "source_text": row.get("source_text"),
            "timbre_ref_text": row.get("timbre_ref_text"),
            "input_text": row.get("text") if mode == "text" else "",
            "content_text": row.get("content_ref_text"),
            "display_text": mode_display_text(row),
            "eval_text_source": row.get("eval_text_source"),
            "source_path": row.get("source_audio"),
            "timbre_path": row.get("timbre_ref_audio"),
            "target_path": str(target),
            "status": status,
            "returncode": manifest_row.get("returncode"),
            "elapsed_sec": manifest_row.get("elapsed_sec"),
            "output_exists": True,
        }
        samples.append(sample)
        status_counts[status] += 1
        mode_counts[mode] += 1
        cell_counts[cell] += 1

    expected_names = {f"{safe_stem(str(row['case_id']))}.wav" for row in validation_rows}
    actual_files = {path.name for path in staged_target_dir.iterdir() if path.is_file()}
    if actual_files != expected_names:
        raise UpdateError("staged target file set differs from fixed320")
    for case_id, spec in target_artifacts.items():
        copied = staged_target_dir / f"{safe_stem(case_id)}.wav"
        if copied.is_symlink() or copied.stat().st_size != int(spec["size"]):
            raise UpdateError(f"staged target copy metadata drift for {case_id}")
        if sha256_file(copied) != str(spec["sha256"]):
            raise UpdateError(f"staged target copy SHA drift for {case_id}")

    asset_manifest = {
        "schema": "moss_codecvc.seedtts_listening_asset_manifest.v1",
        "run_id": run_id,
        "asset_key": asset_key,
        "target_tree_sha256": fingerprint,
        "target_audio_count": len(target_artifacts),
        "files": {
            case_id: {
                "asset_name": f"{safe_stem(case_id)}.wav",
                "source_path": spec["path"],
                "size": spec["size"],
                "sha256": spec["sha256"],
            }
            for case_id, spec in target_artifacts.items()
        },
    }
    asset_manifest_path = staged_asset_dir / "ASSET_MANIFEST.json"
    write_new_file(
        asset_manifest_path,
        (json.dumps(asset_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    fsync_dir(staged_target_dir)
    fsync_dir(staged_asset_dir)

    run = {
        "run_id": run_id,
        "label": run_label,
        "model_path": str(model_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "manifest_jsonl": [str(item["path"]) for item in manifest_artifacts],
        "built_at_utc": utc_now(),
        "counts": {
            "samples": len(samples),
            "status": dict(status_counts),
            "mode": dict(mode_counts),
            "cell": dict(cell_counts),
            "manifest_status_raw": dict(raw_statuses),
            "missing_audio_links": {},
        },
        "publication": {
            "schema": "moss_codecvc.seedtts_listening_atomic_run.v1",
            "role": VER23_ROLE,
            "asset_key": asset_key,
            "target_tree_sha256": fingerprint,
            "asset_manifest_sha256": sha256_file(asset_manifest_path),
            "validation_jsonl": str(validation_jsonl.resolve()),
            "validation_sha256": validation_sha256,
            "target_audio_count": len(target_artifacts),
            "manifest_artifacts": list(manifest_artifacts),
        },
        "samples": samples,
    }
    return run, asset_key, staged_asset_dir


def should_replace_run(run: Mapping[str, Any], *, run_id: str, prefixes: Sequence[str]) -> bool:
    existing_id = str(run.get("run_id") or "")
    publication = run.get("publication")
    role = str(publication.get("role") or "") if isinstance(publication, dict) else ""
    return (
        existing_id == run_id
        or role == VER23_ROLE
        or any(existing_id.startswith(prefix) for prefix in prefixes if prefix)
    )


def build_final_payload(
    existing: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    *,
    replace_prefixes: Sequence[str],
    validation_case_ids: Sequence[str],
) -> tuple[dict[str, Any], list[str], list[str]]:
    candidate_id = str(candidate_run["run_id"])
    baselines: list[dict[str, Any]] = []
    ver2_runs: list[dict[str, Any]] = []
    removed: list[str] = []
    for run in existing["runs"]:
        run_id = str(run.get("run_id") or "")
        if run_id == "ver2_3":
            ver2_runs.append(copy.deepcopy(run))
        elif should_replace_run(run, run_id=candidate_id, prefixes=replace_prefixes):
            removed.append(run_id)
        else:
            baselines.append(copy.deepcopy(run))
    if len(ver2_runs) != 1:
        raise UpdateError(f"expected exactly one ver2_3 run, got {len(ver2_runs)}")
    final = copy.deepcopy(dict(existing))
    final["runs"] = baselines + ver2_runs + [copy.deepcopy(dict(candidate_run))]
    final_ids = [str(run.get("run_id") or "") for run in final["runs"]]
    if len(set(final_ids)) != len(final_ids):
        raise UpdateError(f"candidate payload has duplicate run IDs: {final_ids}")
    if final_ids[-2:] != ["ver2_3", candidate_id]:
        raise UpdateError(f"final run order invariant failed: {final_ids}")
    expected_cases = list(validation_case_ids)
    for run in final["runs"]:
        actual_cases = [str(sample.get("case_id") or "") for sample in run.get("samples", [])]
        if actual_cases != expected_cases:
            raise UpdateError(f"final run {run.get('run_id')} case order differs from fixed320")
    return final, removed, final_ids


def validate_asset_tree(asset_dir: Path, candidate_run: Mapping[str, Any]) -> None:
    samples = candidate_run.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_CASE_COUNT:
        raise UpdateError("candidate run does not contain 320 samples")
    target_dir = asset_dir / "target"
    files = list(target_dir.iterdir()) if target_dir.is_dir() else []
    if len(files) != EXPECTED_CASE_COUNT or any(
        path.is_symlink() or not path.is_file() for path in files
    ):
        raise UpdateError("candidate target asset tree must contain exactly 320 immutable files")
    manifest_path = asset_dir / "ASSET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    publication = candidate_run.get("publication")
    if not isinstance(publication, dict):
        raise UpdateError("candidate run has no publication metadata")
    expected_manifest = {
        "schema": "moss_codecvc.seedtts_listening_asset_manifest.v1",
        "run_id": candidate_run.get("run_id"),
        "asset_key": publication.get("asset_key"),
        "target_tree_sha256": publication.get("target_tree_sha256"),
        "target_audio_count": EXPECTED_CASE_COUNT,
    }
    bad = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if bad:
        raise UpdateError(f"candidate asset manifest identity drift: {bad}")
    if sha256_file(manifest_path) != publication.get("asset_manifest_sha256"):
        raise UpdateError("candidate asset manifest SHA drift")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict) or len(manifest_files) != EXPECTED_CASE_COUNT:
        raise UpdateError("candidate asset manifest must bind exactly 320 files")
    for sample in samples:
        case_id = str(sample.get("case_id") or "")
        spec = manifest_files.get(case_id)
        if not isinstance(spec, dict):
            raise UpdateError(f"candidate asset manifest misses {case_id}")
        copied = target_dir / Path(str(sample["target_audio"])).name
        if copied.name != spec.get("asset_name") or copied.is_symlink() or not copied.is_file():
            raise UpdateError(f"candidate target audio path drift: {case_id}")
        if copied.stat().st_size != int(spec.get("size") or -1):
            raise UpdateError(f"candidate target audio size drift: {case_id}")
        if sha256_file(copied) != str(spec.get("sha256") or ""):
            raise UpdateError(f"candidate target audio SHA drift: {case_id}")


def same_asset_tree(left: Path, right: Path, candidate_run: Mapping[str, Any]) -> bool:
    try:
        validate_asset_tree(left, candidate_run)
        validate_asset_tree(right, candidate_run)
    except (UpdateError, FileNotFoundError):
        return False
    return (left / "ASSET_MANIFEST.json").read_bytes() == (
        right / "ASSET_MANIFEST.json"
    ).read_bytes()


def acquire_lock(page_dir: Path, plan: Mapping[str, Any]) -> Path:
    lock = page_dir / LOCK_NAME
    payload = {
        "schema": "moss_codecvc.seedtts_listening_update_lock.v1",
        "created_utc": utc_now(),
        "pid": os.getpid(),
        "old_data_sha256": plan["old_data_sha256"],
        "candidate_data_sha256": plan["candidate_data_sha256"],
        "run_id": plan["run_id"],
    }
    try:
        write_new_file(lock, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"), mode=0o600)
    except FileExistsError as exc:
        raise UpdateError(f"live page update lock already exists: {lock}") from exc
    return lock


def apply_publication(
    *,
    page_dir: Path,
    data_path: Path,
    old_data: bytes,
    candidate_data: bytes,
    candidate_run: Mapping[str, Any],
    asset_key: str,
    staged_asset_dir: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    lock = acquire_lock(page_dir, plan)
    backup_path: Path | None = None
    live_asset_dir = page_dir / "assets" / "runs" / asset_key
    asset_installed = False
    try:
        if sha256_file(data_path) != plan["old_data_sha256"]:
            raise UpdateError("benchmark_data.js changed after staging; refusing publication")

        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = (
            page_dir
            / "backups"
            / f"benchmark_data.{stamp}.{plan['old_data_sha256'][:12]}.{uuid.uuid4().hex[:8]}.js"
        )
        write_new_file(backup_path, old_data)

        live_asset_dir.parent.mkdir(parents=True, exist_ok=True)
        if live_asset_dir.exists() or live_asset_dir.is_symlink():
            if not live_asset_dir.is_dir() or not same_asset_tree(
                live_asset_dir, staged_asset_dir, candidate_run
            ):
                raise UpdateError(f"content-addressed live asset collision: {live_asset_dir}")
        else:
            os.replace(staged_asset_dir, live_asset_dir)
            fsync_dir(live_asset_dir.parent)
            asset_installed = True
        validate_asset_tree(live_asset_dir, candidate_run)

        # The new assets are now complete and durable, but still unreferenced.
        # Recheck the old payload immediately before the atomic visibility cutover.
        if sha256_file(data_path) != plan["old_data_sha256"]:
            raise UpdateError(
                "benchmark_data.js changed before cutover; new asset remains safely unreferenced"
            )
        atomic_replace_bytes(data_path, candidate_data)
        if sha256_file(data_path) != plan["candidate_data_sha256"]:
            raise UpdateError("post-cutover benchmark_data.js SHA mismatch")
        published, _, _ = load_payload(data_path)
        if [str(run.get("run_id") or "") for run in published["runs"]] != plan["final_run_order"]:
            raise UpdateError("post-cutover run order mismatch")
        result = dict(plan)
        result.update(
            {
                "mode": "applied",
                "live_mutations": True,
                "backup_path": str(backup_path),
                "live_asset_dir": str(live_asset_dir),
                "asset_installed": asset_installed,
                "completed_utc": utc_now(),
            }
        )
        return result
    finally:
        lock.unlink(missing_ok=True)
        fsync_dir(page_dir)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", "--run-label", dest="run_label", required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--page-dir", type=Path, default=DEFAULT_PAGE_DIR)
    ap.add_argument("--validation-jsonl", type=Path, default=DEFAULT_VALIDATION_JSONL)
    ap.add_argument(
        "--expected-validation-sha256",
        default=DEFAULT_VALIDATION_SHA256,
        help="Pinned fixed320 SHA. Pass the test fixture SHA only in isolated tests.",
    )
    ap.add_argument("--expected-case-count", type=int, default=EXPECTED_CASE_COUNT)
    ap.add_argument("--manifest-jsonl", type=Path, action="append", default=[])
    ap.add_argument(
        "--replace-run-prefix",
        action="append",
        default=[],
        help="Additional old latest-run prefix to remove from the payload.",
    )
    ap.add_argument(
        "--staging-parent",
        type=Path,
        default=None,
        help="Sibling staging parent; defaults to page-dir's parent for same-filesystem rename.",
    )
    ap.add_argument("--apply", action="store_true", help="Perform the atomic live cutover.")
    ap.add_argument(
        "--confirm-apply",
        default="",
        help="Live apply requires the exact value ATOMIC_REPLACE.",
    )
    ap.add_argument("--plan-json", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        page_dir_input = args.page_dir.expanduser()
        if page_dir_input.is_symlink():
            raise UpdateError(f"page-dir may not be a symlink: {page_dir_input}")
        page_dir = page_dir_input.resolve()
        if not page_dir.is_dir():
            raise UpdateError(f"page-dir must be a real directory: {page_dir}")
        data_path = page_dir / "benchmark_data.js"
        output_dir = args.output_dir.expanduser().resolve()
        model_path = args.model_path.expanduser().resolve()
        if not output_dir.is_dir():
            raise UpdateError(f"output-dir is not a directory: {output_dir}")
        if not model_path.is_dir():
            raise UpdateError(f"model-path is not a directory: {model_path}")
        run_id = str(args.run_id).strip()
        run_label = str(args.run_label).strip()
        if not run_id or not run_label:
            raise UpdateError("run-id and label must be non-empty")
        if run_id == "ver2_3":
            raise UpdateError("latest run-id may not overwrite the ver2_3 anchor")
        if int(args.expected_case_count) != EXPECTED_CASE_COUNT:
            raise UpdateError("this publisher is intentionally restricted to the fixed 320-case page")
        if args.apply and args.confirm_apply != "ATOMIC_REPLACE":
            raise UpdateError("--apply requires --confirm-apply ATOMIC_REPLACE")

        existing_payload, old_data, old_data_sha = load_payload(data_path)
        validation_rows, validation_sha = validate_validation_rows(
            args.validation_jsonl,
            expected_sha256=str(args.expected_validation_sha256 or ""),
            expected_count=int(args.expected_case_count),
        )
        validation_case_ids = [str(row["case_id"]) for row in validation_rows]
        validate_existing_payload(existing_payload, validation_case_ids)
        validate_live_anchor_links(page_dir, validation_rows)

        manifests = (
            [path.expanduser().resolve() for path in args.manifest_jsonl]
            if args.manifest_jsonl
            else discover_manifests(output_dir)
        )
        manifest_by_case, manifest_artifacts, raw_statuses = load_manifest_rows(
            manifests,
            output_dir=output_dir,
            validation_case_ids=validation_case_ids,
        )
        replace_prefixes = tuple(
            dict.fromkeys((*DEFAULT_REPLACE_PREFIXES, *(str(x) for x in args.replace_run_prefix)))
        )
        staging_parent = (
            args.staging_parent.expanduser().resolve()
            if args.staging_parent is not None
            else page_dir.parent
        )
        if not staging_parent.is_dir():
            raise UpdateError(f"staging-parent must already exist: {staging_parent}")
        if os.stat(staging_parent).st_dev != os.stat(page_dir).st_dev:
            raise UpdateError("staging-parent must be on the same filesystem as page-dir")

        with tempfile.TemporaryDirectory(
            prefix=f".{page_dir.name}.atomic-stage-", dir=staging_parent
        ) as temporary:
            stage_root = Path(temporary)
            candidate_run, asset_key, staged_asset_dir = build_candidate_run(
                run_id=run_id,
                run_label=run_label,
                output_dir=output_dir,
                model_path=model_path,
                validation_jsonl=args.validation_jsonl,
                validation_sha256=validation_sha,
                validation_rows=validation_rows,
                manifest_by_case=manifest_by_case,
                manifest_artifacts=manifest_artifacts,
                raw_statuses=raw_statuses,
                stage_root=stage_root,
            )
            validate_asset_tree(staged_asset_dir, candidate_run)
            candidate_payload, removed_runs, final_run_order = build_final_payload(
                existing_payload,
                candidate_run,
                replace_prefixes=replace_prefixes,
                validation_case_ids=validation_case_ids,
            )
            candidate_data = serialize_payload(candidate_payload)
            candidate_path = stage_root / "benchmark_data.js"
            candidate_path.write_bytes(candidate_data)
            reloaded, _, reloaded_sha = load_payload(candidate_path)
            if reloaded != candidate_payload:
                raise UpdateError("staged payload round-trip mismatch")
            candidate_sha = sha256_bytes(candidate_data)
            if reloaded_sha != candidate_sha:
                raise UpdateError("staged payload SHA mismatch")

            plan: dict[str, Any] = {
                "schema": "moss_codecvc.seedtts_listening_atomic_update_plan.v1",
                "mode": "plan",
                "live_mutations": False,
                "created_utc": utc_now(),
                "page_dir": str(page_dir),
                "data_path": str(data_path),
                "run_id": run_id,
                "run_label": run_label,
                "model_path": str(model_path),
                "output_dir": str(output_dir),
                "validation_jsonl": str(args.validation_jsonl.expanduser().resolve()),
                "validation_sha256": validation_sha,
                "case_count": len(validation_rows),
                "manifest_artifacts": manifest_artifacts,
                "old_data_sha256": old_data_sha,
                "candidate_data_sha256": candidate_sha,
                "asset_key": asset_key,
                "target_audio_count": EXPECTED_CASE_COUNT,
                "removed_runs": removed_runs,
                "final_run_order": final_run_order,
                "baseline_relative_order": final_run_order[:-2],
                "commit_order": [
                    "backup_old_benchmark_data",
                    "publish_content_addressed_asset",
                    "atomic_replace_benchmark_data",
                ],
                "invariants": {
                    "fixed320_case_order": True,
                    "all_live_source_links_resolve": True,
                    "all_live_timbre_links_resolve": True,
                    "all_staged_target_links_resolve": True,
                    "ver2_3_penultimate": True,
                    "latest_last": True,
                    "server_restart_required": False,
                },
            }
            if args.apply:
                result = apply_publication(
                    page_dir=page_dir,
                    data_path=data_path,
                    old_data=old_data,
                    candidate_data=candidate_data,
                    candidate_run=candidate_run,
                    asset_key=asset_key,
                    staged_asset_dir=staged_asset_dir,
                    plan=plan,
                )
            else:
                result = plan

        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.plan_json is not None:
            plan_path = args.plan_json.expanduser().resolve()
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_replace_bytes(plan_path, rendered.encode("utf-8"))
        sys.stdout.write(rendered)
        return 0
    except (UpdateError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
