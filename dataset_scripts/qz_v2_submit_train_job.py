#!/usr/bin/env python3
"""Submit a Qizhi distributed training job through the browser v2 API.

The local qzcli create path depends on a cached resource-spec table. Some
4090 workspaces expose no specs through that cache, while the web UI creates
jobs by querying /api/v1/resource_prices/logic_compute_groups and then calling
/api/v2/train?Action=CreateJobConsole. This helper mirrors that web flow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from qzcli.api import QzAPI, QzAPIError, V2_BROWSER_UA, V2_CLIENT_SOURCE, _curl_post
from qzcli.config import get_cookie


def _die(message: str) -> None:
    raise SystemExit(message)


def _walk_ids(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"job_id", "jobId", "id"} and isinstance(value, str):
                if value.startswith("job-") or len(value) >= 32:
                    yield value
            yield from _walk_ids(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_ids(item)


def _read_cookie() -> str:
    cookie_data = get_cookie() or {}
    cookie = cookie_data.get("cookie") or ""
    if not cookie:
        _die("No Qizhi cookie found. Run qzcli login first.")
    return cookie


def _headers(base_url: str, workspace_id: str, cookie: str) -> Dict[str, str]:
    return {
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json",
        "cookie": cookie,
        "origin": base_url,
        "referer": f"{base_url}/jobs/distributedTraining?spaceId={workspace_id}",
        "user-agent": V2_BROWSER_UA,
        "x-inspire-client-source": V2_CLIENT_SOURCE,
    }


def _query_specs(
    api: QzAPI,
    headers: Dict[str, str],
    *,
    workspace_id: str,
    project_id: str,
    compute_group_id: str,
    task_priority: int,
) -> list[Dict[str, Any]]:
    payload = {
        "logic_compute_group_id": compute_group_id,
        "workspace_id": workspace_id,
        "schedule_config_type": "SCHEDULE_CONFIG_TYPE_TRAIN",
        "project_id": project_id,
        "task_priority": task_priority,
    }
    response = _curl_post(
        f"{api.base_url}/api/v1/resource_prices/logic_compute_groups",
        json=payload,
        headers=headers,
        timeout=60,
    )
    try:
        result = response.json()
    except ValueError as exc:
        _die(f"Resource spec query returned non-JSON HTTP {response.status_code}: {exc}")
    if response.status_code != 200 or result.get("code") != 0:
        _die(f"Resource spec query failed HTTP {response.status_code}: {result}")
    return (result.get("data") or {}).get("lcg_resource_spec_prices") or []


def _select_spec(
    specs: list[Dict[str, Any]],
    *,
    quota_id: str,
    gpus_per_node: int,
    min_cpu: int,
) -> Dict[str, Any]:
    if quota_id:
        for spec in specs:
            if spec.get("quota_id") == quota_id:
                return spec
        _die(f"Requested quota_id was not returned by resource_prices: {quota_id}")

    candidates = [
        spec
        for spec in specs
        if int(spec.get("gpu_count") or 0) == gpus_per_node
        and int(spec.get("cpu_count") or 0) >= min_cpu
    ]
    if not candidates:
        summary = [
            {
                "quota_id": spec.get("quota_id"),
                "cpu_count": spec.get("cpu_count"),
                "gpu_count": spec.get("gpu_count"),
                "memory_size_gib": spec.get("memory_size_gib"),
            }
            for spec in specs
        ]
        _die(f"No matching resource spec found. Available specs: {summary}")
    return sorted(candidates, key=lambda item: int(item.get("cpu_count") or 0), reverse=True)[0]


def _resource_spec_price(spec: Dict[str, Any], compute_group_id: str) -> Dict[str, Any]:
    gpu_info = spec.get("gpu_info") or {}
    cpu_info = spec.get("cpu_info") or {}
    return {
        "cpu_type": cpu_info.get("cpu_type") or "",
        "cpu_count": int(spec.get("cpu_count") or 0),
        "gpu_type": gpu_info.get("gpu_type") or spec.get("gpu_type") or "",
        "gpu_count": int(spec.get("gpu_count") or 0),
        "memory_size_gib": int(float(spec.get("memory_size_gib") or 0)),
        "logic_compute_group_id": compute_group_id,
        "quota_id": spec.get("quota_id") or "",
    }


def _build_payload(args: argparse.Namespace, spec: Dict[str, Any]) -> Dict[str, Any]:
    mem_gi = int(float(spec.get("memory_size_gib") or 0))
    shm_gi = int(float(args.shm_gi))
    if shm_gi > mem_gi:
        _die(f"shm_gi ({shm_gi}) cannot exceed selected memory_size_gib ({mem_gi})")
    return {
        "name": args.name,
        "logic_compute_group_id": args.compute_group,
        "project_id": args.project,
        "workspace_id": args.workspace,
        "framework": args.framework,
        "command": args.command,
        "task_priority": int(args.priority),
        "auto_fault_tolerance": False,
        "framework_config": [
            {
                "cpu": int(spec.get("cpu_count") or 0),
                "gpu_count": int(spec.get("gpu_count") or 0),
                "mem_gi": mem_gi,
                "resource_spec_price": _resource_spec_price(spec, args.compute_group),
                "image": args.image,
                "image_type": args.image_type,
                "instance_count": int(args.instances),
                "shm_gi": shm_gi,
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--compute-group", required=True)
    parser.add_argument("--framework", default="pytorch")
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--shm-gi", type=float, default=512)
    parser.add_argument("--priority", type=int, default=3)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-type", default="SOURCE_PRIVATE")
    parser.add_argument("--command", required=True)
    parser.add_argument("--spec", default="", help="Optional quota_id to force.")
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--min-cpu", type=int, default=1)
    parser.add_argument("--payload-json", type=Path)
    parser.add_argument("--response-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = QzAPI()
    cookie = _read_cookie()
    headers = _headers(api.base_url, args.workspace, cookie)
    specs = _query_specs(
        api,
        headers,
        workspace_id=args.workspace,
        project_id=args.project,
        compute_group_id=args.compute_group,
        task_priority=args.priority,
    )
    spec = _select_spec(
        specs,
        quota_id=args.spec,
        gpus_per_node=args.gpus_per_node,
        min_cpu=args.min_cpu,
    )
    payload = _build_payload(args, spec)

    if args.payload_json:
        args.payload_json.parent.mkdir(parents=True, exist_ok=True)
        args.payload_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.dry_run:
        print(json.dumps({"dry_run": True, "payload": payload}, ensure_ascii=False, indent=2))
        return 0

    try:
        response = api._request_v2("train", "CreateJobConsole", payload)
    except QzAPIError as exc:
        _die(f"CreateJobConsole request failed: {exc}")

    if args.response_json:
        args.response_json.parent.mkdir(parents=True, exist_ok=True)
        args.response_json.write_text(json.dumps(response, ensure_ascii=False, indent=2))

    error = ((response.get("ResponseMetadata") or {}).get("Error") or {})
    if error:
        _die(f"CreateJobConsole failed: {error}")

    job_id: Optional[str] = next(_walk_ids(response), None)
    if job_id and not job_id.startswith("job-") and len(job_id) == 36:
        job_id = f"job-{job_id}"
    if not job_id:
        _die(f"CreateJobConsole did not return a job id: {response}")

    print(
        json.dumps(
            {
                "job_id": job_id,
                "workspace_id": args.workspace,
                "name": args.name,
                "compute_group": args.compute_group,
                "quota_id": payload["framework_config"][0]["resource_spec_price"]["quota_id"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
