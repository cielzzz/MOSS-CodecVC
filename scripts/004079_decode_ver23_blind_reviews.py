#!/usr/bin/env python3
"""Decode blinded listening-page reviews without modifying their inputs.

The decoder joins anonymous choices in a review export with the private
``candidate_mapping`` stored in the corresponding manifest.  It supports the
current multiway export (``winner`` is A-E) and pairwise exports where the
choice is A, B, ``tie``, or ``neither``.  Ties and "neither candidate" are
kept as separate outcomes and are never split between the two real systems.
Manifests may provide either the legacy single ``decision_rule`` or ordered
``decision_rules`` that combine selected role wins with tie/neither counts and
choose the highest-priority matching conclusion.

Outputs use a common prefix::

    <prefix>.details.tsv
    <prefix>.summary.json
    <prefix>.summary.md

If ``--output-prefix`` is omitted, ``foo.review.json`` becomes
``foo.decoded.*`` next to the review file.  Existing outputs require
``--force``; neither the review nor the manifest is ever rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DETAIL_FIELDS = (
    "manifest_index",
    "case_id",
    "mode",
    "cell",
    "selection_stratum",
    "selection_stratum_source",
    "candidate_roles",
    "review_occurrences",
    "raw_choice",
    "normalized_choice",
    "locked",
    "status",
    "winner_letter",
    "winner_role",
    "winner_label",
    "is_tie",
    "is_neither",
    "note",
)

CHOICE_FIELDS = (
    "winner",
    "judgment",
    "choice",
    "selection",
    "answer",
    "preference",
    "blind_winner_locked",
    "blind_winner",
)
LOCK_FIELDS = ("locked", "confirmed", "finalized")
TIE_ALIASES = {
    "tie",
    "tied",
    "equal",
    "same",
    "draw",
    "no_preference",
    "no-preference",
    "难分",
    "平局",
    "一样",
}
NEITHER_ALIASES = {
    "neither",
    "none",
    "neither_one",
    "neither-one",
    "both_bad",
    "both-bad",
    "两个都不像",
    "都不像",
}
STRATUM_ORDER = {"ref-bound": 0, "src-bound": 1, "ambiguous": 2, "missing": 3, "unknown": 4}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Decode a blinded review with its private manifest mapping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--review", required=True, help="Exported review JSON (read-only).")
    ap.add_argument("--manifest", required=True, help="Private page manifest JSON (read-only).")
    ap.add_argument(
        "--output-prefix",
        default="",
        help="Output prefix. Empty derives <review-name>.decoded next to the review.",
    )
    ap.add_argument("--force", action="store_true", help="Replace existing decoder outputs only.")
    return ap.parse_args()


def load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a top-level JSON object: {path}")
    return data


def object_list(data: dict[str, Any], keys: Iterable[str], *, label: str) -> list[dict[str, Any]]:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"{label}.{key} must be a JSON array")
        bad = [i for i, row in enumerate(value) if not isinstance(row, dict)]
        if bad:
            raise ValueError(f"{label}.{key} contains non-object entries at indexes {bad[:5]}")
        return list(value)
    raise ValueError(f"{label} has none of the expected arrays: {', '.join(keys)}")


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "locked", "confirmed"}:
        return True
    if text in {"0", "false", "no", "n", "off", "unlocked"}:
        return False
    return None


def pick_choice(row: dict[str, Any]) -> tuple[str, str]:
    for key in CHOICE_FIELDS:
        if key in row and row.get(key) not in (None, ""):
            return str(row[key]).strip(), key
    return "", ""


def locked_state(row: dict[str, Any], raw_choice: str) -> tuple[bool, bool]:
    """Return (locked, explicit_but_invalid). Missing lock fields imply final if answered."""

    for key in LOCK_FIELDS:
        if key not in row:
            continue
        parsed = parse_bool(row.get(key))
        if parsed is None:
            return False, True
        return parsed, False
    return bool(raw_choice), False


def normalize_choice(value: str) -> str:
    text = value.strip()
    lowered = text.lower().replace(" ", "_")
    if lowered in TIE_ALIASES:
        return "tie"
    if lowered in NEITHER_ALIASES:
        return "neither"
    if lowered == "left":
        return "A"
    if lowered == "right":
        return "B"
    if re.fullmatch(r"[A-Za-z]", text):
        return text.upper()
    return text


def manifest_roles(manifest: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    role_order: list[str] = []
    labels: dict[str, str] = {}
    raw = manifest.get("roles")
    if isinstance(raw, dict):
        for key, metadata in raw.items():
            role = str(key)
            role_order.append(role)
            if isinstance(metadata, dict):
                labels[role] = str(metadata.get("label") or role)
            else:
                labels[role] = str(metadata or role)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                role = str(item.get("role") or item.get("id") or "")
                label = str(item.get("label") or role)
            else:
                role = str(item)
                label = role
            if role:
                role_order.append(role)
                labels[role] = label
    return role_order, labels


def extract_mapping(case: dict[str, Any], labels: dict[str, str]) -> dict[str, dict[str, str]]:
    raw: Any = None
    for key in ("candidate_mapping", "blind_mapping", "mapping"):
        if isinstance(case.get(key), dict):
            raw = case[key]
            break
    if raw is None and isinstance(case.get("candidates"), list):
        raw = {}
        for candidate in case["candidates"]:
            if not isinstance(candidate, dict):
                continue
            letter = candidate.get("letter") or candidate.get("choice") or candidate.get("id")
            if letter not in (None, ""):
                raw[str(letter)] = candidate
    if not isinstance(raw, dict):
        return {}

    out: dict[str, dict[str, str]] = {}
    for blind_key, metadata in raw.items():
        letter = normalize_choice(str(blind_key))
        if letter in {"tie", "neither"}:
            continue
        if isinstance(metadata, dict):
            role = str(metadata.get("role") or metadata.get("system") or metadata.get("id") or "")
            label = str(metadata.get("label") or labels.get(role) or role)
        else:
            role = str(metadata)
            label = labels.get(role, role)
        if role:
            out[letter] = {"role": role, "label": label}
            labels.setdefault(role, label)
    return out


def default_output_prefix(review_path: Path) -> Path:
    name = review_path.name
    if name.endswith(".review.json"):
        stem = name[: -len(".review.json")]
    elif name.endswith(".json"):
        stem = name[: -len(".json")]
    else:
        stem = name
    return review_path.parent / f"{stem}.decoded"


def pct(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def fmt_pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number * 100:.1f}%" if math.isfinite(number) else "—"


def md(value: Any) -> str:
    return str(value if value not in (None, "") else "—").replace("|", "\\|").replace("\n", "<br>")


def summarize_bucket(
    rows: list[dict[str, Any]], role_order: list[str], labels: dict[str, str]
) -> dict[str, Any]:
    wins = Counter(str(row["winner_role"]) for row in rows if row["status"] == "win")
    tie_count = sum(row["status"] == "tie" for row in rows)
    neither_count = sum(row["status"] == "neither" for row in rows)
    valid_decisions = sum(row["status"] in {"win", "tie", "neither"} for row in rows)
    decisive = sum(wins.values())
    incomplete = sum(row["status"] in {"missing_review", "unlocked", "no_choice"} for row in rows)
    invalid = len(rows) - valid_decisions - incomplete
    roles: dict[str, Any] = {}
    for role in role_order:
        count = wins.get(role, 0)
        roles[role] = {
            "label": labels.get(role, role),
            "wins": count,
            "win_rate_all_valid_decisions": pct(count, valid_decisions),
            "share_among_decisive_role_wins": pct(count, decisive),
        }
    return {
        "manifest_case_count": len(rows),
        "valid_decisions": valid_decisions,
        "decisive_role_wins": decisive,
        "ties": tie_count,
        "tie_count": tie_count,
        "tie_rate": pct(tie_count, valid_decisions),
        "neither_count": neither_count,
        "neither_rate": pct(neither_count, valid_decisions),
        "incomplete": incomplete,
        "invalid": invalid,
        "roles": roles,
    }


def positive_int(value: Any, *, field: str, errors: list[str]) -> int | None:
    if isinstance(value, bool):
        errors.append(f"{field} must be a positive integer")
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be a positive integer")
        return None
    if number <= 0 or str(value).strip() not in {str(number), f"{number}.0"}:
        errors.append(f"{field} must be a positive integer")
        return None
    return number


def evaluate_decision_rule(
    raw_rule: Any,
    *,
    overall: dict[str, Any],
    role_order: list[str],
    labels: dict[str, str],
) -> dict[str, Any] | None:
    if raw_rule in (None, ""):
        return None
    if not isinstance(raw_rule, dict):
        return {"present": True, "valid": False, "errors": ["decision_rule must be an object"]}

    errors: list[str] = []
    target_role = str(raw_rule.get("target_role") or "")
    if not target_role:
        errors.append("target_role is required")
    elif target_role not in role_order:
        errors.append(f"target_role is not present in manifest roles/mappings: {target_role}")
    minimum_wins = positive_int(raw_rule.get("minimum_wins"), field="minimum_wins", errors=errors)
    total_cases = positive_int(raw_rule.get("total_cases"), field="total_cases", errors=errors)
    if minimum_wins is not None and total_cases is not None and minimum_wins > total_cases:
        errors.append("minimum_wins cannot exceed total_cases")
    nonwin_policy = parse_bool(raw_rule.get("tie_and_neither_are_nonwins"))
    if nonwin_policy is None:
        errors.append("tie_and_neither_are_nonwins must be a boolean")

    result: dict[str, Any] = {
        "present": True,
        "valid": not errors,
        "target_role": target_role,
        "target_label": labels.get(target_role, target_role),
        "minimum_wins": minimum_wins,
        "total_cases": total_cases,
        "tie_and_neither_are_nonwins": nonwin_policy,
        "errors": errors,
    }
    if errors:
        return result

    assert minimum_wins is not None
    assert total_cases is not None
    actual_wins = int(overall["roles"][target_role]["wins"])
    valid_decisions = int(overall["valid_decisions"])
    remaining_cases = max(total_cases - valid_decisions, 0)
    threshold_pass = actual_wins >= minimum_wins
    result.update(
        {
            "actual_wins": actual_wins,
            "threshold_pass": threshold_pass,
            "threshold_status": "PASS" if threshold_pass else "FAIL",
            "actual_fraction": f"{actual_wins}/{total_cases}",
            "threshold_fraction": f"{minimum_wins}/{total_cases}",
            "observed_valid_decisions": valid_decisions,
            "observed_manifest_cases": int(overall["manifest_case_count"]),
            "evaluation_complete": valid_decisions == total_cases,
            "remaining_cases": remaining_cases,
            "maximum_possible_wins": actual_wins + remaining_cases,
        }
    )
    return result


def string_list(value: Any, *, field: str, errors: list[str]) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        errors.append(f"{field} must be a string or array of strings")
        return []
    out: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{field} must contain only non-empty strings")
            continue
        text = item.strip()
        if text not in out:
            out.append(text)
    return out


def integer(value: Any, *, field: str, errors: list[str], default: int) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        errors.append(f"{field} must be an integer")
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be an integer")
        return default
    if str(value).strip() not in {str(number), f"{number}.0"}:
        errors.append(f"{field} must be an integer")
        return default
    return number


def decision_result_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        preferred = ["verdict", "label", "action", "code", "description"]
        parts: list[str] = []
        for key in preferred:
            item = value.get(key)
            if item not in (None, "") and str(item) not in parts:
                parts.append(str(item))
        if parts:
            return " / ".join(parts)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def evaluate_decision_rules(
    raw_config: Any,
    *,
    overall: dict[str, Any],
    role_order: list[str],
    labels: dict[str, str],
) -> dict[str, Any] | None:
    """Evaluate ordered threshold rules over role wins and non-role outcomes.

    Preferred manifest schema::

        "decision_rules": {
          "strategy": "highest_priority_match",
          "total_cases": 20,
          "require_complete": true,
          "rules": [{
            "id": "b2_clearly_better",
            "priority": 30,
            "minimum_count": 12,
            "count": {"role_wins": ["b2"], "outcomes": []},
            "result": {"verdict": "B2明显更好", "action": "可进30k"}
          }],
          "default_result": {"verdict": "未达三级判据"}
        }

    ``outcomes`` may contain ``tie`` and/or ``neither``.  Neither is counted
    only when explicitly listed.  A bare list of rules is also accepted, and
    older single-rule ``decision_rule`` remains evaluated independently.
    ``require_complete`` defaults to true, so threshold crossings are only
    provisional until ``valid_decisions == total_cases``.
    """

    if raw_config in (None, ""):
        return None
    config_errors: list[str] = []
    if isinstance(raw_config, list):
        config: dict[str, Any] = {"rules": raw_config}
    elif isinstance(raw_config, dict):
        config = raw_config
    else:
        return {"present": True, "valid": False, "errors": ["decision_rules must be an object or array"]}

    raw_rules = config.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        return {
            "present": True,
            "valid": False,
            "errors": ["decision_rules.rules must be a non-empty array"],
        }
    if any(not isinstance(rule, dict) for rule in raw_rules):
        return {
            "present": True,
            "valid": False,
            "errors": ["decision_rules.rules must contain only objects"],
        }

    strategy = str(config.get("strategy") or "highest_priority_match").strip().lower()
    strategy_aliases = {
        "highest_priority": "highest_priority_match",
        "priority": "highest_priority_match",
        "first_match": "ordered_first_match",
        "ordered": "ordered_first_match",
    }
    strategy = strategy_aliases.get(strategy, strategy)
    if strategy not in {"highest_priority_match", "ordered_first_match"}:
        config_errors.append(f"unsupported decision_rules.strategy: {strategy}")

    require_complete = parse_bool(config.get("require_complete", True))
    if require_complete is None:
        config_errors.append("decision_rules.require_complete must be a boolean")
        require_complete = True

    total_cases = positive_int(
        config.get("total_cases", overall["manifest_case_count"]),
        field="decision_rules.total_cases",
        errors=config_errors,
    )
    observed_valid = int(overall["valid_decisions"])
    default_result = config.get("default_result", config.get("default"))
    evaluated: list[dict[str, Any]] = []
    rule_ids: list[str] = []

    for index, raw_rule in enumerate(raw_rules):
        rule_errors: list[str] = []
        rule_id = str(raw_rule.get("id") or raw_rule.get("name") or f"rule_{index + 1}").strip()
        if not rule_id:
            rule_id = f"rule_{index + 1}"
        if rule_id in rule_ids:
            rule_errors.append(f"duplicate rule id: {rule_id}")
        rule_ids.append(rule_id)
        priority = integer(
            raw_rule.get("priority"),
            field=f"decision_rules.rules[{index}].priority",
            errors=rule_errors,
            default=len(raw_rules) - index,
        )
        minimum_count = positive_int(
            raw_rule.get("minimum_count", raw_rule.get("minimum_wins")),
            field=f"decision_rules.rules[{index}].minimum_count",
            errors=rule_errors,
        )
        rule_total = positive_int(
            raw_rule.get("total_cases", total_cases),
            field=f"decision_rules.rules[{index}].total_cases",
            errors=rule_errors,
        )
        if minimum_count is not None and rule_total is not None and minimum_count > rule_total:
            rule_errors.append("minimum_count cannot exceed total_cases")

        count_spec = raw_rule.get("count") or {}
        if not isinstance(count_spec, dict):
            rule_errors.append(f"decision_rules.rules[{index}].count must be an object")
            count_spec = {}
        roles_value = count_spec.get(
            "role_wins",
            raw_rule.get("role_wins", raw_rule.get("target_role")),
        )
        roles = string_list(
            roles_value,
            field=f"decision_rules.rules[{index}].count.role_wins",
            errors=rule_errors,
        )
        for role in roles:
            if role not in role_order:
                rule_errors.append(f"unknown role in count.role_wins: {role}")

        outcomes_value = count_spec.get("outcomes", raw_rule.get("include_outcomes"))
        outcomes_raw = string_list(
            outcomes_value,
            field=f"decision_rules.rules[{index}].count.outcomes",
            errors=rule_errors,
        )
        for flag, outcome in (("include_tie", "tie"), ("include_neither", "neither")):
            if flag not in raw_rule:
                continue
            enabled = parse_bool(raw_rule.get(flag))
            if enabled is None:
                rule_errors.append(f"decision_rules.rules[{index}].{flag} must be a boolean")
            elif enabled and outcome not in outcomes_raw:
                outcomes_raw.append(outcome)
        outcomes: list[str] = []
        for outcome in outcomes_raw:
            normalized = normalize_choice(outcome)
            if normalized not in {"tie", "neither"}:
                rule_errors.append(f"unsupported counted outcome: {outcome}")
            elif normalized not in outcomes:
                outcomes.append(normalized)
        if not roles and not outcomes:
            rule_errors.append("count must include at least one role_wins entry or outcome")

        role_components = {
            role: int(overall["roles"].get(role, {}).get("wins", 0)) for role in roles
        }
        outcome_components = {
            outcome: int(overall["tie_count"] if outcome == "tie" else overall["neither_count"])
            for outcome in outcomes
        }
        actual_count = sum(role_components.values()) + sum(outcome_components.values())
        matched = bool(not rule_errors and minimum_count is not None and actual_count >= minimum_count)
        result_value = raw_rule.get("result", raw_rule.get("conclusion"))
        if result_value in (None, ""):
            result_value = {
                "code": rule_id,
                "label": str(raw_rule.get("label") or rule_id),
            }
        evaluated.append(
            {
                "id": rule_id,
                "label": str(raw_rule.get("label") or rule_id),
                "declaration_index": index,
                "priority": priority,
                "valid": not rule_errors,
                "errors": rule_errors,
                "minimum_count": minimum_count,
                "total_cases": rule_total,
                "count": {"role_wins": roles, "outcomes": outcomes},
                "components": {
                    "role_wins": role_components,
                    "outcomes": outcome_components,
                },
                "actual_count": actual_count,
                "actual_fraction": f"{actual_count}/{rule_total}" if rule_total else None,
                "threshold_fraction": f"{minimum_count}/{rule_total}"
                if minimum_count is not None and rule_total is not None
                else None,
                "threshold_pass": matched,
                "threshold_status": "INVALID" if rule_errors else ("PASS" if matched else "FAIL"),
                "result": result_value,
                "result_text": decision_result_text(result_value),
            }
        )

    duplicate_ids = sorted(rule_id for rule_id, count in Counter(rule_ids).items() if count > 1)
    if duplicate_ids:
        config_errors.append(f"duplicate decision rule ids: {', '.join(duplicate_ids)}")
    rule_error_messages = [
        f"{rule['id']}: {error}"
        for rule in evaluated
        for error in rule["errors"]
    ]
    all_errors = [*config_errors, *rule_error_messages]
    all_valid = not all_errors
    if strategy == "ordered_first_match":
        selection_order = sorted(evaluated, key=lambda rule: rule["declaration_index"])
    else:
        selection_order = sorted(
            evaluated,
            key=lambda rule: (-int(rule["priority"]), int(rule["declaration_index"])),
        )
    assert total_cases is not None or all_errors
    remaining_cases = max((total_cases or 0) - observed_valid, 0)
    evaluation_complete = total_cases is not None and observed_valid == total_cases
    threshold_pass_rules = [rule for rule in selection_order if rule["threshold_pass"]]
    selection_blocked_incomplete = bool(require_complete and not evaluation_complete)
    matched_rules = threshold_pass_rules if all_valid and not selection_blocked_incomplete else []
    selected = matched_rules[0] if matched_rules else None
    if not all_valid:
        selection_status = "INVALID"
        selected_result = None
    elif selection_blocked_incomplete:
        selection_status = "INCOMPLETE"
        selected_result = None
    elif selected:
        selection_status = "MATCH"
        selected_result = selected["result"]
    elif default_result not in (None, ""):
        selection_status = "DEFAULT"
        selected_result = default_result
    else:
        selection_status = "NO_MATCH"
        selected_result = None

    return {
        "present": True,
        "valid": all_valid,
        "errors": all_errors,
        "strategy": strategy,
        "require_complete": require_complete,
        "counting_policy": (
            "only role_wins and outcomes explicitly listed by each rule are counted; "
            "unlisted tie/neither outcomes are non-wins"
        ),
        "total_cases": total_cases,
        "observed_valid_decisions": observed_valid,
        "observed_manifest_cases": int(overall["manifest_case_count"]),
        "evaluation_complete": evaluation_complete,
        "remaining_cases": remaining_cases,
        "selection_blocked_incomplete": selection_blocked_incomplete,
        "rules": evaluated,
        "selection_order": [rule["id"] for rule in selection_order],
        "threshold_pass_rule_ids": [rule["id"] for rule in threshold_pass_rules],
        "provisional_matched_rule_ids": [rule["id"] for rule in threshold_pass_rules]
        if selection_blocked_incomplete
        else [],
        "matched_rule_ids": [rule["id"] for rule in matched_rules],
        "selected_rule_id": selected["id"] if selected else None,
        "selected_priority": selected["priority"] if selected else None,
        "selection_status": selection_status,
        "selected_result": selected_result,
        "selected_result_text": decision_result_text(selected_result),
        "default_result": default_result,
    }


def decode(
    review: dict[str, Any], manifest: dict[str, Any], *, review_path: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_cases = object_list(manifest, ("cases", "items"), label="manifest")
    review_items = object_list(review, ("items", "responses", "reviews"), label="review")
    role_order, labels = manifest_roles(manifest)

    manifest_counts = Counter(str(row.get("case_id") or "") for row in manifest_cases)
    review_counts = Counter(str(row.get("case_id") or "") for row in review_items)
    review_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in review_items:
        review_by_case[str(row.get("case_id") or "")].append(row)

    mappings: list[dict[str, dict[str, str]]] = []
    for case in manifest_cases:
        mapping = extract_mapping(case, labels)
        mappings.append(mapping)
        for metadata in mapping.values():
            role = metadata["role"]
            if role not in role_order:
                role_order.append(role)

    details: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    status_case_ids: dict[str, list[str]] = defaultdict(list)

    for position, (case, mapping) in enumerate(zip(manifest_cases, mappings), start=1):
        case_id = str(case.get("case_id") or "")
        responses = review_by_case.get(case_id, [])
        raw_choices = [pick_choice(row)[0] for row in responses]
        note_values = [str(row.get("note") or "").strip() for row in responses]
        note_values = [value for value in note_values if value]
        for occurrence, row in enumerate(responses, start=1):
            note = str(row.get("note") or "").strip()
            if note:
                notes.append({"case_id": case_id, "review_occurrence": occurrence, "note": note})

        stratum = str(
            case.get("selection_stratum")
            or case.get("selection_bucket")
            or case.get("joint_stratum")
            or (case.get("diagnostics") or {}).get("dual_encoder_stratum")
            or "missing"
        )
        candidate_roles = ";".join(
            f"{letter}={metadata['role']}" for letter, metadata in sorted(mapping.items())
        )
        detail: dict[str, Any] = {
            "manifest_index": case.get("index") or position,
            "case_id": case_id,
            "mode": case.get("mode") or manifest.get("mode") or "",
            "cell": case.get("cell") or "",
            "selection_stratum": stratum,
            "selection_stratum_source": (
                case.get("selection_stratum_source") or case.get("selection_source") or ""
            ),
            "candidate_roles": candidate_roles,
            "review_occurrences": len(responses),
            "raw_choice": " | ".join(raw_choices),
            "normalized_choice": "",
            "locked": "",
            "status": "",
            "winner_letter": "",
            "winner_role": "",
            "winner_label": "",
            "is_tie": False,
            "is_neither": False,
            "note": " || ".join(note_values),
            "_source": "manifest",
        }

        if not case_id:
            status = "missing_manifest_case_id"
        elif manifest_counts[case_id] > 1:
            status = "duplicate_manifest_case"
        elif len(responses) == 0:
            status = "missing_review"
        elif len(responses) > 1:
            status = "duplicate_review_case"
        else:
            response = responses[0]
            raw_choice, _choice_field = pick_choice(response)
            choice = normalize_choice(raw_choice)
            locked, invalid_locked = locked_state(response, raw_choice)
            detail["raw_choice"] = raw_choice
            detail["normalized_choice"] = choice
            detail["locked"] = locked
            if invalid_locked:
                status = "invalid_locked_value"
            elif not choice:
                status = "no_choice"
            elif not locked:
                status = "unlocked"
            elif not mapping:
                status = "missing_candidate_mapping"
            elif choice == "tie":
                status = "tie"
                detail["is_tie"] = True
            elif choice == "neither":
                status = "neither"
                detail["is_neither"] = True
            elif choice not in mapping:
                status = "unknown_choice"
            else:
                winner = mapping[choice]
                status = "win"
                detail["winner_letter"] = choice
                detail["winner_role"] = winner["role"]
                detail["winner_label"] = winner["label"]
        detail["status"] = status
        details.append(detail)
        if status not in {"win", "tie", "neither"}:
            status_case_ids[status].append(case_id or f"<manifest-index-{position}>")

    manifest_ids = set(manifest_counts)
    for occurrence, row in enumerate(review_items, start=1):
        case_id = str(row.get("case_id") or "")
        if case_id in manifest_ids and case_id:
            continue
        raw_choice, _ = pick_choice(row)
        choice = normalize_choice(raw_choice)
        note = str(row.get("note") or "").strip()
        status = "unknown_review_case" if case_id else "missing_review_case_id"
        details.append(
            {
                "manifest_index": "",
                "case_id": case_id,
                "mode": "",
                "cell": "",
                "selection_stratum": "unknown",
                "selection_stratum_source": "",
                "candidate_roles": "",
                "review_occurrences": 1,
                "raw_choice": raw_choice,
                "normalized_choice": choice,
                "locked": locked_state(row, raw_choice)[0],
                "status": status,
                "winner_letter": "",
                "winner_role": "",
                "winner_label": "",
                "is_tie": choice == "tie",
                "is_neither": choice == "neither",
                "note": note,
                "_source": "review_only",
            }
        )
        status_case_ids[status].append(case_id or f"<review-index-{occurrence}>")
        if note:
            notes.append({"case_id": case_id, "review_occurrence": occurrence, "note": note})

    manifest_details = [row for row in details if row["_source"] == "manifest"]
    overall = summarize_bucket(manifest_details, role_order, labels)
    strata: dict[str, Any] = {}
    stratum_names = sorted(
        {str(row["selection_stratum"]) for row in manifest_details},
        key=lambda value: (STRATUM_ORDER.get(value, 100), value),
    )
    for stratum in stratum_names:
        bucket = [row for row in manifest_details if row["selection_stratum"] == stratum]
        strata[stratum] = summarize_bucket(bucket, role_order, labels)

    candidate_counts = [len(mapping) for mapping in mappings if mapping]
    explicit_semantics = str(manifest.get("response_semantics") or "").lower()
    if "pair" in explicit_semantics or (candidate_counts and max(candidate_counts) == 2):
        review_design = "pairwise"
    elif candidate_counts:
        review_design = "multiway"
    else:
        review_design = "unknown"

    page_id_review = str(review.get("page_id") or "")
    page_id_manifest = str(manifest.get("page_id") or "")
    decision_rule = evaluate_decision_rule(
        manifest.get("decision_rule"),
        overall=overall,
        role_order=role_order,
        labels=labels,
    )
    decision_rules = evaluate_decision_rules(
        manifest.get("decision_rules"),
        overall=overall,
        role_order=role_order,
        labels=labels,
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_path": str(review_path),
        "manifest_path": str(manifest_path),
        "page_id": {"review": page_id_review, "manifest": page_id_manifest},
        "page_id_match": not page_id_review or not page_id_manifest or page_id_review == page_id_manifest,
        "review_design": review_design,
        "non_role_outcome_policy": (
            "ties and neither outcomes are counted separately and never assigned fractionally to any role"
        ),
        "tie_policy": "ties are counted separately and never assigned fractionally to any role",
        "neither_policy": (
            "neither outcomes are counted separately and never assigned fractionally to any role"
        ),
        "rate_denominators": {
            "win_rate_all_valid_decisions": "role wins / (all role wins + ties + neither)",
            "share_among_decisive_role_wins": "role wins / all role wins (excluding tie and neither)",
            "tie_rate": "ties / (all role wins + ties + neither)",
            "neither_rate": "neither / (all role wins + ties + neither)",
        },
        "counts": {
            "manifest_case_entries": len(manifest_cases),
            "manifest_unique_case_ids": len({key for key in manifest_counts if key}),
            "review_items": len(review_items),
            "review_unique_case_ids": len({key for key in review_counts if key}),
            "unknown_review_items": sum(row["_source"] == "review_only" for row in details),
            "notes": len(notes),
        },
        "role_order": role_order,
        "overall": overall,
        "by_selection_stratum": strata,
        "decision_rule": decision_rule,
        "decision_rules": decision_rules,
        "diagnostics": {
            "duplicate_manifest_case_ids": sorted(
                case_id for case_id, count in manifest_counts.items() if case_id and count > 1
            ),
            "duplicate_review_case_ids": sorted(
                case_id for case_id, count in review_counts.items() if case_id and count > 1
            ),
            "status_case_ids": {key: value for key, value in sorted(status_case_ids.items())},
            "notes": notes,
        },
        "details": [{key: row.get(key, "") for key in DETAIL_FIELDS} for row in details],
    }
    return details, summary


def write_tsv(path: Path, details: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in details:
            writer.writerow(row)


def render_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# Blind review decoded summary",
        "",
        f"- Review design: `{md(summary['review_design'])}`",
        f"- Page ID: review `{md(summary['page_id']['review'])}` / manifest `{md(summary['page_id']['manifest'])}`",
        f"- Page ID match: `{str(summary['page_id_match']).lower()}`",
        f"- Manifest cases: {overall['manifest_case_count']}",
        (
            f"- Valid decisions: {overall['valid_decisions']} "
            f"(role wins {overall['decisive_role_wins']}, ties {overall['ties']}, "
            f"neither {overall['neither_count']})"
        ),
        f"- Incomplete: {overall['incomplete']}; invalid: {overall['invalid']}",
        "- Tie/neither policy: both are reported separately and are not split between roles.",
        "",
        "## Overall role wins",
        "",
        "| Role | Label | Wins | Win rate / all valid | Share / decisive wins |",
        "|---|---|---:|---:|---:|",
    ]
    for role in summary["role_order"]:
        row = overall["roles"][role]
        lines.append(
            f"| {md(role)} | {md(row['label'])} | {row['wins']} | "
            f"{fmt_pct(row['win_rate_all_valid_decisions'])} | "
            f"{fmt_pct(row['share_among_decisive_role_wins'])} |"
        )
    lines.extend(
        [
            "",
            f"Tie: **{overall['ties']}** / {overall['valid_decisions']} ({fmt_pct(overall['tie_rate'])}).",
            (
                f"Neither: **{overall['neither_count']}** / {overall['valid_decisions']} "
                f"({fmt_pct(overall['neither_rate'])})."
            ),
        ]
    )

    decision_rules = summary.get("decision_rules")
    if decision_rules is not None:
        lines.extend(["", "## Ordered decision rules", ""])
        lines.extend(
            [
                f"- Strategy: `{md(decision_rules.get('strategy'))}`",
                (
                    "- Require complete review before selection: "
                    f"`{str(decision_rules.get('require_complete')).lower()}`"
                ),
                (
                    f"- Evaluation complete: `{str(decision_rules.get('evaluation_complete')).lower()}`; "
                    f"valid decisions {decision_rules.get('observed_valid_decisions')}/"
                    f"{decision_rules.get('total_cases')}"
                ),
            ]
        )
        if decision_rules.get("valid"):
            lines.extend(
                [
                    "",
                    "| Priority | Rule | Counted outcomes | Actual | Required | Match | Result |",
                    "|---:|---|---|---:|---:|---|---|",
                ]
            )
            rules_by_id = {rule["id"]: rule for rule in decision_rules["rules"]}
            for rule_id in decision_rules["selection_order"]:
                rule = rules_by_id[rule_id]
                counted = [f"wins({role})" for role in rule["count"]["role_wins"]]
                counted.extend(rule["count"]["outcomes"])
                lines.append(
                    f"| {rule['priority']} | {md(rule['id'])} | {md(' + '.join(counted))} | "
                    f"{rule['actual_count']} | ≥ {rule['minimum_count']} | "
                    f"{rule['threshold_status']} | {md(rule['result_text'])} |"
                )
            lines.extend(
                [
                    "",
                    f"- Selection status: **{md(decision_rules['selection_status'])}**",
                    f"- Selected rule: `{md(decision_rules.get('selected_rule_id'))}`",
                    f"- Final conclusion: **{md(decision_rules.get('selected_result_text'))}**",
                ]
            )
            if decision_rules["selection_status"] == "INCOMPLETE":
                provisional = ", ".join(decision_rules.get("provisional_matched_rule_ids", []))
                lines.extend(
                    [
                        f"- Provisional threshold passes: `{md(provisional)}`",
                        "- No final conclusion is selected until all required cases are completed.",
                    ]
                )
        else:
            all_errors = list(decision_rules.get("errors", []))
            for rule in decision_rules.get("rules", []):
                all_errors.extend(rule.get("errors", []))
            lines.append(f"Invalid decision rules: {md('; '.join(all_errors))}")

    decision_rule = summary.get("decision_rule")
    if decision_rule is not None:
        lines.extend(["", "## Legacy single decision threshold", ""])
        if decision_rule.get("valid"):
            lines.extend(
                [
                    (
                        f"- Target: `{md(decision_rule['target_role'])}` "
                        f"({md(decision_rule['target_label'])})"
                    ),
                    (
                        f"- Actual wins: **{decision_rule['actual_fraction']}**; required: "
                        f"**≥ {decision_rule['threshold_fraction']}**"
                    ),
                    f"- Threshold result: **{decision_rule['threshold_status']}**",
                    (
                        f"- Evaluation complete: `{str(decision_rule['evaluation_complete']).lower()}`; "
                        f"valid decisions {decision_rule['observed_valid_decisions']}/"
                        f"{decision_rule['total_cases']}"
                    ),
                    (
                        "- Tie and neither are non-wins: "
                        f"`{str(decision_rule['tie_and_neither_are_nonwins']).lower()}`"
                    ),
                ]
            )
        else:
            lines.append(f"Invalid decision rule: {md('; '.join(decision_rule.get('errors', [])))}")

    lines.extend(
        [
            "",
            "## Wins by selection stratum",
            "",
            "| Stratum | Valid | Tie count/rate | Neither count/rate | Role | Wins | Win rate / all valid | Share / decisive wins |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for stratum, bucket in summary["by_selection_stratum"].items():
        for role in summary["role_order"]:
            row = bucket["roles"][role]
            lines.append(
                f"| {md(stratum)} | {bucket['valid_decisions']} | "
                f"{bucket['ties']} / {fmt_pct(bucket['tie_rate'])} | "
                f"{bucket['neither_count']} / {fmt_pct(bucket['neither_rate'])} | "
                f"{md(role)} | {row['wins']} | "
                f"{fmt_pct(row['win_rate_all_valid_decisions'])} | "
                f"{fmt_pct(row['share_among_decisive_role_wins'])} |"
            )

    diagnostics = summary["diagnostics"]
    lines.extend(["", "## Diagnostics", ""])
    lines.append(f"- Duplicate manifest cases: {md(', '.join(diagnostics['duplicate_manifest_case_ids']))}")
    lines.append(f"- Duplicate review cases: {md(', '.join(diagnostics['duplicate_review_case_ids']))}")
    if diagnostics["status_case_ids"]:
        for status, case_ids in diagnostics["status_case_ids"].items():
            lines.append(f"- `{md(status)}`: {md(', '.join(case_ids))}")
    else:
        lines.append("- No incomplete, invalid, duplicate, or unknown cases.")

    lines.extend(["", "## Notes", ""])
    if diagnostics["notes"]:
        lines.extend(["| Case ID | Review occurrence | Note |", "|---|---:|---|"])
        for note in diagnostics["notes"]:
            lines.append(
                f"| {md(note['case_id'])} | {note['review_occurrence']} | {md(note['note'])} |"
            )
    else:
        lines.append("No notes.")

    lines.extend(
        [
            "",
            "## Per-case decoded details",
            "",
            "| # | Case ID | Stratum | Choice | Status | Winner role | Note |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for row in summary["details"]:
        lines.append(
            f"| {md(row['manifest_index'])} | {md(row['case_id'])} | {md(row['selection_stratum'])} | "
            f"{md(row['normalized_choice'])} | {md(row['status'])} | {md(row['winner_role'])} | "
            f"{md(row['note'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    review_path = Path(args.review).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_prefix = (
        Path(args.output_prefix).expanduser().resolve()
        if args.output_prefix
        else default_output_prefix(review_path)
    )
    output_paths = {
        "tsv": Path(str(output_prefix) + ".details.tsv"),
        "json": Path(str(output_prefix) + ".summary.json"),
        "md": Path(str(output_prefix) + ".summary.md"),
    }
    input_paths = {review_path, manifest_path}
    overlap = input_paths.intersection(output_paths.values())
    if overlap:
        raise ValueError(f"Refusing to overwrite an input file: {next(iter(overlap))}")
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Decoder output exists (use --force): {joined}")

    review = load_object(review_path, label="review")
    manifest = load_object(manifest_path, label="manifest")
    details, summary = decode(
        review,
        manifest,
        review_path=review_path,
        manifest_path=manifest_path,
    )
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(output_paths["tsv"], details)
    output_paths["json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_paths["md"].write_text(render_markdown(summary), encoding="utf-8")

    overall = summary["overall"]
    print(f"[blind-review] design={summary['review_design']} page_id_match={summary['page_id_match']}")
    print(
        "[blind-review] "
        f"valid={overall['valid_decisions']} decisive={overall['decisive_role_wins']} "
        f"ties={overall['ties']} neither={overall['neither_count']} "
        f"incomplete={overall['incomplete']} invalid={overall['invalid']}"
    )
    for role in summary["role_order"]:
        role_summary = overall["roles"][role]
        print(
            f"[blind-review] role={role} wins={role_summary['wins']} "
            f"rate={fmt_pct(role_summary['win_rate_all_valid_decisions'])}"
        )
    decision_rules = summary.get("decision_rules")
    if decision_rules is not None:
        if decision_rules.get("valid"):
            print(
                "[blind-review] decision_rules="
                f"{decision_rules['selection_status']} selected={decision_rules['selected_rule_id']} "
                f"result={decision_rules['selected_result_text']}"
            )
            if decision_rules["selection_status"] == "INCOMPLETE":
                print(
                    "[blind-review] provisional_threshold_passes="
                    f"{','.join(decision_rules['provisional_matched_rule_ids']) or 'none'}"
                )
            for rule in decision_rules["rules"]:
                print(
                    f"[blind-review] rule={rule['id']} priority={rule['priority']} "
                    f"actual={rule['actual_count']} minimum={rule['minimum_count']} "
                    f"status={rule['threshold_status']}"
                )
        else:
            print(f"[blind-review] decision_rules=INVALID errors={'; '.join(decision_rules['errors'])}")
    decision_rule = summary.get("decision_rule")
    if decision_rule is not None:
        if decision_rule.get("valid"):
            print(
                "[blind-review] threshold="
                f"{decision_rule['threshold_status']} target={decision_rule['target_role']} "
                f"actual={decision_rule['actual_wins']} minimum={decision_rule['minimum_wins']} "
                f"total={decision_rule['total_cases']}"
            )
        else:
            print(f"[blind-review] threshold=INVALID errors={'; '.join(decision_rule['errors'])}")
    for kind, path in output_paths.items():
        print(f"[blind-review] {kind}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
