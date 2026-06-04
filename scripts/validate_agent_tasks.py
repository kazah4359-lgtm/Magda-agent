"""Validate the Magda Agent self-improvement task manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


VALID_STATUSES = {"todo", "in_progress", "done", "blocked"}
VALID_RISKS = {"low", "medium", "high", "critical"}
REQUIRED_TASK_FIELDS = {
    "id",
    "status",
    "area",
    "risk",
    "title",
    "description",
    "allowed_paths",
    "acceptance",
}


class ValidationError(ValueError):
    """Raised when the task manifest violates the expected schema."""


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON task manifest from disk."""
    try:
        with path.open("r", encoding="utf-8") as manifest_file:
            data = json.load(manifest_file)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValidationError("manifest root must be an object")
    return data


def validate_manifest(data: dict[str, Any]) -> list[str]:
    """Validate a task manifest and return human-readable warnings."""
    warnings: list[str] = []

    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValidationError("schema_version must be 1")

    risk_levels = data.get("risk_levels")
    if set(risk_levels or []) != VALID_RISKS:
        raise ValidationError(f"risk_levels must be exactly {sorted(VALID_RISKS)}")

    merge_policy = data.get("merge_policy")
    if not isinstance(merge_policy, dict):
        raise ValidationError("merge_policy must be an object")
    missing_policy = VALID_RISKS.difference(merge_policy)
    if missing_policy:
        raise ValidationError(f"merge_policy missing risk levels: {sorted(missing_policy)}")

    replenishment_policy = data.get("replenishment_policy")
    minimum_todo_tasks = validate_replenishment_policy(replenishment_policy)

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValidationError("tasks must be a non-empty array")

    seen_ids: set[str] = set()
    todo_count = 0
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValidationError(f"task #{index + 1} must be an object")
        validate_task(task, index)

        task_id = task["id"]
        if task_id in seen_ids:
            raise ValidationError(f"duplicate task id: {task_id}")
        seen_ids.add(task_id)

        if task["status"] == "todo":
            todo_count += 1

    if todo_count == 0:
        warnings.append("no todo tasks remain")
    elif todo_count < minimum_todo_tasks:
        warnings.append(
            f"todo task pool is low: {todo_count} remaining, minimum is {minimum_todo_tasks}"
        )

    return warnings


def validate_replenishment_policy(policy: Any) -> int:
    """Validate task replenishment policy and return the minimum todo count."""
    if not isinstance(policy, dict):
        raise ValidationError("replenishment_policy must be an object")

    minimum_todo_tasks = policy.get("minimum_todo_tasks")
    if not isinstance(minimum_todo_tasks, int) or minimum_todo_tasks < 0:
        raise ValidationError("replenishment_policy.minimum_todo_tasks must be a non-negative integer")

    batch_size = policy.get("batch_size")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValidationError("replenishment_policy.batch_size must be a positive integer")

    allowed_risks = policy.get("allowed_risks_for_generated_tasks")
    if not isinstance(allowed_risks, list) or not allowed_risks:
        raise ValidationError("replenishment_policy.allowed_risks_for_generated_tasks must be non-empty")

    unknown_risks = set(allowed_risks).difference(VALID_RISKS)
    if unknown_risks:
        raise ValidationError(
            f"replenishment_policy contains unknown risk levels: {sorted(unknown_risks)}"
        )
    if any(risk in {"high", "critical"} for risk in allowed_risks):
        raise ValidationError("generated replenishment tasks may only be low or medium risk")

    instruction = policy.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValidationError("replenishment_policy.instruction must be a non-empty string")

    return minimum_todo_tasks


def validate_task(task: dict[str, Any], index: int) -> None:
    """Validate one task object from the manifest."""
    missing = REQUIRED_TASK_FIELDS.difference(task)
    if missing:
        raise ValidationError(f"task #{index + 1} missing fields: {sorted(missing)}")

    task_id = task["id"]
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValidationError(f"task #{index + 1} id must be a non-empty string")

    status = task["status"]
    if status not in VALID_STATUSES:
        raise ValidationError(f"task {task_id}: unknown status {status!r}")

    risk = task["risk"]
    if risk not in VALID_RISKS:
        raise ValidationError(f"task {task_id}: unknown risk {risk!r}")

    for field_name in ("area", "title", "description"):
        value = task[field_name]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"task {task_id}: {field_name} must be a non-empty string")

    allowed_paths = task["allowed_paths"]
    if not isinstance(allowed_paths, list) or not allowed_paths:
        raise ValidationError(f"task {task_id}: allowed_paths must be a non-empty array")
    for pattern in allowed_paths:
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValidationError(f"task {task_id}: allowed_paths entries must be strings")
        if pattern.startswith("/") or ".." in Path(pattern).parts:
            raise ValidationError(f"task {task_id}: unsafe allowed path pattern {pattern!r}")

    acceptance = task["acceptance"]
    if not isinstance(acceptance, list) or not acceptance:
        raise ValidationError(f"task {task_id}: acceptance must be a non-empty array")
    for criterion in acceptance:
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValidationError(f"task {task_id}: acceptance entries must be strings")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for task manifest validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default="agent_tasks.json")
    args = parser.parse_args(argv)

    try:
        data = load_manifest(Path(args.manifest))
        warnings = validate_manifest(data)
    except ValidationError as exc:
        print(f"agent task validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"agent task validation passed: {args.manifest}")
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
