"""Small CLI bridge that lets Codex/Jules work with Magda task manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("agent_tasks.json")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load the agent task manifest from disk."""
    with path.open("r", encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    return data


def iter_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return task objects from a manifest."""
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("manifest tasks must be a list")
    return [task for task in tasks if isinstance(task, dict)]


def todo_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return tasks with status todo."""
    return [task for task in iter_tasks(manifest) if task.get("status") == "todo"]


def next_task(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Return the next todo task in manifest order."""
    tasks = todo_tasks(manifest)
    return tasks[0] if tasks else None


def task_by_id(manifest: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    """Find one task by id."""
    for task in iter_tasks(manifest):
        if task.get("id") == task_id:
            return task
    return None


def queue_status(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return summary information about the task queue."""
    tasks = iter_tasks(manifest)
    todos = todo_tasks(manifest)
    policy = manifest.get("replenishment_policy") or {}
    minimum = int(policy.get("minimum_todo_tasks", 0))
    return {
        "total_tasks": len(tasks),
        "todo_tasks": len(todos),
        "minimum_todo_tasks": minimum,
        "needs_replenishment": len(todos) < minimum,
        "next_task_id": todos[0].get("id") if todos else None,
    }


def render_prompt(task: dict[str, Any]) -> str:
    """Render a focused prompt for Codex/Jules from one task object."""
    allowed_paths = "\n".join(f"- {path}" for path in task.get("allowed_paths", []))
    acceptance = "\n".join(f"- {item}" for item in task.get("acceptance", []))
    return f"""You are working on Magda Agent.

Read first:
- AGENTS.md
- docs/jules_autonomous_loop.md
- docs/cognitive_architecture.md
- docs/hermes_inspired_feature_plan.md
- docs/codex_worker_plan.md

Task id: {task.get("id")}
Title: {task.get("title")}
Area: {task.get("area")}
Risk: {task.get("risk")}

Description:
{task.get("description")}

Allowed paths:
{allowed_paths}

Acceptance criteria:
{acceptance}

Rules:
- Keep the PR scoped to this task id.
- Do not touch files outside allowed paths unless the task manifest is updated with a clear reason.
- Update agent_tasks.json when the task is complete.
- If you discover new work, add new todo tasks instead of widening this PR.
"""


def validate_manifest(path: Path) -> int:
    """Run the repository manifest validator."""
    try:
        from scripts.validate_agent_tasks import load_manifest as load_for_validation
        from scripts.validate_agent_tasks import validate_manifest as validate

        warnings = validate(load_for_validation(path))
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"validation passed: {path}")
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to agent_tasks.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Validate the task manifest")
    subparsers.add_parser("status", help="Print queue status as JSON")
    subparsers.add_parser("next-task", help="Print the next todo task as JSON")

    prompt_parser = subparsers.add_parser("render-prompt", help="Render a Codex prompt for a task")
    prompt_parser.add_argument("--task-id", help="Task id to render. Defaults to the next todo task.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)

    if args.command == "validate":
        return validate_manifest(manifest_path)

    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        print(f"failed to load manifest: {exc}", file=sys.stderr)
        return 1

    if args.command == "status":
        print(json.dumps(queue_status(manifest), ensure_ascii=False, indent=2))
        return 0

    if args.command == "next-task":
        task = next_task(manifest)
        if task is None:
            print("no todo tasks found", file=sys.stderr)
            return 2
        print(json.dumps(task, ensure_ascii=False, indent=2))
        return 0

    if args.command == "render-prompt":
        task = task_by_id(manifest, args.task_id) if args.task_id else next_task(manifest)
        if task is None:
            print("task not found", file=sys.stderr)
            return 2
        print(render_prompt(task))
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
