"""Mark duplicate todo tasks as blocked when equivalent work is already done."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# todo task id -> done task id that already covers this work
DUPLICATE_OF_DONE: dict[str, str] = {
    "mcp-tools-export": "mcp-tool-export",
    "mcp-tools-exporter": "mcp-tool-export",
    "mcp-compatibility": "mcp-tool-export",
    "mcp-agent-tools-export-v3": "mcp-tool-export",
    "mcp-tools-export-v2": "mcp-tool-export",
    "mcp-action-tools-v2": "mcp-tool-export",
    "mcp-action-tools-registry-new": "mcp-tool-export",
    "mcp-skill-export": "mcp-tool-export",
    "mcp-skill-export-integration": "mcp-tool-export",
    "mcp-server-prefixed-tool-names": "mcp-tool-export",
    "mcp-action-tool-governance": "mcp-tool-export",
    "agent-guardrails-policy-layer": "policy-layer-tool-calls",
    "agent-guardrails-policy-layer-v2": "policy-layer-tool-calls",
    "agent-guardrails-policy-layer-v3": "policy-layer-tool-calls",
    "agent-guard-runtime-governance": "policy-layer-tool-calls",
    "agent-guard-policy": "policy-layer-tool-calls",
    "agent-guard-runtime-policy": "policy-layer-tool-calls",
    "agent-guard-runtime-policy-layer": "policy-layer-tool-calls",
    "acs-runtime-safety": "policy-layer-tool-calls",
    "agent-control-specification": "policy-layer-tool-calls",
    "prempti-audit-trail-interceptor-new": "tool-call-audit-trail",
    "context-engine-plugin-architecture": "context-engine-plugin",
    "context-engine-plugin-interface": "context-engine-plugin",
    "context-engine-plugins": "context-engine-plugin",
    "context-engine-compression-v2": "context-engine-plugin",
    "context-compression-selective-retrieval": "context-engine-plugin",
    "context-engine-canvas-vis-new": "context-engine-plugin",
    "online-rl-from-feedback": "online-learning-from-feedback",
    "online-rl-from-feedback-v2": "online-learning-from-feedback",
    "online-rl-feedback": "online-learning-from-feedback",
    "openclaw-rl-online-learning": "online-learning-from-feedback",
    "openclaw-rl-implicit-feedback": "online-learning-from-feedback",
    "openclaw-rl-interactions": "online-learning-from-feedback",
    "openclaw-online-rl-feedback": "online-learning-from-feedback",
    "a2a-protocol-adapter": "a2a-agent-card",
    "a2a-peer-delegation": "a2a-agent-card",
    "a2a-protocol-integration": "a2a-agent-card",
    "p2p-agent-discovery": "a2a-agent-card",
    "a2a-task-delegation": "a2a-agent-card",
    "a2a-agent-cards-discovery": "a2a-agent-card",
    "agent-teams-isolation": "sub-agent-spawning",
    "agent-teams-subagents": "sub-agent-spawning",
    "agent-teams": "sub-agent-spawning",
    "agent-teams-isolation-v2": "sub-agent-spawning",
    "agent-teams-isolation-v3": "sub-agent-spawning",
    "agent-teams-git-worktree": "sub-agent-spawning",
    "multi-agent-workflows-f4f56e7c": "sub-agent-spawning",
    "multi-agent-workflows-git-isolation": "sub-agent-spawning",
    "sub-agents-rpc": "sub-agent-spawning",
    "mcp-action-tool-subagent": "sub-agent-spawning",
    "mcp-action-tool-subagent-team": "sub-agent-spawning",
    "longitudinal-quality-metrics": "quality-metrics-tracking",
    "hermes-cron-scheduler-expansion": "cron-scheduler",
    "skill-creation-from-experience-hermes": "skill-creation-from-experience",
    "task-dependency-graph": "dependency-graph-tasks",
    "planner-task-dependency-graphs": "dependency-graph-tasks",
    "runtime-function-tool-concurrency": "runtime-tool-concurrency-eca483c3",
}

REPLENISHMENT_POLICY_UPDATE: dict[str, Any] = {
    "minimum_todo_tasks": 3,
    "batch_size": 3,
    "always_add_tasks": False,
    "max_todo_tasks": 20,
    "tasks_per_run": 1,
    "allowed_risks_for_generated_tasks": ["low", "medium"],
    "instruction": (
        "Add new todo tasks ONLY when the pool falls below minimum_todo_tasks. "
        "Before adding, scan existing done and blocked tasks — never recreate implemented features. "
        "Prefer stabilization, wiring, and test coverage over new trend modules. "
        "Read docs/trends.md for inspiration when replenishment is needed."
    ),
}


def dedupe_manifest(data: dict[str, Any]) -> tuple[int, int]:
    """Block duplicate todos. Returns (blocked_count, remaining_todo_count)."""
    tasks = data.get("tasks", [])
    blocked = 0

    for task in tasks:
        task_id = task.get("id")
        if task.get("status") != "todo":
            continue
        superseded_by = DUPLICATE_OF_DONE.get(task_id)
        if not superseded_by:
            continue
        task["status"] = "blocked"
        suffix = f" [blocked: duplicate of done task '{superseded_by}']"
        if suffix not in task.get("description", ""):
            task["description"] = task["description"].rstrip() + suffix
        blocked += 1

    data["replenishment_policy"] = {
        **(data.get("replenishment_policy") or {}),
        **REPLENISHMENT_POLICY_UPDATE,
    }

    todo_count = sum(1 for task in tasks if task.get("status") == "todo")
    return blocked, todo_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default="agent_tasks.json")
    args = parser.parse_args(argv)

    path = Path(args.manifest)
    with path.open("r", encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)

    blocked, todo_count = dedupe_manifest(data)

    with path.open("w", encoding="utf-8") as manifest_file:
        json.dump(data, manifest_file, indent=2, ensure_ascii=False)
        manifest_file.write("\n")

    print(f"blocked {blocked} duplicate todo tasks")
    print(f"remaining todo tasks: {todo_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
