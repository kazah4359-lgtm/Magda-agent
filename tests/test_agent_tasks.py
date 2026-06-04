import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_agent_tasks import ValidationError, load_manifest, validate_manifest


def test_repository_task_manifest_is_valid():
    manifest = load_manifest(Path("agent_tasks.json"))

    warnings = validate_manifest(manifest)

    assert warnings == []


def test_rejects_duplicate_task_ids():
    manifest = load_manifest(Path("agent_tasks.json"))
    duplicate = copy.deepcopy(manifest["tasks"][0])
    manifest["tasks"].append(duplicate)

    with pytest.raises(ValidationError, match="duplicate task id"):
        validate_manifest(manifest)


def test_rejects_unknown_risk():
    manifest = load_manifest(Path("agent_tasks.json"))
    manifest["tasks"][0]["risk"] = "reckless"

    with pytest.raises(ValidationError, match="unknown risk"):
        validate_manifest(manifest)


def test_rejects_empty_acceptance():
    manifest = load_manifest(Path("agent_tasks.json"))
    manifest["tasks"][0]["acceptance"] = []

    with pytest.raises(ValidationError, match="acceptance"):
        validate_manifest(manifest)


def test_warns_when_no_todo_tasks_remain():
    manifest = load_manifest(Path("agent_tasks.json"))
    for task in manifest["tasks"]:
        task["status"] = "done"

    warnings = validate_manifest(manifest)

    assert "no todo tasks remain" in warnings


def test_warns_when_todo_pool_is_low():
    manifest = load_manifest(Path("agent_tasks.json"))
    manifest["replenishment_policy"]["minimum_todo_tasks"] = 3
    todo_seen = 0
    for task in manifest["tasks"]:
        if task["status"] == "todo":
            todo_seen += 1
            if todo_seen > 2:
                task["status"] = "done"

    warnings = validate_manifest(manifest)

    assert "todo task pool is low: 2 remaining, minimum is 3" in warnings


def test_rejects_high_risk_replenishment_tasks():
    manifest = load_manifest(Path("agent_tasks.json"))
    manifest["replenishment_policy"]["allowed_risks_for_generated_tasks"] = ["low", "high"]

    with pytest.raises(ValidationError, match="low or medium risk"):
        validate_manifest(manifest)


def test_rejects_unsafe_allowed_path():
    manifest = load_manifest(Path("agent_tasks.json"))
    manifest["tasks"][0]["allowed_paths"] = ["../outside"]

    with pytest.raises(ValidationError, match="unsafe allowed path"):
        validate_manifest(manifest)


def test_manifest_file_stays_pretty_printed():
    manifest_path = Path("agent_tasks.json")
    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"

    assert manifest_path.read_text(encoding="utf-8") == expected


def test_rejects_missing_claim_fields():
    manifest = load_manifest(Path("agent_tasks.json"))
    manifest["tasks"][0]["claimed_by"] = "worker-1"
    # missing claimed_at

    with pytest.raises(ValidationError, match="claimed_by and claimed_at must both be present"):
        validate_manifest(manifest)


def test_rejects_invalid_claim_datetime():
    manifest = load_manifest(Path("agent_tasks.json"))
    manifest["tasks"][0]["claimed_by"] = "worker-1"
    manifest["tasks"][0]["claimed_at"] = "not-a-date"

    with pytest.raises(ValidationError, match="valid ISO-8601 datetime string"):
        validate_manifest(manifest)
