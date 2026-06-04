"""Tests for the Codex/Jules CLI bridge."""

import json
from pathlib import Path
from typing import Any

import pytest
from magda_agent.codex_bridge import (
    iter_tasks,
    load_manifest,
    main,
    next_task,
    queue_status,
    render_prompt,
    render_review_prompt,
    task_by_id,
    todo_tasks,
    validate_manifest,
)


@pytest.fixture
def mock_manifest_data() -> dict[str, Any]:
    """Provide a mock manifest payload."""
    return {
        "tasks": [
            {
                "id": "task-done",
                "status": "done",
                "title": "A finished task",
            },
            {
                "id": "task-1",
                "status": "todo",
                "title": "First todo task",
                "area": "testing",
                "risk": "low",
                "description": "Do the first thing.",
                "allowed_paths": ["file1.py", "tests/test_file1.py"],
                "acceptance": ["criterion A", "criterion B"],
            },
            {
                "id": "task-2",
                "status": "todo",
                "title": "Second todo task",
            },
        ],
        "replenishment_policy": {
            "minimum_todo_tasks": 3,
        },
    }


@pytest.fixture
def mock_manifest_file(tmp_path: Path, mock_manifest_data: dict[str, Any]) -> Path:
    """Provide a temporary manifest file."""
    manifest_path = tmp_path / "agent_tasks.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(mock_manifest_data, f)
    return manifest_path


def test_load_manifest(mock_manifest_file: Path, mock_manifest_data: dict[str, Any]) -> None:
    """Test loading the manifest."""
    data = load_manifest(mock_manifest_file)
    assert data == mock_manifest_data


def test_load_manifest_invalid(tmp_path: Path) -> None:
    """Test loading an invalid manifest."""
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("[]")
    with pytest.raises(ValueError, match="manifest root must be an object"):
        load_manifest(invalid_path)


def test_iter_tasks(mock_manifest_data: dict[str, Any]) -> None:
    """Test iterating over tasks."""
    tasks = iter_tasks(mock_manifest_data)
    assert len(tasks) == 3
    assert tasks[0]["id"] == "task-done"


def test_iter_tasks_invalid() -> None:
    """Test iterating over missing tasks."""
    with pytest.raises(ValueError, match="manifest tasks must be a list"):
        iter_tasks({})


def test_todo_tasks(mock_manifest_data: dict[str, Any]) -> None:
    """Test filtering todo tasks."""
    todos = todo_tasks(mock_manifest_data)
    assert len(todos) == 2
    assert todos[0]["id"] == "task-1"
    assert todos[1]["id"] == "task-2"


def test_next_task(mock_manifest_data: dict[str, Any]) -> None:
    """Test retrieving the next todo task."""
    task = next_task(mock_manifest_data)
    assert task is not None
    assert task["id"] == "task-1"


def test_next_task_empty() -> None:
    """Test retrieving the next task when none are todo."""
    assert next_task({"tasks": [{"id": "task-done", "status": "done"}]}) is None


def test_task_by_id(mock_manifest_data: dict[str, Any]) -> None:
    """Test finding a task by ID."""
    task = task_by_id(mock_manifest_data, "task-2")
    assert task is not None
    assert task["id"] == "task-2"

    assert task_by_id(mock_manifest_data, "non-existent") is None


def test_queue_status(mock_manifest_data: dict[str, Any]) -> None:
    """Test status reporting including total count and replenishment flag."""
    status = queue_status(mock_manifest_data)
    assert status["total_tasks"] == 3
    assert status["todo_tasks"] == 2
    assert status["minimum_todo_tasks"] == 3
    assert status["needs_replenishment"] is True
    assert status["next_task_id"] == "task-1"


def test_render_prompt(mock_manifest_data: dict[str, Any]) -> None:
    """Test rendering the Codex prompt."""
    task = next_task(mock_manifest_data)
    assert task is not None
    prompt = render_prompt(task)
    assert "Task id: task-1" in prompt
    assert "Title: First todo task" in prompt
    assert "Area: testing" in prompt
    assert "Risk: low" in prompt
    assert "- file1.py" in prompt
    assert "- tests/test_file1.py" in prompt
    assert "- criterion A" in prompt
    assert "- criterion B" in prompt


def test_render_review_prompt() -> None:
    """Test rendering the PR review prompt."""
    title = "Add new feature"
    changed_files = ["file_a.py", "file_b.py"]
    diff_summary = "+ def new_func(): pass"
    prompt = render_review_prompt(title, changed_files, diff_summary)

    assert "PR Title: Add new feature" in prompt
    assert "- file_a.py" in prompt
    assert "- file_b.py" in prompt
    assert "+ def new_func(): pass" in prompt
    assert "Findings (ordered by severity):" in prompt
    assert "Risk:" in prompt
    assert "Missing Tests:" in prompt


def test_validate_manifest(mock_manifest_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validating the manifest delegates properly."""
    called = False

    def mock_validate(data: dict[str, Any]) -> list[str]:
        nonlocal called
        called = True
        return ["a warning"]

    # The function imports `load_manifest` and `validate_manifest` from scripts.
    import sys
    from types import ModuleType

    # Create a mock module structure
    mock_scripts = ModuleType("scripts")
    mock_validate_module = ModuleType("scripts.validate_agent_tasks")
    mock_validate_module.load_manifest = lambda p: {}
    mock_validate_module.validate_manifest = mock_validate
    mock_scripts.validate_agent_tasks = mock_validate_module

    monkeypatch.setitem(sys.modules, "scripts", mock_scripts)
    monkeypatch.setitem(sys.modules, "scripts.validate_agent_tasks", mock_validate_module)

    code = validate_manifest(mock_manifest_file)
    assert called is True
    assert code == 0


def test_validate_manifest_error(mock_manifest_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test validating the manifest handles errors."""

    def mock_validate(data: dict[str, Any]) -> list[str]:
        raise ValueError("test error")

    import sys
    from types import ModuleType

    mock_scripts = ModuleType("scripts")
    mock_validate_module = ModuleType("scripts.validate_agent_tasks")
    mock_validate_module.load_manifest = lambda p: {}
    mock_validate_module.validate_manifest = mock_validate
    mock_scripts.validate_agent_tasks = mock_validate_module

    monkeypatch.setitem(sys.modules, "scripts", mock_scripts)
    monkeypatch.setitem(sys.modules, "scripts.validate_agent_tasks", mock_validate_module)

    code = validate_manifest(mock_manifest_file)
    assert code == 1
    out, err = capsys.readouterr()
    assert "validation failed: test error" in err


def test_main_status(mock_manifest_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI status command."""
    code = main(["--manifest", str(mock_manifest_file), "status"])
    assert code == 0
    out, _ = capsys.readouterr()
    parsed = json.loads(out)
    assert parsed["total_tasks"] == 3
    assert parsed["todo_tasks"] == 2


def test_main_next_task(mock_manifest_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI next-task command."""
    code = main(["--manifest", str(mock_manifest_file), "next-task"])
    assert code == 0
    out, _ = capsys.readouterr()
    parsed = json.loads(out)
    assert parsed["id"] == "task-1"


def test_main_render_prompt(mock_manifest_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI render-prompt command."""
    code = main(["--manifest", str(mock_manifest_file), "render-prompt", "--task-id", "task-2"])
    assert code == 0
    out, _ = capsys.readouterr()
    assert "Task id: task-2" in out


def test_main_render_prompt_default(mock_manifest_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI render-prompt without task ID falls back to next task."""
    code = main(["--manifest", str(mock_manifest_file), "render-prompt"])
    assert code == 0
    out, _ = capsys.readouterr()
    assert "Task id: task-1" in out


def test_main_review_prompt_cli(mock_manifest_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI review-prompt command."""
    code = main([
        "--manifest", str(mock_manifest_file),
        "review-prompt",
        "--title", "Fix bug",
        "--changed-files", "a.py", "b.py",
        "--diff-summary", "summary of diff"
    ])
    assert code == 0
    out, _ = capsys.readouterr()
    assert "PR Title: Fix bug" in out
    assert "- a.py" in out
    assert "- b.py" in out
    assert "summary of diff" in out
    assert "Findings (ordered by severity):" in out


def test_main_load_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI handles load errors."""
    missing = tmp_path / "missing.json"
    code = main(["--manifest", str(missing), "status"])
    assert code == 1
    _, err = capsys.readouterr()
    assert "failed to load manifest:" in err
