import pytest
from unittest.mock import patch

from magda_agent.skills.codex_worker import codex_worker


@pytest.fixture
def mock_manifest():
    return {
        "tasks": [
            {
                "id": "task-1",
                "status": "done",
                "title": "Task 1",
                "area": "core",
                "risk": "low",
                "description": "Done task",
                "allowed_paths": [],
                "acceptance": []
            },
            {
                "id": "task-2",
                "status": "todo",
                "title": "Task 2",
                "area": "api",
                "risk": "low",
                "description": "Todo task 2",
                "allowed_paths": ["path/to/file"],
                "acceptance": ["check 1"]
            },
            {
                "id": "task-3",
                "status": "todo",
                "title": "Task 3",
                "area": "cli",
                "risk": "medium",
                "description": "Todo task 3",
                "allowed_paths": [],
                "acceptance": []
            }
        ]
    }


@patch("magda_agent.skills.codex_worker.load_manifest")
def test_codex_worker_no_task_id(mock_load_manifest, mock_manifest):
    mock_load_manifest.return_value = mock_manifest

    prompt = codex_worker()

    # Should get task-2 since it's the first todo task
    assert "Task id: task-2" in prompt
    assert "Title: Task 2" in prompt
    assert "- path/to/file" in prompt


@patch("magda_agent.skills.codex_worker.load_manifest")
def test_codex_worker_with_task_id(mock_load_manifest, mock_manifest):
    mock_load_manifest.return_value = mock_manifest

    prompt = codex_worker("task-3")

    assert "Task id: task-3" in prompt
    assert "Title: Task 3" in prompt
    assert "Area: cli" in prompt


@patch("magda_agent.skills.codex_worker.load_manifest")
def test_codex_worker_task_not_found(mock_load_manifest, mock_manifest):
    mock_load_manifest.return_value = mock_manifest

    prompt = codex_worker("nonexistent-task")

    assert prompt == "Task with ID 'nonexistent-task' not found."


@patch("magda_agent.skills.codex_worker.load_manifest")
def test_codex_worker_no_todo_tasks(mock_load_manifest):
    mock_load_manifest.return_value = {
        "tasks": [
            {
                "id": "task-1",
                "status": "done"
            }
        ]
    }

    prompt = codex_worker()

    assert prompt == "No 'todo' tasks found in the manifest."


@patch("magda_agent.skills.codex_worker.load_manifest")
def test_codex_worker_manifest_load_error(mock_load_manifest):
    mock_load_manifest.side_effect = Exception("File not found")

    prompt = codex_worker()

    assert "Error loading task manifest: File not found" in prompt
