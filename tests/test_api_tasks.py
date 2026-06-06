"""HTTP-level tests for the autonomous task endpoints.

The background executor is stopped so these tests exercise the queue/REST surface
deterministically without running the (LLM-dependent) execution loop.
"""
import pytest
from fastapi.testclient import TestClient

import magda_agent.api as api_module
from magda_agent.api import app
from magda_agent.autonomy.task_store import TaskStore


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    store = TaskStore(path=str(tmp_path / "tasks.json"))
    monkeypatch.setattr(api_module, "task_store", store)
    # Prevent the real background worker from consuming/advancing tasks.
    monkeypatch.setattr(api_module.autonomous_executor, "start", _noop)
    monkeypatch.setattr(api_module.autonomous_executor, "stop", _noop)
    monkeypatch.setattr(api_module.autonomous_executor, "store", store)


async def _noop(*args, **kwargs):
    return None


client = TestClient(app)


def test_create_and_get_task():
    resp = client.post("/tasks", json={"goal": "summarize the news", "user_id": 5})
    assert resp.status_code == 200
    task = resp.json()["task"]
    assert task["status"] == "queued"
    assert task["goal"] == "summarize the news"
    task_id = task["id"]

    detail = client.get(f"/tasks/{task_id}")
    assert detail.status_code == 200
    assert detail.json()["goal"] == "summarize the news"


def test_list_tasks_filtered_by_user():
    client.post("/tasks", json={"goal": "a", "user_id": 100})
    client.post("/tasks", json={"goal": "b", "user_id": 200})
    resp = client.get("/tasks", params={"user_id": 100})
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    assert all(t["user_id"] == 100 for t in tasks)
    assert len(tasks) == 1


def test_max_iterations_is_clamped():
    resp = client.post("/tasks", json={"goal": "x", "max_iterations": 9999})
    assert resp.json()["task"]["max_iterations"] == 200


def test_cancel_queued_task():
    task_id = client.post("/tasks", json={"goal": "cancel me"}).json()["task"]["id"]
    resp = client.post(f"/tasks/{task_id}/cancel")
    assert resp.status_code == 200
    assert client.get(f"/tasks/{task_id}").json()["status"] == "cancelled"


def test_get_missing_task_returns_404():
    assert client.get("/tasks/doesnotexist").status_code == 404


def test_resume_non_paused_task_conflicts():
    task_id = client.post("/tasks", json={"goal": "x"}).json()["task"]["id"]
    # A queued task cannot be resumed (it isn't paused).
    assert client.post(f"/tasks/{task_id}/resume").status_code == 409
