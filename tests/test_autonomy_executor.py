import json
import os

import pytest

from magda_agent.autonomy.executor import AutonomousExecutor
from magda_agent.autonomy.task_store import TaskStore, TaskStatus


class FakePlanner:
    """Minimal stand-in for magda_agent.planning.planner.Planner."""

    def __init__(self, plans):
        # plans: list of step-lists, one per generate_plan call
        self._plans = list(plans)
        self._current = []
        self.completed_steps = []

    async def generate_plan(self, objective, user_id=None):
        steps = self._plans.pop(0) if self._plans else []
        self._current = [dict(s) for s in steps]
        for i, step in enumerate(self._current):
            if "id" not in step:
                step["id"] = f"step_{i}"
            if "dependencies" not in step:
                step["dependencies"] = []
        self.completed_steps = []
        return self._current

    def get_current_plan(self, user_id=None):
        return self._current

    def get_executable_steps(self, user_id=None):
        completed_ids = {s["id"] for s in self.completed_steps}
        executable = []
        for step in self._current:
            if all(dep in completed_ids for dep in step.get("dependencies", [])):
                executable.append(step)
        return executable

    def get_completed_steps(self, user_id=None):
        return self.completed_steps

    def mark_step_id_completed(self, step_id, result, user_id=None):
        for i, step in enumerate(self._current):
            if step["id"] == step_id:
                step = self._current.pop(i)
                step["result"] = result
                self.completed_steps.append(step)
                return

    def clear_pending_plan(self, user_id=None):
        self._current = []


class FakeSkills:
    def __init__(self):
        self.calls = []

    def execute_skill(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return f"ran {name}"


class FakeLLM:
    """Returns queued evaluation responses (JSON strings)."""

    def __init__(self, responses):
        self._responses = list(responses)

    async def chat_completion(self, messages, temperature=0.7):
        return self._responses.pop(0) if self._responses else json.dumps({"done": True, "summary": "fallback"})


@pytest.fixture
def store_path(tmp_path):
    return os.path.join(str(tmp_path), "tasks.json")


@pytest.mark.asyncio
async def test_task_completes_when_evaluator_says_done(store_path):
    store = TaskStore(path=store_path)
    skills = FakeSkills()
    llm = FakeLLM([json.dumps({"done": True, "summary": "all good", "next_objective": ""})])
    planner = FakePlanner([[{"id": "s1", "description": "search", "skill": "internet_search", "skill_kwargs": {"query": "x"}}]])

    executor = AutonomousExecutor(store, llm, skills, planner_factory=lambda: planner)
    task = await store.add_task("find the answer", user_id=7)
    await store.claim_next_queued()

    await executor._execute_task(task.id)

    result = await store.get(task.id)
    assert result.status == TaskStatus.COMPLETED
    assert result.result == "all good"
    assert skills.calls == [("internet_search", {"query": "x"})]
    events = [p["event"] for p in result.progress]
    assert "started" in events and "step" in events and "completed" in events


@pytest.mark.asyncio
async def test_task_iterates_until_done(store_path):
    store = TaskStore(path=store_path)
    skills = FakeSkills()
    # First eval: not done, refine objective. Second eval: done.
    llm = FakeLLM([
        json.dumps({"done": False, "summary": "step 1 done", "next_objective": "phase 2"}),
        json.dumps({"done": True, "summary": "finished", "next_objective": ""}),
    ])
    planner = FakePlanner([
        [{"id": "s1", "description": "phase1", "skill": "internet_search", "skill_kwargs": {}}],
        [{"id": "s2", "description": "phase2", "skill": "internet_search", "skill_kwargs": {}}],
    ])

    executor = AutonomousExecutor(store, llm, skills, planner_factory=lambda: planner)
    task = await store.add_task("multi-step goal")
    await store.claim_next_queued()

    await executor._execute_task(task.id)

    result = await store.get(task.id)
    assert result.status == TaskStatus.COMPLETED
    assert result.iterations == 2
    assert len(skills.calls) == 2


@pytest.mark.asyncio
async def test_max_iterations_budget_stops_task(store_path):
    store = TaskStore(path=store_path)
    skills = FakeSkills()
    # Evaluator never says done.
    llm = FakeLLM([json.dumps({"done": False, "summary": "still going", "next_objective": "again"})] * 10)
    planner = FakePlanner([[{"id": "s", "description": "loop", "skill": "internet_search", "skill_kwargs": {}}]] * 10)

    executor = AutonomousExecutor(store, llm, skills, planner_factory=lambda: planner)
    task = await store.add_task("never ending", max_iterations=3)
    await store.claim_next_queued()

    await executor._execute_task(task.id)

    result = await store.get(task.id)
    assert result.status == TaskStatus.COMPLETED
    assert result.iterations == 3


@pytest.mark.asyncio
async def test_cancellation_stops_before_completion(store_path):
    store = TaskStore(path=store_path)
    skills = FakeSkills()
    llm = FakeLLM([json.dumps({"done": False, "summary": "wip", "next_objective": "more"})] * 5)
    planner = FakePlanner([[{"id": "s", "description": "loop", "skill": "internet_search", "skill_kwargs": {}}]] * 5)

    executor = AutonomousExecutor(store, llm, skills, planner_factory=lambda: planner)
    task = await store.add_task("cancel me", max_iterations=5)
    await store.claim_next_queued()
    await store.request_cancel(task.id)  # cancel before execution starts iterating

    await executor._execute_task(task.id)

    result = await store.get(task.id)
    assert result.status == TaskStatus.CANCELLED
    assert skills.calls == []


def test_parse_decision_handles_markdown_fences():
    raw = "```json\n{\"done\": true, \"summary\": \"ok\", \"next_objective\": \"\"}\n```"
    decision = AutonomousExecutor._parse_decision(raw, "obj")
    assert decision["done"] is True
    assert decision["summary"] == "ok"


def test_parse_decision_non_json_is_conservative():
    decision = AutonomousExecutor._parse_decision("I think we are not done yet", "current-obj")
    assert decision["done"] is False
    assert decision["next_objective"] == "current-obj"
