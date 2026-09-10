import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.architecture.dependency_router_v4 import (
    MagenticOneDependencyRouterV4,
    DependencyRouterV4,
)


class MockSubagentExecutor:
    """Mock agent executor that records task execution and simulates delay."""

    def __init__(self, step_id: str, return_data: dict, delay: float = 0.01, fail: bool = False):
        self.step_id = step_id
        self.return_data = return_data
        self.delay = delay
        self.fail = fail

    async def execute(self, context: list, **kwargs) -> dict:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"Step '{self.step_id}' failed execution")
        return self.return_data


@pytest.fixture
def mock_spawner():
    spawner = MagicMock()

    async def mock_spawn_subagent(task_description, full_context, agent_executor, **kwargs):
        if hasattr(agent_executor, "execute"):
            return await agent_executor.execute(full_context, **kwargs)
        elif callable(agent_executor):
            return await agent_executor(full_context, **kwargs)
        return {"status": "default"}

    spawner.spawn_subagent = AsyncMock(side_effect=mock_spawn_subagent)
    return spawner


def test_topological_sort_and_layers():
    router = MagenticOneDependencyRouterV4()

    plan_steps = [
        {"id": "task3", "task": "Task 3", "dependencies": ["task1", "task2"]},
        {"id": "task1", "task": "Task 1", "dependencies": []},
        {"id": "task2", "task": "Task 2", "dependencies": None},  # Test None dependencies
        {"id": "task4", "task": "Task 4", "dependencies": ["task3"]},
    ]

    layers = router.resolve_execution_layers(plan_steps)
    assert len(layers) == 3

    # Layer 0: task1 and task2 (independent)
    layer0_ids = {s["id"] for s in layers[0]}
    assert layer0_ids == {"task1", "task2"}

    # Layer 1: task3
    layer1_ids = {s["id"] for s in layers[1]}
    assert layer1_ids == {"task3"}

    # Layer 2: task4
    layer2_ids = {s["id"] for s in layers[2]}
    assert layer2_ids == {"task4"}

    sorted_steps = router.topological_sort(plan_steps)
    sorted_ids = [s["id"] for s in sorted_steps]

    assert sorted_ids.index("task1") < sorted_ids.index("task3")
    assert sorted_ids.index("task2") < sorted_ids.index("task3")
    assert sorted_ids.index("task3") < sorted_ids.index("task4")


def test_cycle_detection_raises_value_error():
    router = MagenticOneDependencyRouterV4()

    plan_with_cycle = [
        {"id": "taskA", "task": "Task A", "dependencies": ["taskB"]},
        {"id": "taskB", "task": "Task B", "dependencies": ["taskA"]},
    ]

    with pytest.raises(ValueError, match="Cycle detected"):
        router.resolve_execution_layers(plan_with_cycle)


def test_missing_dependency_reference_raises_value_error():
    router = MagenticOneDependencyRouterV4()

    plan_missing_dep = [
        {"id": "taskA", "task": "Task A", "dependencies": ["non_existent_step"]},
    ]

    with pytest.raises(ValueError, match="Dependency 'non_existent_step' for step 'taskA' not found"):
        router.resolve_execution_layers(plan_missing_dep)


def test_missing_step_id_raises_value_error():
    router = MagenticOneDependencyRouterV4()

    invalid_plan = [
        {"task": "Task without ID", "dependencies": []}
    ]

    with pytest.raises(ValueError, match="Each step must specify an 'id' or 'task_id'"):
        router.resolve_execution_layers(invalid_plan)


@pytest.mark.asyncio
async def test_route_dag_plan_execution(mock_spawner):
    router = MagenticOneDependencyRouterV4(spawner=mock_spawner)

    plan_steps = [
        {"id": "step1", "task": "Data Prep A", "dependencies": []},
        {"id": "step2", "task": "Data Prep B", "dependencies": None},
        {"id": "step3", "task": "Combine Data", "dependencies": ["step1", "step2"]},
    ]

    executors = {
        "step1": MockSubagentExecutor("step1", {"step1_result": "data_a"}),
        "step2": MockSubagentExecutor("step2", {"step2_result": "data_b"}),
        "step3": MockSubagentExecutor("step3", {"step3_result": "combined"}),
    }

    factory_idx = 0
    def custom_factory():
        nonlocal factory_idx
        items = list(executors.values())
        executor = items[factory_idx]
        factory_idx += 1
        return executor

    res = await router.route_dag_plan(
        plan_steps=plan_steps,
        base_context=[{"role": "system", "content": "Init"}],
        agent_executor_factory=custom_factory,
        merge_results=False,
    )

    assert res["layers_executed"] == 2
    assert "step1" in res["task_outputs"]
    assert "step2" in res["task_outputs"]
    assert "step3" in res["task_outputs"]

    assert res["task_outputs"]["step1"] == {"step1_result": "data_a"}
    assert res["task_outputs"]["step2"] == {"step2_result": "data_b"}
    assert res["task_outputs"]["step3"] == {"step3_result": "combined"}

    assert res["merged_state"] == {
        "step1_result": "data_a",
        "step2_result": "data_b",
        "step3_result": "combined",
    }


@pytest.mark.asyncio
async def test_route_dag_plan_failure_handling(mock_spawner):
    router = MagenticOneDependencyRouterV4(spawner=mock_spawner)

    plan_steps = [
        {"id": "p1", "task": "Prereq Fails", "dependencies": []},
        {"id": "p2", "task": "Independent Succeeds", "dependencies": []},
        {"id": "p3", "task": "Dependent on p1", "dependencies": ["p1"]},
    ]

    call_count = 0
    def factory():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockSubagentExecutor("p1", {}, fail=True)
        else:
            return MockSubagentExecutor("p2", {"p2_data": "ok"})

    res = await router.route_dag_plan(
        plan_steps=plan_steps,
        agent_executor_factory=factory,
    )

    assert "p1" in res["failed_tasks"]
    assert "p3" in res["failed_tasks"]
    assert res["task_outputs"]["p2"] == {"p2_data": "ok"}
    assert isinstance(res["task_outputs"]["p1"], Exception)
    assert isinstance(res["task_outputs"]["p3"], Exception)


def test_alias_compatibility():
    assert DependencyRouterV4 is MagenticOneDependencyRouterV4
