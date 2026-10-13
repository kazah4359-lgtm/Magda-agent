"""
Unit tests for HierarchicalPlannerV3 in magda_agent/architecture/hierarchical_planner_v3.py.
"""

import asyncio
import pytest
from unittest.mock import MagicMock

from magda_agent.architecture.dependency_graph_v2 import SubagentTask, TaskStatus
from magda_agent.architecture.hierarchical_planner_v3 import HierarchicalPlannerV3


def test_default_decomposition():
    """Verifies default 3-stage heuristic decomposition."""
    planner = HierarchicalPlannerV3()
    graph = planner.decompose_request("Build standard web crawler")

    tasks = graph.get_all_tasks()
    assert len(tasks) == 3

    task_map = {t.task_id: t for t in tasks}
    assert "task_analysis" in task_map
    assert "task_execution" in task_map
    assert "task_verification" in task_map

    assert task_map["task_execution"].dependencies == {"task_analysis"}
    assert task_map["task_verification"].dependencies == {"task_execution"}

    is_valid, errors = planner.validate_graph()
    assert is_valid
    assert len(errors) == 0


def test_custom_plan_decomposition():
    """Verifies decomposition from a list of dicts."""
    custom_plan = [
        {"id": "step_a", "description": "Fetch data", "dependencies": []},
        {"id": "step_b", "description": "Process data", "dependencies": ["step_a"]},
        {"id": "step_c", "description": "Store data", "dependencies": ["step_a"]},
        {"id": "step_d", "description": "Report", "dependencies": ["step_b", "step_c"]},
    ]

    planner = HierarchicalPlannerV3()
    planner.decompose_request("Custom Data Pipeline", custom_plan=custom_plan)

    layers = planner.get_execution_layers()
    assert len(layers) == 3
    assert [t.task_id for t in layers[0]] == ["step_a"]
    assert set(t.task_id for t in layers[1]) == {"step_b", "step_c"}
    assert [t.task_id for t in layers[2]] == ["step_d"]


def test_graph_validation_detects_cycles():
    """Verifies that cycles are caught by validate_graph."""
    plan = [
        {"id": "step_1", "description": "One", "dependencies": ["step_2"]},
        {"id": "step_2", "description": "Two", "dependencies": ["step_1"]},
    ]
    planner = HierarchicalPlannerV3()
    planner.decompose_request("Cyclic Request", custom_plan=plan)

    is_valid, errors = planner.validate_graph()
    assert not is_valid
    assert any("Cycle" in err for err in errors)


def test_expand_subtask():
    """Verifies expanding a parent task into child sub-tasks."""
    plan = [
        {"id": "parent", "description": "Complex Parent", "dependencies": []},
        {"id": "downstream", "description": "Final Task", "dependencies": ["parent"]},
    ]
    planner = HierarchicalPlannerV3()
    planner.decompose_request("Hierarchical Job", custom_plan=plan)

    subtasks = [
        {"id": "sub_1", "description": "Child Part 1", "dependencies": []},
        {"id": "sub_2", "description": "Child Part 2", "dependencies": ["sub_1"]},
    ]

    added = planner.expand_subtask("parent", subtasks)
    assert len(added) == 2
    assert not planner.graph.has_task("parent")

    downstream_task = planner.graph.get_task("downstream")
    assert downstream_task is not None
    # downstream task should now depend on leaf subtask "sub_2"
    assert "sub_2" in downstream_task.dependencies

    layers = planner.get_execution_layers()
    layer_ids = [[t.task_id for t in layer] for layer in layers]
    assert layer_ids == [["sub_1"], ["sub_2"], ["downstream"]]


def test_execute_plan_async_mock_dispatch():
    """Verifies async execution layer ordering without executing real sub-agents."""
    plan = [
        {"id": "t1", "description": "Init DB", "dependencies": []},
        {"id": "t2", "description": "Seed Table A", "dependencies": ["t1"]},
        {"id": "t3", "description": "Seed Table B", "dependencies": ["t1"]},
        {"id": "t4", "description": "Validate DB", "dependencies": ["t2", "t3"]},
    ]

    planner = HierarchicalPlannerV3()
    planner.decompose_request("DB Setup", custom_plan=plan)

    dispatched_order = []

    async def mock_dispatch(task: SubagentTask, ctx: dict):
        dispatched_order.append(task.task_id)
        return f"Done {task.task_id}"

    async def run_test():
        return await planner.execute_plan_async(dispatch_fn=mock_dispatch)

    result = asyncio.run(run_test())

    assert result["status"] == "completed"
    assert set(result["completed_tasks"]) == {"t1", "t2", "t3", "t4"}
    assert dispatched_order[0] == "t1"
    assert set(dispatched_order[1:3]) == {"t2", "t3"}
    assert dispatched_order[3] == "t4"


def test_execute_plan_async_failure_propagation():
    """Verifies failure of a task blocks dependent sub-tasks."""
    plan = [
        {"id": "a", "description": "Step A", "dependencies": []},
        {"id": "b", "description": "Step B", "dependencies": ["a"]},
    ]

    planner = HierarchicalPlannerV3()
    planner.decompose_request("Failing Job", custom_plan=plan)

    async def mock_failing_dispatch(task: SubagentTask, ctx: dict):
        if task.task_id == "a":
            raise RuntimeError("Step A failed")
        return "Success"

    async def run_test():
        return await planner.execute_plan_async(dispatch_fn=mock_failing_dispatch)

    result = asyncio.run(run_test())

    assert result["status"] == "failed"
    assert "a" in result["failed_tasks"]

    task_b = planner.graph.get_task("b")
    assert task_b.status == TaskStatus.BLOCKED


def test_serialization():
    """Verifies summary and dictionary serialization."""
    planner = HierarchicalPlannerV3()
    planner.decompose_request("Serialization Test")

    summary = planner.get_plan_summary()
    assert summary["total_tasks"] == 3
    assert summary["pending"] == 3

    data = planner.to_dict()
    assert "summary" in data
    assert "layers" in data
    assert len(data["layers"]) == 3
