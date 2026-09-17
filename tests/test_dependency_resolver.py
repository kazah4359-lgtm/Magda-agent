import pytest
from magda_agent.architecture.dependency_resolver import (
    TaskDependencyResolver,
    DependencyResolver,
)


def test_resolve_execution_layers_empty():
    layers = TaskDependencyResolver.resolve_execution_layers([])
    assert layers == []


def test_resolve_execution_layers_linear():
    plan = [
        {"id": "step1", "dependencies": []},
        {"id": "step2", "dependencies": ["step1"]},
        {"id": "step3", "dependencies": ["step2"]},
    ]
    layers = TaskDependencyResolver.resolve_execution_layers(plan)
    assert len(layers) == 3
    assert [s["id"] for s in layers[0]] == ["step1"]
    assert [s["id"] for s in layers[1]] == ["step2"]
    assert [s["id"] for s in layers[2]] == ["step3"]


def test_resolve_execution_layers_diamond_dag():
    plan = [
        {"id": "stepA", "dependencies": []},
        {"id": "stepB", "dependencies": ["stepA"]},
        {"id": "stepC", "dependencies": ["stepA"]},
        {"id": "stepD", "dependencies": ["stepB", "stepC"]},
    ]
    layers = TaskDependencyResolver.resolve_execution_layers(plan)
    assert len(layers) == 3
    assert [s["id"] for s in layers[0]] == ["stepA"]
    layer1_ids = {s["id"] for s in layers[1]}
    assert layer1_ids == {"stepB", "stepC"}
    assert [s["id"] for s in layers[2]] == ["stepD"]


def test_resolve_execution_layers_task_id_key():
    plan = [
        {"task_id": "task1", "dependencies": []},
        {"task_id": "task2", "dependencies": ["task1"]},
    ]
    layers = DependencyResolver.resolve_execution_layers(plan)
    assert len(layers) == 2
    assert layers[0][0]["task_id"] == "task1"
    assert layers[1][0]["task_id"] == "task2"


def test_resolve_execution_layers_cycle_raises():
    plan = [
        {"id": "stepA", "dependencies": ["stepB"]},
        {"id": "stepB", "dependencies": ["stepA"]},
    ]
    with pytest.raises(ValueError, match="Cycle detected"):
        TaskDependencyResolver.resolve_execution_layers(plan)


def test_resolve_execution_layers_missing_dependency():
    plan = [
        {"id": "step1", "dependencies": ["non_existent_step"]},
    ]
    with pytest.raises(ValueError, match="not found in plan steps"):
        TaskDependencyResolver.resolve_execution_layers(plan)


def test_topological_sort():
    plan = [
        {"id": "step3", "dependencies": ["step2"]},
        {"id": "step1", "dependencies": []},
        {"id": "step2", "dependencies": ["step1"]},
    ]
    sorted_steps = TaskDependencyResolver.topological_sort(plan)
    sorted_ids = [s["id"] for s in sorted_steps]
    assert sorted_ids == ["step1", "step2", "step3"]


def test_get_executable_steps():
    plan = [
        {"id": "step1", "dependencies": []},
        {"id": "step2", "dependencies": ["step1"]},
        {"id": "step3", "dependencies": ["step1", "step2"]},
    ]

    # Initial state: step1 executable
    exec_0 = TaskDependencyResolver.get_executable_steps(plan, completed_step_ids=set())
    assert [s["id"] for s in exec_0] == ["step1"]

    # After completing step1: step2 executable
    exec_1 = TaskDependencyResolver.get_executable_steps(plan, completed_step_ids={"step1"})
    assert [s["id"] for s in exec_1] == ["step2"]

    # After completing step1 & step2: step3 executable
    exec_2 = TaskDependencyResolver.get_executable_steps(
        plan, completed_step_ids={"step1", "step2"}
    )
    assert [s["id"] for s in exec_2] == ["step3"]
