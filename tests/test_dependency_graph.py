import pytest
from magda_agent.architecture.dependency_graph import DependencyGraph

def test_topological_sort_empty():
    """Test that topological sort works on an empty list of tasks."""
    plan_steps = []
    sorted_steps = DependencyGraph.topological_sort(plan_steps)
    assert sorted_steps == []

def test_get_executable_steps():
    plan_steps = [
        {"id": "step1", "dependencies": []},
        {"id": "step2", "dependencies": ["step1"]},
        {"id": "step3", "dependencies": ["step1", "step2"]}
    ]

    # Initially only step1 is executable
    exec_steps = DependencyGraph.get_executable_steps(plan_steps, set())
    assert len(exec_steps) == 1
    assert exec_steps[0]["id"] == "step1"

    # After step1, step2 is executable
    exec_steps = DependencyGraph.get_executable_steps(plan_steps, {"step1"})
    assert len(exec_steps) == 1
    assert exec_steps[0]["id"] == "step2"

    # After step1 and step2, step3 is executable
    exec_steps = DependencyGraph.get_executable_steps(plan_steps, {"step1", "step2"})
    assert len(exec_steps) == 1
    assert exec_steps[0]["id"] == "step3"

    # After all are completed, no executable steps remain
    exec_steps = DependencyGraph.get_executable_steps(plan_steps, {"step1", "step2", "step3"})
    assert len(exec_steps) == 0

def test_topological_sort_success():
    plan_steps = [
        {"id": "step3", "dependencies": ["step2"]},
        {"id": "step1", "dependencies": []},
        {"id": "step2", "dependencies": ["step1"]}
    ]

    sorted_steps = DependencyGraph.topological_sort(plan_steps)
    sorted_ids = [step["id"] for step in sorted_steps]

    assert sorted_ids == ["step1", "step2", "step3"]

def test_topological_sort_cycle():
    plan_steps = [
        {"id": "step1", "dependencies": ["step3"]},
        {"id": "step2", "dependencies": ["step1"]},
        {"id": "step3", "dependencies": ["step2"]}
    ]

    with pytest.raises(ValueError, match="Cycle detected in plan dependencies"):
        DependencyGraph.topological_sort(plan_steps)

def test_topological_sort_missing_dependency():
    """
    Test that an unknown dependency is ignored but logged.
    """
    plan_steps = [
        {"id": "step1", "dependencies": ["unknown_step"]}
    ]
    sorted_steps = DependencyGraph.topological_sort(plan_steps)
    assert len(sorted_steps) == 1
    assert sorted_steps[0]["id"] == "step1"
