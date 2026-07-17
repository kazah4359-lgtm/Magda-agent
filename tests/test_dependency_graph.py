import pytest
from magda_agent.planning.dependency_graph import DependencyGraph

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

def test_get_execution_layers():
    plan_steps = [
        {"id": "A", "dependencies": []},
        {"id": "B", "dependencies": ["A"]},
        {"id": "C", "dependencies": ["A"]},
        {"id": "D", "dependencies": ["B", "C"]},
        {"id": "E", "dependencies": ["C"]},
        {"id": "F", "dependencies": ["D", "E"]}
    ]
    layers = DependencyGraph.get_execution_layers(plan_steps)
    # Expected layers:
    # Layer 0: A
    # Layer 1: B, C
    # Layer 2: D, E
    # Layer 3: F
    assert len(layers) == 4
    assert [step["id"] for step in layers[0]] == ["A"]
    assert set(step["id"] for step in layers[1]) == {"B", "C"}
    assert set(step["id"] for step in layers[2]) == {"D", "E"}
    assert [step["id"] for step in layers[3]] == ["F"]

def test_get_execution_layers_cycle():
    plan_steps = [
        {"id": "A", "dependencies": ["B"]},
        {"id": "B", "dependencies": ["A"]}
    ]
    with pytest.raises(ValueError, match="Cycle detected"):
        DependencyGraph.get_execution_layers(plan_steps)

def test_get_critical_path():
    plan_steps = [
        {"id": "A", "dependencies": []},
        {"id": "B", "dependencies": ["A"]},
        {"id": "C", "dependencies": ["A"]},
        {"id": "D", "dependencies": ["B"]},
        {"id": "E", "dependencies": ["D"]},
        {"id": "F", "dependencies": ["C", "E"]}
    ]
    path = DependencyGraph.get_critical_path(plan_steps)
    # The longest path is A -> B -> D -> E -> F (length 5)
    # A -> C -> F is length 3
    assert [step["id"] for step in path] == ["A", "B", "D", "E", "F"]
