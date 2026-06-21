import pytest
from magda_agent.core.task_graph import DependencyGraph

def test_dag_planner_topological_sort_linear():
    plan_steps = [
        {"id": "step_2", "description": "Second", "dependencies": ["step_1"]},
        {"id": "step_1", "description": "First", "dependencies": []},
        {"id": "step_3", "description": "Third", "dependencies": ["step_2"]}
    ]

    sorted_steps = DependencyGraph.topological_sort(plan_steps)
    assert len(sorted_steps) == 3
    assert sorted_steps[0]["id"] == "step_1"
    assert sorted_steps[1]["id"] == "step_2"
    assert sorted_steps[2]["id"] == "step_3"

def test_dag_planner_topological_sort_diamond():
    plan_steps = [
        {"id": "D", "dependencies": ["B", "C"]},
        {"id": "B", "dependencies": ["A"]},
        {"id": "C", "dependencies": ["A"]},
        {"id": "A", "dependencies": []}
    ]

    sorted_steps = DependencyGraph.topological_sort(plan_steps)
    ids = [step["id"] for step in sorted_steps]

    # A must be first
    assert ids[0] == "A"
    # D must be last
    assert ids[-1] == "D"
    # B and C must be in the middle
    assert set(ids[1:3]) == {"B", "C"}

def test_dag_planner_cycle_detection():
    plan_steps = [
        {"id": "A", "dependencies": ["C"]},
        {"id": "B", "dependencies": ["A"]},
        {"id": "C", "dependencies": ["B"]}
    ]

    with pytest.raises(ValueError, match="Cycle detected"):
        DependencyGraph.topological_sort(plan_steps)

def test_get_executable_steps():
    plan_steps = [
        {"id": "A", "dependencies": []},
        {"id": "B", "dependencies": ["A"]},
        {"id": "C", "dependencies": ["A"]},
        {"id": "D", "dependencies": ["B", "C"]}
    ]

    # Initially, only A is executable
    exec_steps = DependencyGraph.get_executable_steps(plan_steps, set())
    assert len(exec_steps) == 1
    assert exec_steps[0]["id"] == "A"

    # After A is done, B and C are executable
    exec_steps = DependencyGraph.get_executable_steps(plan_steps, {"A"})
    assert len(exec_steps) == 2
    assert set(s["id"] for s in exec_steps) == {"B", "C"}

    # After A and B are done, only C is executable (D still needs C)
    exec_steps = DependencyGraph.get_executable_steps(plan_steps, {"A", "B"})
    assert len(exec_steps) == 1
    assert exec_steps[0]["id"] == "C"

    # After A, B, and C are done, D is executable
    exec_steps = DependencyGraph.get_executable_steps(plan_steps, {"A", "B", "C"})
    assert len(exec_steps) == 1
    assert exec_steps[0]["id"] == "D"

    # After all are done, nothing is executable
    exec_steps = DependencyGraph.get_executable_steps(plan_steps, {"A", "B", "C", "D"})
    assert len(exec_steps) == 0
