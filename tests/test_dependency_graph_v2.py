import pytest
from magda_agent.architecture.dependency_graph_v2 import (
    ClaudeSubagentDependencyGraphV2,
    DependencyGraphV2,
    SubagentTask,
    TaskStatus,
)


def test_subagent_task_creation_and_dict_conversion():
    task = SubagentTask(
        task_id="t1",
        description="Write unit test",
        dependencies={"t0"},
        assigned_subagent="coder",
        required_capability="python",
        input_data={"file": "test.py"},
    )
    assert task.task_id == "t1"
    assert task.status == TaskStatus.PENDING
    assert "t0" in task.dependencies

    d = task.to_dict()
    assert d["id"] == "t1"
    assert d["task_id"] == "t1"
    assert d["assigned_subagent"] == "coder"
    assert d["status"] == "pending"

    recreated = SubagentTask.from_dict(d)
    assert recreated.task_id == "t1"
    assert recreated.description == "Write unit test"
    assert recreated.dependencies == {"t0"}
    assert recreated.assigned_subagent == "coder"


def test_subagent_task_validation():
    with pytest.raises(ValueError, match="Task ID must be a non-empty string"):
        SubagentTask(task_id="")

    with pytest.raises(ValueError, match="Dictionary must specify 'task_id' or 'id'"):
        SubagentTask.from_dict({"description": "no id"})


def test_graph_add_get_remove_task():
    graph = ClaudeSubagentDependencyGraphV2()
    task1 = SubagentTask(task_id="t1", description="Task 1")
    graph.add_task(task1)

    assert graph.has_task("t1")
    assert graph.get_task("t1").description == "Task 1"
    assert len(graph.get_all_tasks()) == 1

    # Add duplicate task
    with pytest.raises(ValueError, match="Duplicate task ID"):
        graph.add_task({"id": "t1", "description": "Duplicate"})

    # Add task from dict
    graph.add_task({"id": "t2", "description": "Task 2", "dependencies": ["t1"]})
    assert graph.has_task("t2")
    assert "t1" in graph.get_task("t2").dependencies

    # Remove task
    graph.remove_task("t1")
    assert not graph.has_task("t1")
    assert "t1" not in graph.get_task("t2").dependencies


def test_validate_graph_valid_and_invalid():
    # Valid DAG
    graph = DependencyGraphV2.from_dict_list([
        {"id": "t1", "dependencies": []},
        {"id": "t2", "dependencies": ["t1"]},
        {"id": "t3", "dependencies": ["t1", "t2"]},
    ])
    is_valid, errors = graph.validate_graph()
    assert is_valid
    assert len(errors) == 0

    # Self dependency
    self_dep_graph = DependencyGraphV2.from_dict_list([
        {"id": "t1", "dependencies": ["t1"]},
    ])
    is_valid, errors = self_dep_graph.validate_graph()
    assert not is_valid
    assert any("cannot depend on itself" in e for e in errors)

    # Non-existent dependency
    missing_dep_graph = DependencyGraphV2.from_dict_list([
        {"id": "t1", "dependencies": ["non_existent"]},
    ])
    is_valid, errors = missing_dep_graph.validate_graph()
    assert not is_valid
    assert any("non-existent dependency" in e for e in errors)

    # Cyclic dependencies
    cycle_graph = DependencyGraphV2.from_dict_list([
        {"id": "t1", "dependencies": ["t2"]},
        {"id": "t2", "dependencies": ["t3"]},
        {"id": "t3", "dependencies": ["t1"]},
    ])
    is_valid, errors = cycle_graph.validate_graph()
    assert not is_valid
    assert any("Cycle detected" in e for e in errors)


def test_validate_execution_order():
    graph = DependencyGraphV2.from_dict_list([
        {"id": "t1", "dependencies": []},
        {"id": "t2", "dependencies": ["t1"]},
        {"id": "t3", "dependencies": ["t2"]},
    ])

    # Valid order
    is_valid, errors = graph.validate_execution_order(["t1", "t2", "t3"])
    assert is_valid
    assert len(errors) == 0

    # Out of order execution (t2 before t1)
    is_valid, errors = graph.validate_execution_order(["t2", "t1", "t3"])
    assert not is_valid
    assert any("before its dependencies" in e for e in errors)

    # Missing task in execution order
    is_valid, errors = graph.validate_execution_order(["t1", "t2"])
    assert not is_valid
    assert any("missing tasks" in e for e in errors)

    # Unknown task ID in order
    is_valid, errors = graph.validate_execution_order(["t1", "t2", "t3", "unknown"])
    assert not is_valid
    assert any("does not exist in graph" in e for e in errors)


def test_execution_layers_and_topological_sort():
    graph = DependencyGraphV2.from_dict_list([
        {"id": "t4", "dependencies": ["t2", "t3"]},
        {"id": "t2", "dependencies": ["t1"]},
        {"id": "t3", "dependencies": ["t1"]},
        {"id": "t1", "dependencies": []},
    ])

    layers = graph.get_execution_layers()
    assert len(layers) == 3
    assert {t.task_id for t in layers[0]} == {"t1"}
    assert {t.task_id for t in layers[1]} == {"t2", "t3"}
    assert {t.task_id for t in layers[2]} == {"t4"}

    sorted_tasks = graph.topological_sort()
    sorted_ids = [t.task_id for t in sorted_tasks]
    assert sorted_ids[0] == "t1"
    assert sorted_ids[-1] == "t4"


def test_task_state_transitions_complete_and_failed():
    graph = DependencyGraphV2.from_dict_list([
        {"id": "t1", "dependencies": []},
        {"id": "t2", "dependencies": ["t1"]},
        {"id": "t3", "dependencies": ["t2"]},
    ])

    # Initially t1 is executable
    executable = graph.get_executable_tasks()
    assert [t.task_id for t in executable] == ["t1"]

    # Mark t1 completed
    newly_executable = graph.mark_task_completed("t1", output_data={"result": "ok"})
    assert graph.get_task("t1").status == TaskStatus.COMPLETED
    assert graph.get_task("t1").output_data == {"result": "ok"}
    assert [t.task_id for t in newly_executable] == ["t2"]

    # Mark t2 failed
    blocked = graph.mark_task_failed("t2", error="Network error")
    assert graph.get_task("t2").status == TaskStatus.FAILED
    assert graph.get_task("t3").status == TaskStatus.BLOCKED
    assert any(b.task_id == "t3" for b in blocked)

    # Reset graph
    graph.reset()
    assert graph.get_task("t1").status == TaskStatus.PENDING
    assert graph.get_task("t2").status == TaskStatus.PENDING
    assert graph.get_task("t3").status == TaskStatus.PENDING
