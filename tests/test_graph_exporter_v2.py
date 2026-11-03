"""
Unit tests for GraphExporterV2 (Claude Dependency Graph Planner Export V2).
"""

import json
import pytest

from magda_agent.architecture.dependency_graph_v2 import (
    ClaudeSubagentDependencyGraphV2,
    SubagentTask,
    TaskStatus,
)
from magda_agent.planning.graph_exporter_v2 import (
    GraphExporterV2,
    DependencyGraphExporterV2,
    ClaudeDependencyGraphExporterV2,
    MCP_TASK_SCHEMA,
    MCP_TASK_GRAPH_SCHEMA,
)


def test_init_and_aliases():
    """Verifies class aliases and default initialization."""
    assert GraphExporterV2 is DependencyGraphExporterV2
    assert GraphExporterV2 is ClaudeDependencyGraphExporterV2

    exporter = GraphExporterV2()
    assert exporter.graph_id == "default_dag"
    assert len(exporter.graph.get_all_tasks()) == 0


def test_serialize_single_task_to_mcp():
    """Verifies serialization of a single SubagentTask into MCP task format."""
    task = SubagentTask(
        task_id="t_prepare",
        description="Prepare environment",
        dependencies={"t_init"},
        assigned_subagent="devops_agent",
        required_capability="docker",
        status=TaskStatus.RUNNING,
        input_data={"image": "python:3.12"},
    )
    exporter = GraphExporterV2()
    mcp_task = exporter.serialize_task_to_mcp(task, layer_index=1)

    assert mcp_task["id"] == "t_prepare"
    assert mcp_task["name"] == "t_prepare"
    assert mcp_task["description"] == "Prepare environment"
    assert mcp_task["status"] == "running"
    assert mcp_task["dependencies"] == ["t_init"]
    assert mcp_task["input"] == {"image": "python:3.12"}
    assert mcp_task["metadata"]["assigned_subagent"] == "devops_agent"
    assert mcp_task["metadata"]["required_capability"] == "docker"
    assert mcp_task["metadata"]["layer_index"] == 1


def test_serialize_task_from_dict():
    """Verifies serialization of task specified as dictionary."""
    task_dict = {
        "id": "t_verify",
        "description": "Run smoke tests",
        "dependencies": ["t_build"],
        "status": "pending",
        "input_data": {"test_target": "unit"},
    }
    exporter = GraphExporterV2()
    mcp_task = exporter.serialize_task_to_mcp(task_dict)

    assert mcp_task["id"] == "t_verify"
    assert mcp_task["status"] == "pending"
    assert mcp_task["dependencies"] == ["t_build"]
    assert mcp_task["input"] == {"test_target": "unit"}


def test_export_to_mcp_format_and_schema():
    """Verifies complete DAG export to MCP format with Kahn layers and JSON schema validation."""
    graph = ClaudeSubagentDependencyGraphV2.from_dict_list([
        {"id": "t1", "description": "Step 1", "dependencies": []},
        {"id": "t2", "description": "Step 2", "dependencies": ["t1"]},
        {"id": "t3", "description": "Step 3", "dependencies": ["t1"]},
        {"id": "t4", "description": "Step 4", "dependencies": ["t2", "t3"]},
    ])
    graph.get_task("t1").status = TaskStatus.COMPLETED
    graph.get_task("t1").output_data = {"status": "success"}

    exporter = GraphExporterV2(graph=graph, graph_id="pipeline_v2")
    exported = exporter.export_to_mcp_format(extra_metadata={"environment": "production"})

    assert exported["graph_id"] == "pipeline_v2"
    assert len(exported["tasks"]) == 4
    assert exported["summary"]["total_tasks"] == 4
    assert exported["summary"]["completed"] == 1
    assert exported["summary"]["pending"] == 3
    assert exported["summary"]["failed"] == 0
    assert exported["metadata"] == {"environment": "production"}

    # Kahn parallel execution layers
    assert exported["execution_layers"] == [["t1"], ["t2", "t3"], ["t4"]]
    assert exported["topological_order"][0] == "t1"
    assert exported["topological_order"][-1] == "t4"

    # Dependency edges
    edges = exported["edges"]
    assert {"from": "t1", "to": "t2"} in edges
    assert {"from": "t1", "to": "t3"} in edges
    assert {"from": "t2", "to": "t4"} in edges
    assert {"from": "t3", "to": "t4"} in edges

    # Validate JSON Schema compliance
    assert exporter.validate_export_schema(exported) is True


def test_to_json_export():
    """Verifies JSON string export."""
    steps = [
        {"id": "step_a", "description": "Step A", "dependencies": []},
        {"id": "step_b", "description": "Step B", "dependencies": ["step_a"]},
    ]
    exporter = GraphExporterV2.from_steps(steps, graph_id="step_graph")
    json_str = exporter.to_json(indent=2)
    assert isinstance(json_str, str)

    parsed = json.loads(json_str)
    assert parsed["graph_id"] == "step_graph"
    assert len(parsed["tasks"]) == 2
    assert exporter.validate_export_schema(parsed) is True


def test_factory_methods():
    """Verifies factory constructors from_graph and from_steps."""
    graph = ClaudeSubagentDependencyGraphV2()
    graph.add_task(SubagentTask(task_id="t_root", description="Root task"))

    exp1 = GraphExporterV2.from_graph(graph, graph_id="g1")
    assert exp1.graph_id == "g1"
    assert len(exp1.export_tasks()) == 1

    steps = [{"id": "s1", "dependencies": []}]
    exp2 = GraphExporterV2.from_steps(steps, graph_id="g2")
    assert exp2.graph_id == "g2"
    assert exp2.graph.has_task("s1")


def test_jsonrpc_handling():
    """Verifies JSON-RPC 2.0 interface for tasks/list, tasks/get, and tasks/graph."""
    graph = ClaudeSubagentDependencyGraphV2.from_dict_list([
        {"id": "task_alpha", "description": "Alpha task", "dependencies": []},
        {"id": "task_beta", "description": "Beta task", "dependencies": ["task_alpha"]},
    ])
    exporter = GraphExporterV2(graph=graph)

    # 1. Invalid JSON-RPC request
    res_err = exporter.handle_jsonrpc({"id": "1", "jsonrpc": "1.0", "method": "tasks/list"})
    assert "error" in res_err
    assert res_err["error"]["code"] == -32600

    # 2. Unknown method
    res_unknown = exporter.handle_jsonrpc({"id": "2", "jsonrpc": "2.0", "method": "unknown/method"})
    assert "error" in res_unknown
    assert res_unknown["error"]["code"] == -32601

    # 3. tasks/list
    res_list = exporter.handle_jsonrpc({"id": "3", "jsonrpc": "2.0", "method": "tasks/list"})
    assert res_list["id"] == "3"
    assert "result" in res_list
    assert len(res_list["result"]["tasks"]) == 2
    assert res_list["result"]["summary"]["total_tasks"] == 2

    # 4. tasks/graph
    res_graph = exporter.handle_jsonrpc({"id": "4", "jsonrpc": "2.0", "method": "tasks/graph"})
    assert res_graph["id"] == "4"
    assert "tasks" in res_graph["result"]
    assert "execution_layers" in res_graph["result"]

    # 5. tasks/get successful
    res_get = exporter.handle_jsonrpc({
        "id": "5",
        "jsonrpc": "2.0",
        "method": "tasks/get",
        "params": {"id": "task_alpha"},
    })
    assert res_get["id"] == "5"
    assert res_get["result"]["id"] == "task_alpha"
    assert res_get["result"]["description"] == "Alpha task"

    # 6. tasks/get missing id
    res_get_missing = exporter.handle_jsonrpc({
        "id": "6",
        "jsonrpc": "2.0",
        "method": "tasks/get",
        "params": {},
    })
    assert "error" in res_get_missing
    assert res_get_missing["error"]["code"] == -32602

    # 7. tasks/get non-existent
    res_get_not_found = exporter.handle_jsonrpc({
        "id": "7",
        "jsonrpc": "2.0",
        "method": "tasks/get",
        "params": {"id": "task_missing"},
    })
    assert "error" in res_get_not_found
    assert res_get_not_found["error"]["code"] == -32004


def test_error_and_blocked_task_export():
    """Verifies serialization with failed and blocked tasks."""
    graph = ClaudeSubagentDependencyGraphV2.from_dict_list([
        {"id": "t1", "dependencies": []},
        {"id": "t2", "dependencies": ["t1"]},
    ])
    graph.mark_task_failed("t1", error="Connection timeout")

    exporter = GraphExporterV2(graph=graph)
    exported = exporter.export_to_mcp_format()

    assert exported["summary"]["failed"] == 1
    assert exported["summary"]["blocked"] == 1
    t1_exported = next(t for t in exported["tasks"] if t["id"] == "t1")
    t2_exported = next(t for t in exported["tasks"] if t["id"] == "t2")

    assert t1_exported["status"] == "failed"
    assert t1_exported["error"] == "Connection timeout"
    assert t2_exported["status"] == "blocked"
    assert "Blocked due to failure" in t2_exported["error"]
