"""
Claude Dependency Graph Planner Export V2 module.

Inspired by the Claude Agent SDK task system:
Serializes internal dependency graphs into standard MCP task format
for external observability, telemetry, and integration with MCP clients/dashboards.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Union

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

from magda_agent.architecture.dependency_graph_v2 import (
    ClaudeSubagentDependencyGraphV2,
    SubagentTask,
    TaskStatus,
)

logger = logging.getLogger(__name__)


MCP_TASK_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MCPTask",
    "type": "object",
    "required": ["id", "name", "status", "dependencies"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["pending", "running", "completed", "failed", "blocked", "skipped"],
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "string"},
        },
        "input": {"type": ["object", "null"]},
        "output": {},
        "error": {"type": ["string", "null"]},
        "metadata": {"type": "object"},
    },
}

MCP_TASK_GRAPH_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MCPTaskGraph",
    "type": "object",
    "required": ["tasks", "execution_layers", "topological_order", "summary"],
    "properties": {
        "graph_id": {"type": "string"},
        "tasks": {
            "type": "array",
            "items": MCP_TASK_SCHEMA,
        },
        "execution_layers": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "topological_order": {
            "type": "array",
            "items": {"type": "string"},
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                },
            },
        },
        "summary": {
            "type": "object",
            "required": ["total_tasks", "completed", "failed", "blocked", "pending"],
            "properties": {
                "total_tasks": {"type": "integer"},
                "completed": {"type": "integer"},
                "failed": {"type": "integer"},
                "blocked": {"type": "integer"},
                "pending": {"type": "integer"},
                "running": {"type": "integer"},
                "skipped": {"type": "integer"},
            },
        },
        "metadata": {"type": "object"},
    },
}


class GraphExporterV2:
    """
    GraphExporterV2 for serializing internal dependency graphs into standard MCP task format.

    Supports conversion of ClaudeSubagentDependencyGraphV2, SubagentTask collections,
    or raw step lists into standard MCP task observability schemas, and facilitates
    JSON-RPC 2.0 query resolution (such as 'tasks/list', 'tasks/get').
    """

    def __init__(
        self,
        graph: Optional[Union[ClaudeSubagentDependencyGraphV2, List[Union[SubagentTask, Dict[str, Any]]]]] = None,
        graph_id: str = "default_dag",
    ) -> None:
        """
        Initializes GraphExporterV2.

        Args:
            graph: Dependency graph instance or collection of tasks/dicts.
            graph_id: Identifier for the exported graph.
        """
        self.graph_id = graph_id
        if isinstance(graph, ClaudeSubagentDependencyGraphV2):
            self.graph = graph
        elif isinstance(graph, list):
            self.graph = ClaudeSubagentDependencyGraphV2.from_dict_list(
                [t.to_dict() if isinstance(t, SubagentTask) else t for t in graph]
            )
        elif graph is None:
            self.graph = ClaudeSubagentDependencyGraphV2()
        else:
            # Fallback for any graph-like object with get_all_tasks
            if hasattr(graph, "get_all_tasks"):
                self.graph = graph
            else:
                raise TypeError(f"Unsupported graph type: {type(graph)}")

    def serialize_task_to_mcp(self, task: Union[SubagentTask, Dict[str, Any]], layer_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Serializes a single task node into the standard MCP task format.

        Args:
            task: SubagentTask instance or dictionary representation.
            layer_index: Optional layer index indicating execution order depth.

        Returns:
            Dict[str, Any]: Conforming to MCP_TASK_SCHEMA.
        """
        if isinstance(task, dict):
            task_obj = SubagentTask.from_dict(task)
        elif isinstance(task, SubagentTask):
            task_obj = task
        else:
            raise TypeError(f"Expected SubagentTask or dict, got {type(task)}")

        status_str = task_obj.status.value if isinstance(task_obj.status, TaskStatus) else str(task_obj.status)

        metadata: Dict[str, Any] = {}
        if task_obj.assigned_subagent:
            metadata["assigned_subagent"] = task_obj.assigned_subagent
        if task_obj.required_capability:
            metadata["required_capability"] = task_obj.required_capability
        if layer_index is not None:
            metadata["layer_index"] = layer_index

        mcp_task = {
            "id": task_obj.task_id,
            "name": task_obj.task_id,
            "description": task_obj.description or "",
            "status": status_str,
            "dependencies": sorted(list(task_obj.dependencies)),
            "input": task_obj.input_data if task_obj.input_data else {},
            "output": task_obj.output_data,
            "error": task_obj.error,
            "metadata": metadata,
        }

        return mcp_task

    def export_tasks(self) -> List[Dict[str, Any]]:
        """
        Exports all tasks in the dependency graph formatted according to the MCP task schema.

        Returns:
            List[Dict[str, Any]]: List of MCP-formatted task dictionaries.
        """
        # Map task_id to its parallel execution layer index if valid DAG
        layer_indices: Dict[str, int] = {}
        try:
            layers = self.graph.get_execution_layers()
            for idx, layer in enumerate(layers):
                for task in layer:
                    layer_indices[task.task_id] = idx
        except Exception as exc:
            logger.debug("Could not calculate execution layers for task indexing: %s", exc)

        tasks = self.graph.get_all_tasks()
        return [
            self.serialize_task_to_mcp(t, layer_index=layer_indices.get(t.task_id))
            for t in tasks
        ]

    def export_to_mcp_format(self, extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Serializes the entire dependency graph into the standard MCP task graph format.

        Returns:
            Dict[str, Any]: Conforming to MCP_TASK_GRAPH_SCHEMA.
        """
        tasks = self.export_tasks()

        # Compute execution layers
        execution_layers: List[List[str]] = []
        topological_order: List[str] = []
        try:
            layers = self.graph.get_execution_layers()
            for layer in layers:
                execution_layers.append([t.task_id for t in layer])

            sorted_tasks = self.graph.topological_sort()
            topological_order = [t.task_id for t in sorted_tasks]
        except Exception as exc:
            logger.warning("Could not compute Kahn layers/topological sort: %s", exc)
            # Fallback to simple task id listing
            topological_order = [t["id"] for t in tasks]
            execution_layers = [[t["id"] for t in tasks]]

        # Compute dependency edges
        edges: List[Dict[str, str]] = []
        for task in self.graph.get_all_tasks():
            for dep in sorted(list(task.dependencies)):
                edges.append({"from": dep, "to": task.task_id})

        # Summary counts
        all_tasks = self.graph.get_all_tasks()
        summary = {
            "total_tasks": len(all_tasks),
            "completed": sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in all_tasks if t.status == TaskStatus.FAILED),
            "blocked": sum(1 for t in all_tasks if t.status == TaskStatus.BLOCKED),
            "pending": sum(1 for t in all_tasks if t.status == TaskStatus.PENDING),
            "running": sum(1 for t in all_tasks if t.status == TaskStatus.RUNNING),
            "skipped": sum(1 for t in all_tasks if t.status == TaskStatus.SKIPPED),
        }

        payload: Dict[str, Any] = {
            "graph_id": self.graph_id,
            "tasks": tasks,
            "execution_layers": execution_layers,
            "topological_order": topological_order,
            "edges": edges,
            "summary": summary,
            "metadata": extra_metadata or {},
        }

        return payload

    def to_json(self, indent: Optional[int] = 2, extra_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Serializes graph to a JSON string representation in MCP format.
        """
        return json.dumps(self.export_to_mcp_format(extra_metadata=extra_metadata), indent=indent)

    def validate_export_schema(self, exported_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validates the exported dictionary against the MCP task graph JSON Schema.

        Returns:
            bool: True if schema validation succeeds or jsonschema is not installed.
        """
        data = exported_data if exported_data is not None else self.export_to_mcp_format()
        if not HAS_JSONSCHEMA:
            # Fallback validation without jsonschema library
            required_top = ["tasks", "execution_layers", "topological_order", "summary"]
            if not all(k in data for k in required_top):
                return False
            for t in data.get("tasks", []):
                if not all(k in t for k in ["id", "name", "status", "dependencies"]):
                    return False
            return True

        jsonschema.validate(instance=data, schema=MCP_TASK_GRAPH_SCHEMA)
        return True

    def handle_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles JSON-RPC 2.0 requests for task observability.
        Supported methods:
        - "tasks/list": Returns all MCP tasks and execution summary.
        - "tasks/get": Returns a specific MCP task by task_id / id in params.
        - "tasks/graph": Returns the full MCP task graph structure.

        Args:
            request: JSON-RPC 2.0 request dictionary.

        Returns:
            Dict[str, Any]: JSON-RPC 2.0 response dictionary.
        """
        req_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request: jsonrpc version must be '2.0'"},
            }

        method = request.get("method")
        params = request.get("params", {}) or {}

        if method == "tasks/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tasks": self.export_tasks(),
                    "summary": self.export_to_mcp_format()["summary"],
                },
            }
        elif method == "tasks/graph":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": self.export_to_mcp_format(),
            }
        elif method == "tasks/get":
            task_id = params.get("id") or params.get("task_id")
            if not task_id:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "Missing 'id' or 'task_id' parameter"},
                }
            task = self.graph.get_task(str(task_id))
            if not task:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32004, "message": f"Task '{task_id}' not found"},
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": self.serialize_task_to_mcp(task),
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method '{method}' not found"},
            }

    @classmethod
    def from_graph(
        cls,
        graph: ClaudeSubagentDependencyGraphV2,
        graph_id: str = "default_dag",
    ) -> "GraphExporterV2":
        """Factory constructor from an existing ClaudeSubagentDependencyGraphV2."""
        return cls(graph=graph, graph_id=graph_id)

    @classmethod
    def from_steps(
        cls,
        steps: List[Dict[str, Any]],
        graph_id: str = "steps_dag",
    ) -> "GraphExporterV2":
        """Factory constructor from a list of step dictionaries."""
        return cls(graph=steps, graph_id=graph_id)


# Aliases for compatibility
DependencyGraphExporterV2 = GraphExporterV2
ClaudeDependencyGraphExporterV2 = GraphExporterV2
