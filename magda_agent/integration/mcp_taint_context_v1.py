"""MCP Server Taint Context Isolation Exporter (V1).

Exports SkillRegistry skills as MCP tools while enforcing taint tracking
and taint propagation on exported tool execution responses.
"""
import inspect
from typing import Any, Callable, Dict, List, Optional
from magda_agent.skills.registry import SkillRegistry
from magda_agent.safety.taint import TaintTracker, mark_tainted, is_tainted


class MCPTaintContextExporterV1:
    """
    MCP Server adapter for Magda's SkillRegistry with Taint Context Isolation.
    Ensures taint tags propagate correctly to exported tool responses.
    """

    def __init__(self, registry: SkillRegistry, tracker: Optional[TaintTracker] = None) -> None:
        """Initialize exporter with registry and optional TaintTracker."""
        self.registry = registry
        self.tracker = tracker if tracker is not None else TaintTracker()

    def _get_json_schema(self, func: Callable) -> Dict[str, Any]:
        """Extracts JSON schema parameters from function signature."""
        if hasattr(func, "__mcp_schema__"):
            return getattr(func, "__mcp_schema__")

        sig = inspect.signature(func)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for name, param in sig.parameters.items():
            if name in ("self", "kwargs", "args"):
                continue

            param_type = "string"
            if param.annotation is not inspect.Parameter.empty:
                if param.annotation is int:
                    param_type = "integer"
                elif param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"
                elif param.annotation in (list, List):
                    param_type = "array"
                elif param.annotation in (dict, Dict):
                    param_type = "object"

            properties[name] = {"type": param_type}

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """Lists all registered skills as MCP tool definitions with taint isolation flags."""
        tools: List[Dict[str, Any]] = []
        for name, func in self.registry.skills.items():
            description = self.registry.descriptions.get(name, "")
            schema = self._get_json_schema(func)
            tools.append({
                "name": name,
                "description": description,
                "inputSchema": schema,
                "taintIsolation": True,
            })
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes a skill synchronously, evaluating taint propagation."""
        if not self.registry.has_skill(name):
            return {
                "content": [{"type": "text", "text": f"Error: Tool '{name}' not found."}],
                "isError": True,
                "tainted": False,
            }

        inputs_tainted = self.tracker.is_tainted(arguments)

        try:
            raw_result = self.registry.execute_skill(name, **arguments)
            result_tainted = inputs_tainted or self.tracker.is_tainted(raw_result)
            output_content = str(raw_result)
            is_error = isinstance(raw_result, str) and raw_result.startswith("Error")

            return {
                "content": [{"type": "text", "text": output_content}],
                "isError": is_error,
                "tainted": result_tainted,
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error executing tool {name}: {e}"}],
                "isError": True,
                "tainted": inputs_tainted,
            }

    async def call_tool_async(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes an async skill via MCP protocol format with taint propagation."""
        if not self.registry.has_skill(name):
            return {
                "content": [{"type": "text", "text": f"Error: Tool '{name}' not found."}],
                "isError": True,
                "tainted": False,
            }

        inputs_tainted = self.tracker.is_tainted(arguments)

        try:
            raw_result = self.registry.execute_skill(name, **arguments)
            if inspect.isawaitable(raw_result):
                raw_result = await raw_result

            result_tainted = inputs_tainted or self.tracker.is_tainted(raw_result)
            output_content = str(raw_result)
            is_error = isinstance(raw_result, str) and raw_result.startswith("Error")

            return {
                "content": [{"type": "text", "text": output_content}],
                "isError": is_error,
                "tainted": result_tainted,
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error executing tool {name}: {e}"}],
                "isError": True,
                "tainted": inputs_tainted,
            }
