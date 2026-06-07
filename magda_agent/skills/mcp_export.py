import inspect
from typing import Dict, Any, List
from magda_agent.skills.registry import SkillRegistry

class MagdaMCPAdapter:
    """
    MCP Server adapter for Magda's SkillRegistry.
    Exposes Magda skills as MCP-compatible tools.
    """
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def _get_json_schema(self, func) -> Dict[str, Any]:
        """
        Extracts JSON schema parameters from the function signature.
        """
        if hasattr(func, "__mcp_schema__"):
            return getattr(func, "__mcp_schema__")

        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name in ('self', 'kwargs', 'args'):
                continue

            param_type = "string"
            if param.annotation is not inspect.Parameter.empty:
                if param.annotation is int:
                    param_type = "integer"
                elif param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"
                elif param.annotation is list or param.annotation is List:
                    param_type = "array"
                elif param.annotation is dict or param.annotation is Dict:
                    param_type = "object"

            properties[name] = {"type": param_type}

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists all registered skills as MCP-compatible tool definitions.
        """
        tools = []
        for name, func in self.registry.skills.items():
            description = self.registry.descriptions.get(name, "")
            schema = self._get_json_schema(func)
            tools.append({
                "name": name,
                "description": description,
                "inputSchema": schema
            })
        return tools

    async def call_tool_async(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes an async skill via the MCP protocol format.
        """
        if not self.registry.has_skill(name):
            return {
                "content": [{"type": "text", "text": f"Error: Tool '{name}' not found."}],
                "isError": True
            }

        try:
            result = self.registry.execute_skill(name, **arguments)
            if inspect.isawaitable(result):
                try:
                    result = await result
                except Exception as e:
                    return {
                        "content": [{"type": "text", "text": f"Error executing async tool {name}: {e}"}],
                        "isError": True
                    }
            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error executing tool {name}: {e}"}],
                "isError": True
            }

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes a skill via the MCP protocol format.
        """
        if not self.registry.has_skill(name):
            return {
                "content": [{"type": "text", "text": f"Error: Tool '{name}' not found."}],
                "isError": True
            }

        try:
            result = self.registry.execute_skill(name, **arguments)
            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error executing tool {name}: {e}"}],
                "isError": True
            }
