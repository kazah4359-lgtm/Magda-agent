import inspect
from typing import Dict, Any, List, Callable, get_origin
from magda_agent.skills.registry import SkillRegistry

class MCPSkillExporter:
    """
    Exports Magda skills as MCP-compatible JSON-RPC tools.
    Enables external consumption of agent skills via the Model Context Protocol.
    """
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def _get_json_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Extracts JSON schema parameters from the function signature.
        """
        if hasattr(func, "__mcp_schema__"):
            return getattr(func, "__mcp_schema__")

        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name == 'self':
                continue

            # Basic type mapping
            param_type = "string"
            annotation = param.annotation
            origin = get_origin(annotation)

            actual_type = origin if origin is not None else annotation

            if actual_type is int:
                param_type = "integer"
            elif actual_type is float:
                param_type = "number"
            elif actual_type is bool:
                param_type = "boolean"
            elif actual_type in (list, List):
                param_type = "array"
            elif actual_type in (dict, Dict):
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

    async def handle_rpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles an incoming JSON-RPC 2.0 request for a tool execution.
        """
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if request.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request: Only JSON-RPC 2.0 is supported."}
            }

        if not method:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "Method not found: Missing method name."}
            }

        if not self.registry.has_skill(method):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method '{method}' not found."}
            }

        try:
            result = self.registry.execute_skill(method, **params)

            # Handle potential error string from registry
            if isinstance(result, str) and result.startswith("Error"):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": result}
                }

            if inspect.isawaitable(result):
                try:
                    result = await result
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32000, "message": f"Error executing async skill '{method}': {e}"}
                    }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(result)}],
                    "isError": False
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": f"Error executing skill '{method}': {e}"}
            }
