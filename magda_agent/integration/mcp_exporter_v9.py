from typing import Dict, Any, List, Optional
import uuid
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.mcp_export import MagdaMCPAdapter

class MCPExporterV9:
    """
    Exports Magda skills as MCP-compatible JSON-RPC tools with advanced validation.
    Acts as a server-side bridge.
    """
    def __init__(self, registry: SkillRegistry) -> None:
        """Initialize the MCPExporterV9 with a SkillRegistry."""
        self.registry = registry
        self.adapter = MagdaMCPAdapter(registry)

    def export_tools(self) -> List[Dict[str, Any]]:
        """
        Returns a list of exported MCP tools.
        """
        return self.adapter.list_tools()

    def _validate_schema(self, schema: Dict[str, Any], arguments: Dict[str, Any]) -> Optional[str]:
        """
        Validates arguments against a simplified JSON schema.

        Args:
            schema: The tool's inputSchema.
            arguments: The arguments provided in the request.

        Returns:
            An error message if validation fails, or None if validation succeeds.
        """
        if schema.get("type") != "object":
            return None # Skip validation if not an object schema

        required_fields = schema.get("required", [])
        for req in required_fields:
            if req not in arguments:
                return f"Missing required parameter: '{req}'"

        properties = schema.get("properties", {})
        for arg_name, arg_value in arguments.items():
            if arg_name not in properties:
                # Based on strict validation, extra args could be allowed or disallowed.
                # Let's just check the ones that are declared.
                continue

            expected_type = properties[arg_name].get("type")

            # Basic type checking
            if expected_type == "string" and not isinstance(arg_value, str):
                return f"Parameter '{arg_name}' must be of type string"
            elif expected_type == "integer" and (not isinstance(arg_value, int) or isinstance(arg_value, bool)):
                # bool is a subclass of int in python, so we check explicitly
                return f"Parameter '{arg_name}' must be of type integer"
            elif expected_type == "number" and (not isinstance(arg_value, (int, float)) or isinstance(arg_value, bool)):
                return f"Parameter '{arg_name}' must be of type number"
            elif expected_type == "boolean" and not isinstance(arg_value, bool):
                return f"Parameter '{arg_name}' must be of type boolean"
            elif expected_type == "array" and not isinstance(arg_value, list):
                return f"Parameter '{arg_name}' must be of type array"
            elif expected_type == "object" and not isinstance(arg_value, dict):
                return f"Parameter '{arg_name}' must be of type object"

        return None

    async def handle_rpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles an incoming JSON-RPC request for a tool execution.

        Args:
            request: A dictionary representing a JSON-RPC 2.0 request.

        Returns:
            A dictionary representing a JSON-RPC 2.0 response.
        """
        req_id = request.get("id", str(uuid.uuid4()))
        method = request.get("method")
        params: Dict[str, Any] = request.get("params", {})

        arguments: Dict[str, Any] = params.get("arguments", params)

        if request.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request"}
            }

        if not method:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "Method not found"}
            }

        if not self.registry.has_skill(method):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method '{method}' not found"}
            }

        # Advanced Validation
        tools = self.export_tools()
        tool_schema = None
        for t in tools:
            if t["name"] == method:
                tool_schema = t.get("inputSchema")
                break

        if tool_schema:
            validation_error = self._validate_schema(tool_schema, arguments)
            if validation_error:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"Invalid params: {validation_error}"}
                }

        adapter_result = await self.adapter.call_tool_async(method, arguments)

        # Check if the adapter explicitly reported an error, or if it returned a string
        # that starts with 'Error' (which happens when registry.execute_skill returns an error string).
        is_error: bool = adapter_result.get("isError", False)

        # safely extract error message
        content = adapter_result.get("content", [])
        error_msg = ""
        if content and len(content) > 0:
             error_msg = content[0].get("text", "")

        if is_error or str(error_msg).startswith("Error"):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": error_msg or "Unknown error"}
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": adapter_result
        }
