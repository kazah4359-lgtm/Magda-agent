from typing import Dict, Any, List
import uuid
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.mcp_export import MagdaMCPAdapter

class MCPExporter:
    """
    Exports Magda skills as MCP-compatible JSON-RPC tools.
    Acts as a server-side bridge.
    """
    def __init__(self, registry: SkillRegistry) -> None:
        """Initialize the MCPExporter with a SkillRegistry."""
        self.registry = registry
        self.adapter = MagdaMCPAdapter(registry)

    def export_tools(self) -> List[Dict[str, Any]]:
        """
        Returns a list of exported MCP tools.
        """
        return self.adapter.list_tools()

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
        params = request.get("params", {})

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

        adapter_result = await self.adapter.call_tool_async(method, params)

        # Check if the adapter explicitly reported an error, or if it returned a string
        # that starts with 'Error' (which happens when registry.execute_skill returns an error string).
        is_error = adapter_result.get("isError", False)
        error_msg = adapter_result.get("content", [{"text": ""}])[0].get("text", "")

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
