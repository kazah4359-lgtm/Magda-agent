import json
from typing import Dict, Any, List
from magda_agent.integration.mcp_exporter import MCPExporter

class MCPServer:
    """
    MCP JSON-RPC protocol server interface.
    Handles raw JSON-RPC string payloads and returns JSON string responses.
    """
    def __init__(self, exporter: MCPExporter, server_id: str = "magda") -> None:
        """Initializes the MCPServer with an MCPExporter and a server_id."""
        self.exporter = exporter
        self.server_id = server_id

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List exported tools with server-prefixed names.

        Returns:
            A list of MCP-compatible tool definitions.
        """
        tools = self.exporter.export_tools()
        if not self.server_id:
            return tools

        for tool in tools:
            # Avoid double prefixing if already prefixed
            prefix = f"{self.server_id}_"
            if not tool["name"].startswith(prefix):
                tool["name"] = f"{prefix}{tool['name']}"
        return tools

    async def handle_request(self, payload: str) -> str:
        """
        Process a JSON-RPC payload string.
        Strips the server_id prefix from the method name if present.

        Args:
            payload: A JSON string representing the RPC request.

        Returns:
            A JSON string representing the RPC response.
        """
        try:
            request = json.loads(payload)
        except json.JSONDecodeError:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            })

        method = request.get("method", "")
        if method and isinstance(method, str) and self.server_id:
            prefix = f"{self.server_id}_"
            if method.startswith(prefix):
                request["method"] = method[len(prefix):]

        response = await self.exporter.handle_rpc_request(request)
        return json.dumps(response)
