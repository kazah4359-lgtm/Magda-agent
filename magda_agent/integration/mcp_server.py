import json
from typing import Dict, Any, List
from magda_agent.integration.mcp_exporter import MCPExporter

class MCPServer:
    """
    MCP JSON-RPC protocol server interface.
    Handles raw JSON-RPC string payloads and returns JSON string responses.
    """
    def __init__(self, exporter: MCPExporter) -> None:
        """Initializes the MCPServer with an MCPExporter."""
        self.exporter = exporter

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List exported tools.

        Returns:
            A list of MCP-compatible tool definitions.
        """
        return self.exporter.export_tools()

    async def handle_request(self, payload: str) -> str:
        """
        Process a JSON-RPC payload string.

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

        response = await self.exporter.handle_rpc_request(request)
        return json.dumps(response)
