import json
from typing import Dict, Any, List
from magda_agent.integration.mcp_exporter import MCPExporter

class MCPServer:
    """
    MCP JSON-RPC protocol server interface.
    Handles raw JSON-RPC string payloads and returns JSON string responses.
    Supports single requests, notifications, and batch requests.
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
        Supports single requests, notifications, and batch requests.
        Strips the server_id prefix from the method name if present.

        Args:
            payload: A JSON string representing the RPC request(s).

        Returns:
            A JSON string representing the RPC response(s), or empty string if no response.
        """
        try:
            request_data = json.loads(payload)
        except json.JSONDecodeError:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            })

        if isinstance(request_data, list):
            if not request_data:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"}
                })

            # Process batch requests
            responses = []
            for req in request_data:
                if not isinstance(req, dict):
                    responses.append({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32600, "message": "Invalid Request"}
                    })
                    continue

                resp = await self._handle_single_request(req)
                if resp is not None:
                    responses.append(resp)

            if not responses:
                return ""
            return json.dumps(responses)

        elif isinstance(request_data, dict):
            resp = await self._handle_single_request(request_data)
            if resp is None:
                return ""
            return json.dumps(resp)

        else:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"}
            })

    async def _handle_single_request(self, request: Dict[str, Any]) -> Any:
        """
        Helper to process a single JSON-RPC request or notification.

        Args:
            request: A dictionary representing a single JSON-RPC request.

        Returns:
            A dictionary representing the JSON-RPC response, or None if it is a notification.
        """
        is_notification = "id" not in request

        method = request.get("method", "")
        if method and isinstance(method, str) and self.server_id:
            prefix = f"{self.server_id}_"
            if method.startswith(prefix):
                request["method"] = method[len(prefix):]

        response = await self.exporter.handle_rpc_request(request)

        if is_notification:
            return None
        return response
