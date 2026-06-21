import asyncio
import json
import threading
from typing import List, Dict, Any
from magda_agent.integration.mcp_server import MCPServer

class MCPConcurrentHandlerV2:
    """
    Handles MCP server-prefixed tool names and runtime function tool concurrency.
    Supports executing multiple MCP tools in parallel using asyncio and provides
    thread-safe task tracking to ensure safe operations.
    """
    def __init__(self, server: MCPServer, server_prefix: str = "", max_concurrency: int = 10) -> None:
        """
        Initializes the MCPConcurrentHandlerV2.

        Args:
            server: The MCPServer instance to handle execution.
            server_prefix: Optional string prefix to namespace tools.
            max_concurrency: The maximum number of concurrent tool executions allowed.
        """
        self.server = server
        self.server_prefix = server_prefix
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.lock = threading.Lock()
        self.active_tasks_count = 0

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists available tools, optionally prepending the server_prefix to their names.

        Returns:
            A list of dictionary definitions for MCP compatible tools.
        """
        tools = self.server.list_tools()
        if not self.server_prefix:
            return tools

        prefixed_tools = []
        for tool in tools:
            new_tool = dict(tool)
            new_tool["name"] = f"{self.server_prefix}__{tool['name']}"
            prefixed_tools.append(new_tool)
        return prefixed_tools

    async def handle_request(self, payload: str) -> str:
        """
        Handles a JSON-RPC payload string concurrently. Supports batch requests.
        Respects max_concurrency limit.

        Args:
            payload: A raw JSON string containing a single object or an array of objects.

        Returns:
            A JSON string representing the JSON-RPC response or batch responses.
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

            tasks = []
            for req in request_data:
                tasks.append(self._process_single_request_with_concurrency_safe(req.copy()))

            results = await asyncio.gather(*tasks)
            return json.dumps(results)
        else:
            result = await self._process_single_request_with_concurrency_safe(request_data.copy())
            return json.dumps(result)

    async def _process_single_request_with_concurrency_safe(self, req: dict) -> dict:
        """
        A safe wrapper around single request processing to catch unhandled exceptions.
        """
        try:
            return await self._process_single_request_with_concurrency(req)
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req.get("id") if isinstance(req, dict) else None,
                "error": {"code": -32000, "message": f"Internal Error: {str(e)}"}
            }

    async def _process_single_request_with_concurrency(self, req: dict) -> dict:
        """
        Processes a single JSON-RPC request dictionary within the concurrency limits.

        Args:
            req: A dictionary representing a single JSON-RPC request.

        Returns:
            A dictionary representing the JSON-RPC response.
        """
        async with self.semaphore:
            with self.lock:
                self.active_tasks_count += 1

            try:
                return await self._process_single_request(req)
            finally:
                with self.lock:
                    self.active_tasks_count -= 1

    async def _process_single_request(self, req: dict) -> dict:
        """
        Processes a single JSON-RPC request dictionary, stripping the server prefix if applicable.

        Args:
            req: A dictionary representing a single JSON-RPC request.

        Returns:
            A dictionary representing the JSON-RPC response.
        """
        if not isinstance(req, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"}
            }

        method = req.get("method")
        if method and isinstance(method, str) and self.server_prefix:
            prefix_str = f"{self.server_prefix}__"
            if method.startswith(prefix_str):
                req["method"] = method[len(prefix_str):]
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req.get("id"),
                    "error": {"code": -32601, "message": f"Method not found or missing server prefix: {method}"}
                }

        return await self.server.exporter.handle_rpc_request(req)
