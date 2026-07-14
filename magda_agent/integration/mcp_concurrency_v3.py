import asyncio
import inspect
import json
import threading
import uuid
from typing import List, Dict, Any, Optional

from magda_agent.integration.mcp_server import MCPServer


class MCPConcurrentHandlerV3:
    """
    Handles MCP server-prefixed tool names and runtime function tool concurrency.
    Supports executing multiple MCP tools in parallel using asyncio and provides
    thread-safe task tracking to ensure safe operations.
    Supports dynamic mapping of multiple MCP servers with namespace prefixes and
    supports separators like '__', '-', and '_'.
    """

    def __init__(
        self,
        server: Optional[MCPServer] = None,
        server_prefix: str = "",
        max_concurrency: int = 10,
        separators: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the MCPConcurrentHandlerV3.

        Args:
            server: Optional initial MCPServer instance to register.
            server_prefix: Optional prefix for the initial server.
            max_concurrency: The maximum number of concurrent tool executions allowed.
            separators: Optional list of separators to check for namespacing.
        """
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.lock = threading.Lock()
        self.active_tasks_count = 0
        self.separators = separators or ["__", "-", "_"]
        self.servers: Dict[str, MCPServer] = {}

        if server is not None:
            self.register_server(server_prefix, server)

    def register_server(self, prefix: str, server: MCPServer) -> None:
        """
        Registers an MCPServer with a given prefix.

        Args:
            prefix: The server prefix/namespace.
            server: The MCPServer instance.
        """
        self.servers[prefix] = server

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists available tools across all registered servers, optionally prepending prefixes.

        Returns:
            A list of dictionary definitions for MCP compatible tools.
        """
        all_tools = []
        for prefix, server in self.servers.items():
            tools = server.list_tools()
            if not prefix:
                all_tools.extend(tools)
                continue

            for tool in tools:
                new_tool = dict(tool)
                tool_name = tool["name"]

                # Avoid double prefixing with server_id if applicable
                server_id_prefix = f"{server.server_id}_" if getattr(server, "server_id", None) else ""
                if server_id_prefix and tool_name.startswith(server_id_prefix):
                    tool_name = tool_name[len(server_id_prefix):]

                sep = self.separators[0] if self.separators else "__"
                new_tool["name"] = f"{prefix}{sep}{tool_name}"
                all_tools.append(new_tool)

        return all_tools

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
                if isinstance(req, dict):
                    tasks.append(self._process_single_request_with_concurrency_safe(req.copy()))
                else:
                    # Handle invalid payload element in batch
                    async def invalid_element() -> Dict[str, Any]:
                        return {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32600, "message": "Invalid Request"}
                        }
                    tasks.append(invalid_element())

            results = await asyncio.gather(*tasks)
            return json.dumps(results)
        else:
            result = await self._process_single_request_with_concurrency_safe(request_data.copy())
            return json.dumps(result)

    async def _process_single_request_with_concurrency_safe(self, req: dict) -> dict:
        """
        A safe wrapper around single request processing to catch unhandled exceptions.

        Args:
            req: A dictionary representing a single JSON-RPC request.

        Returns:
            A dictionary representing the JSON-RPC response.
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
        Processes a single JSON-RPC request dictionary, mapping and stripping the server prefix.

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

        req_id = req.get("id", str(uuid.uuid4()))
        method = req.get("method")
        params = req.get("params", {})

        if not method or not isinstance(method, str):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "Method not found"}
            }

        # Resolve server prefix
        matched_prefix = None
        matched_server = None
        matched_sep = None

        sorted_prefixes = sorted(self.servers.keys(), key=len, reverse=True)
        for prefix in sorted_prefixes:
            if not prefix:
                continue
            for sep in self.separators:
                full_prefix = f"{prefix}{sep}"
                if method.startswith(full_prefix):
                    matched_prefix = prefix
                    matched_server = self.servers[prefix]
                    matched_sep = sep
                    break
            if matched_server:
                break

        if not matched_server and "" in self.servers:
            matched_prefix = ""
            matched_server = self.servers[""]

        if not matched_server:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found or missing server prefix: {method}"}
            }

        # Strip prefix for inner execution
        if matched_prefix:
            stripped_method = method[len(matched_prefix) + len(matched_sep):]
            req["method"] = stripped_method
            method_to_check = stripped_method
        else:
            method_to_check = method

        # Identify if the skill is async or sync
        is_async = True
        registry = getattr(getattr(matched_server, "exporter", None), "registry", None)
        if registry and hasattr(registry, "skills") and method_to_check in registry.skills:
            skill_func = registry.skills[method_to_check]
            if not inspect.iscoroutinefunction(skill_func):
                is_async = False

        if not is_async:
            # Run sync tool inside thread to avoid blocking the asyncio event loop
            def run_sync() -> Any:
                return registry.execute_skill(method_to_check, **params)

            try:
                result = await asyncio.to_thread(run_sync)
                result_str = str(result)
                if result_str.startswith("Error"):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32000, "message": result_str}
                    }

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result_str}],
                        "isError": False
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": f"Error executing tool {method_to_check}: {e}"}
                }
        else:
            # Async execution
            return await matched_server.exporter.handle_rpc_request(req)
