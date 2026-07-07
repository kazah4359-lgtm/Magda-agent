import asyncio
import inspect
import logging
from typing import Dict, Any, List, Callable, Coroutine


class MCPConcurrentRouterV2:
    """
    Router that handles concurrent execution of tools with server prefixes,
    according to MCP standard trends.
    It can route multiple tool calls concurrently.
    """

    def __init__(self) -> None:
        """Initializes the MCP Concurrent Router V2."""
        # Maps server name to a routing function that handles that server's tools
        self.server_handlers: Dict[str, Callable[[str, Dict[str, Any]], Any]] = {}

    def register_server(self, server_name: str, handler: Callable[[str, Dict[str, Any]], Any]) -> None:
        """
        Registers a handler for a specific server prefix.

        Args:
            server_name (str): The prefix used to identify the server.
            handler (Callable): Function that takes (tool_name, tool_arguments) and returns a result.
        """
        self.server_handlers[server_name] = handler
        logging.info(f"Registered MCP server handler for prefix: {server_name}")

    def _route_tool(self, tool_call_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Routes a single tool call by its prefixed name.

        Args:
            tool_call_name (str): The full tool name, e.g., 'weather_server_get_forecast'.
            arguments (Dict[str, Any]): The arguments for the tool call.

        Returns:
            Any: The result of the routed tool execution.

        Raises:
            ValueError: If the tool name does not contain a prefix, or if the server prefix is unknown.
        """
        if "_" not in tool_call_name:
            raise ValueError(f"Tool name '{tool_call_name}' must be prefixed with a server name (e.g., 'server_tool').")

        matched_server = None
        matched_tool = None

        for server_name in sorted(self.server_handlers.keys(), key=len, reverse=True):
            prefix = f"{server_name}_"
            if tool_call_name.startswith(prefix):
                matched_server = server_name
                matched_tool = tool_call_name[len(prefix):]
                break

        if not matched_server:
            raise ValueError(f"No registered MCP server handler found for tool: {tool_call_name}")

        handler = self.server_handlers[matched_server]
        return handler(matched_tool, arguments)

    async def execute_concurrently(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """
        Executes a list of tool calls concurrently.
        Each element in tool_calls must have 'name' and 'kwargs'.

        Args:
            tool_calls (List[Dict[str, Any]]): List of tool calls to execute.

        Returns:
            List[Any]: A list of results corresponding to the input tool calls.
        """
        tasks = []
        for call in tool_calls:
            name = call.get("name")
            kwargs = call.get("kwargs", {})

            if not name:
                tasks.append(asyncio.to_thread(lambda: "Error: Tool name is missing."))
                continue

            async def wrap_execution(n: str = name, k: Dict[str, Any] = kwargs) -> Any:
                try:
                    res = await asyncio.to_thread(self._route_tool, n, k)
                    if inspect.isawaitable(res):
                        return await res
                    return res
                except Exception as e:
                    return f"Error executing tool '{n}': {e}"

            tasks.append(wrap_execution())

        return await asyncio.gather(*tasks)
