import logging
from typing import Dict, Any, List, Optional, Callable

class MCPRouterV3:
    """
    Router that handles tools with server prefixes, according to MCP standard trends.
    It can route tool calls of the form 'server_name_tool_name' to the appropriate server handler.
    """

    def __init__(self):
        """Initializes the MCP Router V3."""
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

    def route_tool(self, tool_call_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Routes a tool call by its prefixed name.

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

        # We split on the first underscore only, assuming the format is <server_name>_<tool_name>
        # Alternatively, we could iterate over registered server names to find a prefix match.
        # Let's find the longest matching registered server prefix.

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
