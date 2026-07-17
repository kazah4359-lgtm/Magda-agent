from typing import Optional, Tuple

class MCPToolNameParser:
    """
    Parser for handling MCP (Model Context Protocol) server-prefixed tool names.
    Extracts the server prefix and the underlying tool name from a prefixed string.
    """

    @staticmethod
    def parse(tool_name: str) -> Tuple[Optional[str], str]:
        """
        Parses a server-prefixed tool name.

        Args:
            tool_name: The tool name, possibly prefixed with a server ID and an underscore.

        Returns:
            A tuple containing the server prefix (or None if no prefix is found) and the
            actual tool name.
        """
        if not tool_name:
            return None, ""

        parts = tool_name.split("_", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]

        return None, tool_name

    @staticmethod
    def apply_prefix(server_id: str, tool_name: str) -> str:
        """
        Applies a server prefix to a tool name, ensuring no double prefixing occurs.

        Args:
            server_id: The server ID to prefix.
            tool_name: The base tool name.

        Returns:
            The prefixed tool name.
        """
        if not server_id:
            return tool_name

        prefix = f"{server_id}_"
        if tool_name.startswith(prefix):
            return tool_name

        return f"{prefix}{tool_name}"
