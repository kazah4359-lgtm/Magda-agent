import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class MCPServerPrefixedToolsV3:
    """
    Manages MCP action tools by automatically prefixing their names with a server ID.
    Inspired by OpenAI Agents SDK tool namespacing.
    """

    def __init__(self) -> None:
        """
        Initialize the MCPServerPrefixedToolsV3.
        """
        self.tools: Dict[str, Dict[str, Any]] = {}

    def format_tool_name(self, server_id: str, tool_name: str) -> str:
        """
        Formats a tool name with the given server ID prefix.

        Args:
            server_id: The ID of the MCP server.
            tool_name: The original name of the tool.

        Returns:
            The prefixed tool name string.
        """
        return f"{server_id}_{tool_name}"

    def register_tool(self, server_id: str, tool_name: str, description: str, input_schema: Dict[str, Any]) -> str:
        """
        Registers a tool with an automatic server prefix.

        Args:
            server_id: The ID of the MCP server providing the tool.
            tool_name: The original name of the tool.
            description: A description of the tool.
            input_schema: JSON schema for tool inputs.

        Returns:
            The prefixed name under which the tool was registered.
        """
        prefixed_name = self.format_tool_name(server_id, tool_name)

        self.tools[prefixed_name] = {
            "name": prefixed_name,
            "original_name": tool_name,
            "server_id": server_id,
            "description": description,
            "inputSchema": input_schema,
            "is_action": True
        }

        logger.info(f"Registered MCP prefixed tool: {prefixed_name}")
        return prefixed_name

    def get_tool(self, prefixed_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a tool's metadata using its prefixed name.

        Args:
            prefixed_name: The namespaced tool name.

        Returns:
            The tool metadata dict if found, else None.
        """
        return self.tools.get(prefixed_name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists all registered prefixed tools.

        Returns:
            A list of tool metadata dictionaries.
        """
        return list(self.tools.values())
