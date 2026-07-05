import logging
from typing import Dict, Any, List

class MCPRegistryV5:
    """
    Registry specialized in handling tools exported via the Model Context Protocol (MCP) version 5.
    """

    def __init__(self) -> None:
        """Initialize the MCP Registry V5."""
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}

    def load_tool(self, tool_schema: Dict[str, Any]) -> bool:
        """
        Dynamically loads and verifies an external MCP tool schema.

        Args:
            tool_schema (Dict[str, Any]): The MCP tool schema dictionary.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not self._is_valid_schema(tool_schema):
            logging.error(f"Failed to load MCP tool: Invalid schema {tool_schema}")
            return False

        name: str = tool_schema["name"]
        self.mcp_tools[name] = tool_schema
        logging.info(f"Successfully loaded MCP tool: {name}")
        return True

    def _is_valid_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Verifies if the schema complies with MCP tool standards.

        Args:
            schema (Dict[str, Any]): The MCP tool schema.

        Returns:
            bool: True if valid, False otherwise.
        """
        if not isinstance(schema, dict):
            return False

        required_fields = ["name", "description"]
        for field in required_fields:
            if field not in schema or not isinstance(schema[field], str) or not schema[field]:
                return False

        return True

    def unload_tool(self, name: str) -> bool:
        """
        Dynamically unregisters and removes an MCP tool from the registry.

        Args:
            name (str): The name of the MCP tool to unload.

        Returns:
            bool: True if the tool was successfully unloaded, False if not found.
        """
        if name in self.mcp_tools:
            del self.mcp_tools[name]
            logging.info(f"Successfully unloaded MCP tool: {name}")
            return True
        logging.warning(f"Failed to unload MCP tool: {name} not found in registry.")
        return False
