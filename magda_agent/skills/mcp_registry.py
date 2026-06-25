import logging
from typing import Dict, Any, List

class MCPRegistry:
    """
    Registry specialized in handling tools exported via the Model Context Protocol (MCP).
    """

    def __init__(self) -> None:
        """Initialize the MCP Registry."""
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

    def get_tool(self, name: str) -> Dict[str, Any]:
        """
        Retrieve a loaded MCP tool by name.

        Args:
            name (str): The name of the MCP tool.

        Returns:
            Dict[str, Any]: The tool schema, or empty dict if not found.
        """
        return self.mcp_tools.get(name, {})

    def list_tools(self) -> List[str]:
        """
        List all available MCP tools.

        Returns:
            List[str]: A list of tool names.
        """
        return list(self.mcp_tools.keys())
