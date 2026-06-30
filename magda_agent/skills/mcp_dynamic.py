import logging
from typing import Dict, Any

from magda_agent.skills.mcp_registry import MCPRegistry

class MCPDynamicRegistrar:
    """
    Endpoint for dynamically registering new MCP tools at runtime.
    """

    def __init__(self, registry: MCPRegistry) -> None:
        """
        Initialize the dynamic registrar.

        Args:
            registry (MCPRegistry): The existing MCP tool registry instance.
        """
        self.registry = registry

    def register_tool_at_runtime(self, tool_schema: Dict[str, Any]) -> bool:
        """
        Dynamically registers a new MCP tool at runtime using the provided schema.

        Args:
            tool_schema (Dict[str, Any]): The MCP tool schema dictionary to register.

        Returns:
            bool: True if the tool was registered successfully, False otherwise.
        """
        logging.info("Attempting to dynamically register MCP tool at runtime.")

        # Load the tool into the registry
        success = self.registry.load_tool(tool_schema)

        if success:
            tool_name = tool_schema.get("name", "Unknown")
            logging.info(f"Dynamically registered tool: {tool_name}")
        else:
            logging.error("Failed to dynamically register tool.")

        return success
