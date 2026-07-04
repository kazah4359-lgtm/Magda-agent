import logging
from typing import Dict, Any

from magda_agent.skills.mcp_registry import MCPRegistry


class MCPDynamicRegistrarV4:
    """
    Endpoint for dynamically registering new MCP tools at runtime via MCP protocol.
    Specifically handles async skills to support true runtime concurrency.
    """

    def __init__(self, registry: MCPRegistry) -> None:
        """
        Initialize the dynamic registrar V4.

        Args:
            registry (MCPRegistry): The existing MCP tool registry instance.
        """
        self.registry = registry


    def _perform_enhanced_validation(self, tool_schema: Dict[str, Any]) -> bool:
        """
        Performs enhanced validation on the MCP tool schema.

        Args:
            tool_schema (Dict[str, Any]): The MCP tool schema to validate.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        # Check if name is valid (e.g., alphanumeric and underscores only)
        if "name" in tool_schema:
            import re
            if not re.match(r"^[a-zA-Z0-9_-]+$", tool_schema["name"]):
                logging.error("Enhanced Validation Failed: Invalid tool name format.")
                return False

        # Check if inputSchema exists and is a dictionary if provided
        if "inputSchema" in tool_schema:
            if not isinstance(tool_schema["inputSchema"], dict):
                logging.error("Enhanced Validation Failed: inputSchema must be a dictionary.")
                return False

        return True

    def register_tool_at_runtime(self, tool_schema: Dict[str, Any], is_async: bool = False) -> bool:
        """
        Dynamically registers a new MCP tool at runtime using the provided schema.

        Args:
            tool_schema (Dict[str, Any]): The MCP tool schema dictionary to register.
            is_async (bool): If True, indicates the underlying execution will be asynchronous.

        Returns:
            bool: True if the tool was registered successfully, False otherwise.
        """

        logging.info("Attempting to dynamically register MCP tool at runtime (V4).")

        # Perform enhanced validation
        if not self._perform_enhanced_validation(tool_schema):
            return False


        # Load the tool into the registry
        success = self.registry.load_tool(tool_schema)

        if success:
            tool_name = tool_schema.get("name", "Unknown")

            # Additional logic: explicitly track or handle async designation in schema
            # for ConcurrentSkillExecutor to optimize or for wrapper creation.
            tool_schema["__is_async__"] = is_async

            logging.info(f"Dynamically registered tool (V4): {tool_name}, is_async={is_async}")
        else:
            logging.error("Failed to dynamically register tool (V4).")

        return success
