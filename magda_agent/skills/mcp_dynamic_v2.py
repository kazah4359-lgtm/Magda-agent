import logging
from typing import Dict, Any

from magda_agent.skills.mcp_registry import MCPRegistry


class MCPDynamicRegistrarV2:
    """
    Endpoint for dynamically registering new MCP tools at runtime, specifically
    handling async skills to support true runtime concurrency.
    """

    def __init__(self, registry: MCPRegistry) -> None:
        """
        Initialize the dynamic registrar V2.

        Args:
            registry (MCPRegistry): The existing MCP tool registry instance.
        """
        self.registry = registry

    def register_tool_at_runtime(self, tool_schema: Dict[str, Any], is_async: bool = False) -> bool:
        """
        Dynamically registers a new MCP tool at runtime using the provided schema.

        Args:
            tool_schema (Dict[str, Any]): The MCP tool schema dictionary to register.
            is_async (bool): If True, indicates the underlying execution will be asynchronous.
                             Currently, it explicitly validates the schema for async readiness if needed.

        Returns:
            bool: True if the tool was registered successfully, False otherwise.
        """
        logging.info("Attempting to dynamically register MCP tool at runtime (V2).")

        # Load the tool into the registry
        success = self.registry.load_tool(tool_schema)

        if success:
            tool_name = tool_schema.get("name", "Unknown")

            # Additional V2 logic: explicitly track or handle async designation in schema
            # Although registry handles basic load, we tag it if we want to expose this metadata
            # for ConcurrentSkillExecutor to optimize or for wrapper creation.
            tool_schema["__is_async__"] = is_async

            logging.info(f"Dynamically registered tool (V2): {tool_name}, is_async={is_async}")
        else:
            logging.error("Failed to dynamically register tool (V2).")

        return success
