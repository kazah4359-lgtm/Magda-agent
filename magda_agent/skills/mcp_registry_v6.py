import logging
from typing import Dict, Any, List, Protocol, runtime_checkable

@runtime_checkable
class MCPAdapter(Protocol):
    """
    Protocol for dynamically providing external MCP tools to the registry.
    """
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Retrieves a list of MCP tool schemas from an external source.

        Returns:
            List[Dict[str, Any]]: A list of tool schemas.
        """
        ...

class MCPRegistryV6:
    """
    Registry specialized in handling tools exported via the Model Context Protocol (MCP) version 6.
    Allows for dynamic synchronization from registered adapters.
    """

    def __init__(self) -> None:
        """Initialize the MCP Registry V6."""
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}
        self.adapters: List[MCPAdapter] = []

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
        Also validates that 'inputSchema' is a dictionary if provided.

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

        # Validate inputSchema if it exists
        if "inputSchema" in schema and not isinstance(schema["inputSchema"], dict):
            return False

        return True

    def get_tool(self, name: str) -> Dict[str, Any]:
        """
        Retrieve a loaded MCP tool by name.

        Args:
            name (str): The name of the MCP tool.

        Returns:
            Dict[str, Any]: The tool schema, or an empty dictionary if not found.
        """
        return self.mcp_tools.get(name, {})

    def list_tools(self) -> List[str]:
        """
        List all available MCP tools.

        Returns:
            List[str]: A list of tool names.
        """
        return list(self.mcp_tools.keys())

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

    def register_adapter(self, adapter: MCPAdapter) -> None:
        """
        Registers an adapter to dynamically fetch external tools.

        Args:
            adapter (MCPAdapter): The adapter instance providing tools.
        """
        self.adapters.append(adapter)
        logging.info(f"Registered MCP adapter: {adapter}")

    def sync_from_adapters(self) -> int:
        """
        Synchronizes tools from all registered adapters.

        Returns:
            int: The total number of tools successfully synced and loaded.
        """
        loaded_count = 0
        for adapter in self.adapters:
            try:
                tools = adapter.get_tools()
                for tool in tools:
                    if self.load_tool(tool):
                        loaded_count += 1
            except Exception as e:
                logging.error(f"Error syncing from adapter {adapter}: {e}")

        logging.info(f"Synchronized {loaded_count} tools from adapters.")
        return loaded_count
