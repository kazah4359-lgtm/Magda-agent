import logging
from typing import Dict, Any
from magda_agent.skills.mcp_client import MCPClient
from magda_agent.skills.registry import SkillRegistry

class MCPEngine:
    """
    Engine to seamlessly import and execute external MCP tools, converting them
    into Magda's native procedural skills dynamically.
    """
    def __init__(self, registry: SkillRegistry, mcp_client: MCPClient) -> None:
        """
        Initializes the MCPEngine.

        Args:
            registry (SkillRegistry): Magda's native skill registry.
            mcp_client (MCPClient): The MCP client used for remote tool execution.
        """
        self.registry = registry
        self.mcp_client = mcp_client

    def import_mcp_tool(self, tool_def: Dict[str, Any], connection_info: Dict[str, Any], server_name: str = None) -> None:
        """
        Reads MCP standard tool definitions and wraps them into Magda's SkillRegistry.

        Args:
            tool_def (Dict[str, Any]): Definition containing at least "name" and "description".
                                       Optionally contains "inputSchema".
            connection_info (Dict[str, Any]): Information needed to execute the remote tool.
            server_name (str, optional): An optional server name to prefix the tool name to avoid conflicts.
        """
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("MCP tool definition must include a 'name'.")

        description = tool_def.get("description", "Imported MCP tool.")
        input_schema = tool_def.get("inputSchema", {})

        if server_name:
            # 1a. Register the remote server routing with the MCPClient
            tool_name = f"{server_name}__{tool_name}"
            self.mcp_client.register_mcp_server(server_name, connection_info)
        else:
            # 1b. Register the remote tool routing directly with the MCPClient
            self.mcp_client.register_remote_tool(tool_name, connection_info)

        import asyncio
        import concurrent.futures

        # 2. Create the wrapper skill that delegates to MCPClient
        async def mcp_wrapper_skill(**kwargs: Any) -> Any:
            """
            Dynamically executes the imported MCP tool via the MCPClient.
            Runs asynchronously to support concurrent execution of tools.

            Args:
                kwargs: Arguments to pass to the MCP tool.

            Returns:
                Any: Result of the MCP tool execution.
            """
            return await self.mcp_client.execute_tool(tool_name, **kwargs)

        # Add the schema attribute so MagdaMCPAdapter natively picks it up if needed.
        setattr(mcp_wrapper_skill, "__mcp_schema__", input_schema)
        # Preserve original name in __name__ for generic introspection
        setattr(mcp_wrapper_skill, "__name__", tool_name)

        # 3. Register natively in Magda's SkillRegistry
        self.registry.register_skill(
            name=tool_name,
            func=mcp_wrapper_skill,
            description=description
        )

        logging.info(f"Dynamically wrapped MCP tool '{tool_name}' into Magda skill registry.")
