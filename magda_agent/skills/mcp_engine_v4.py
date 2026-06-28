import logging
import asyncio
from typing import Dict, Any, List, Optional
from magda_agent.skills.mcp_client import MCPClient
from magda_agent.skills.registry import SkillRegistry
from magda_agent.memory.context_engine import ContextPlugin

class MCPEngineV4(ContextPlugin):
    """
    Engine to seamlessly import and execute external MCP tools, converting them
    into Magda's native procedural skills dynamically, while strictly following
    the ContextPlugin protocol.
    """
    def __init__(self, registry: SkillRegistry, mcp_client: MCPClient) -> None:
        """
        Initializes the MCPEngineV4.

        Args:
            registry (SkillRegistry): Magda's native skill registry.
            mcp_client (MCPClient): The MCP client used for remote tool execution.
        """
        self.registry = registry
        self.mcp_client = mcp_client
        self.hook_registry: Optional[Any] = None

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        self.hook_registry = config.get("hook_registry")
        logging.info("MCPEngineV4 bootstrapped.")

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Process incoming content before it is stored or used."""
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble the context string from retrieved items for the LLM."""
        return "\n".join([str(item) for item in context_items])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Compact or summarize the context when limits are reached."""
        return context_items

    def before_retrieval(self, query: str, user_id: int) -> str:
        """Called before context is retrieved. Can modify the query."""
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """Called after context is retrieved. Can modify the retrieved context."""
        return context

    def before_write(self, context: Any, user_id: int) -> Any:
        """Called before context is written. Can modify the context."""
        return context

    def after_write(self, context: Any, user_id: int) -> None:
        """Called after context is written."""
        pass

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """Called when the overall context is updated."""
        pass

    def import_mcp_tool(self, tool_def: Dict[str, Any], connection_info: Dict[str, Any]) -> None:
        """
        Reads MCP standard tool definitions and wraps them into Magda's SkillRegistry.

        Args:
            tool_def (Dict[str, Any]): Definition containing at least "name" and "description".
            connection_info (Dict[str, Any]): Information needed to execute the remote tool.
        """
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("MCP tool definition must include a 'name'.")

        description = tool_def.get("description", "Imported MCP tool.")
        input_schema = tool_def.get("inputSchema", {})

        self.mcp_client.register_remote_tool(tool_name, connection_info)

        async def mcp_wrapper_skill(**kwargs: Any) -> Any:
            """
            Dynamically executes the imported MCP tool via the MCPClient,
            triggering context engine lifecycle hooks before and after tool usage.

            Args:
                kwargs: Arguments to pass to the MCP tool.

            Returns:
                Any: Result of the MCP tool execution.
            """
            # Trigger before tool usage hook
            if self.hook_registry and hasattr(self.hook_registry, 'trigger_broadcast_async'):
                try:
                    await self.hook_registry.trigger_broadcast_async("before_tool_use", tool_name, kwargs)
                except Exception as e:
                    logging.warning(f"Error triggering before_tool_use hook: {e}")

            result = await self.mcp_client.execute_tool(tool_name, **kwargs)

            # Trigger after tool usage hook
            if self.hook_registry and hasattr(self.hook_registry, 'trigger_broadcast_async'):
                try:
                    await self.hook_registry.trigger_broadcast_async("after_tool_use", tool_name, result)
                except Exception as e:
                    logging.warning(f"Error triggering after_tool_use hook: {e}")

            return result

        setattr(mcp_wrapper_skill, "__mcp_schema__", input_schema)
        setattr(mcp_wrapper_skill, "__name__", tool_name)

        self.registry.register_skill(
            name=tool_name,
            func=mcp_wrapper_skill,
            description=description
        )

        logging.info(f"Dynamically wrapped MCP tool '{tool_name}' into Magda skill registry.")
