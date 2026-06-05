import logging
from typing import Any, Dict, List, Optional
from magda_agent.context.plugin import ContextPlugin

class ContextEngine:
    """
    Orchestrates the lifecycle of context management via plugins.
    """
    def __init__(self, plugins: Optional[List[ContextPlugin]] = None):
        self.plugins = plugins or []

    async def bootstrap_all(self, config: Dict[str, Any]) -> None:
        for plugin in self.plugins:
            await plugin.bootstrap(config)

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        current_content = content
        for plugin in self.plugins:
            current_content = await plugin.ingest(current_content, metadata)
        return current_content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        # For assemble, we might want one plugin to handle it or a chain.
        # OpenClaw usually has one main assembly. Here we'll chain them.
        assembled_context = ""
        for plugin in self.plugins:
            # Each plugin can augment or replace the context string
            assembled_context = await plugin.assemble(context_items, metadata)
        return assembled_context

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        current_items = context_items
        for plugin in self.plugins:
            current_items = await plugin.compact(current_items, metadata)
        return current_items

    def add_plugin(self, plugin: ContextPlugin):
        self.plugins.append(plugin)
