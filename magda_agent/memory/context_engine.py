import logging
from typing import List, Protocol, Any, Callable, Dict, Optional

class ContextPlugin(Protocol):
    """Protocol defining the lifecycle hooks for a Context Engine plugin."""
    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        ...

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Process incoming content before it is stored or used."""
        ...

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble the context string from retrieved items for the LLM."""
        ...

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Compact or summarize the context when limits are reached."""
        ...

    def before_retrieval(self, query: str, user_id: int) -> str:
        """Called before context is retrieved. Can modify the query."""
        ...

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """Called after context is retrieved. Can modify the retrieved context."""
        ...

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """Called when the overall context is updated."""
        ...


class ContextEngine:
    """
    ContextEngine manages context dynamically using a plugin architecture
    with lifecycle hooks.
    """
    def __init__(self, plugins: Optional[List[ContextPlugin]] = None) -> None:
        self._plugins: List[ContextPlugin] = plugins or []

    def register_plugin(self, plugin: ContextPlugin) -> None:
        """Registers a new plugin with the context engine."""
        self._plugins.append(plugin)
        logging.debug(f"Registered plugin: {plugin.__class__.__name__}")

    def add_plugin(self, plugin: ContextPlugin) -> None:
        """Alias for register_plugin for compatibility."""
        self.register_plugin(plugin)

    async def bootstrap_all(self, config: Dict[str, Any]) -> None:
        """Initialize all plugins."""
        for plugin in self._plugins:
            if hasattr(plugin, 'bootstrap'):
                await plugin.bootstrap(config)

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Run content through ingest hook of all plugins."""
        current_content = content
        for plugin in self._plugins:
            if hasattr(plugin, 'ingest'):
                current_content = await plugin.ingest(current_content, metadata)
        return current_content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble context using all plugins."""
        assembled_context = ""
        # If no plugins, default behavior
        if not self._plugins:
            return "\n".join([str(item) for item in context_items])

        for plugin in self._plugins:
            if hasattr(plugin, 'assemble'):
                assembled_context = await plugin.assemble(context_items, metadata)
        return assembled_context

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Compact context through all plugins."""
        current_items = context_items
        for plugin in self._plugins:
            if hasattr(plugin, 'compact'):
                current_items = await plugin.compact(current_items, metadata)
        return current_items

    def retrieve_context(self, query: str, user_id: int, base_retrieval_func: Callable[[str, int], List[Any]]) -> List[Any]:
        """
        Retrieves context by executing lifecycle hooks before and after
        calling the base retrieval function.
        """
        current_query: str = query
        for plugin in self._plugins:
            if hasattr(plugin, 'before_retrieval'):
                current_query = plugin.before_retrieval(current_query, user_id)

        context: List[Any] = base_retrieval_func(current_query, user_id)

        for plugin in self._plugins:
            if hasattr(plugin, 'after_retrieval'):
                context = plugin.after_retrieval(context, current_query, user_id)

        return context

    def update_context(self, new_context: Any, user_id: int) -> None:
        """Triggers the on_context_update hook for all registered plugins."""
        for plugin in self._plugins:
            if hasattr(plugin, 'on_context_update'):
                plugin.on_context_update(new_context, user_id)

# Context Engine lifecycle complete
