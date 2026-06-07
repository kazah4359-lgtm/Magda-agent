import logging
from typing import List, Protocol, Any, Callable

class ContextPlugin(Protocol):
    """Protocol defining the lifecycle hooks for a Context Engine plugin."""
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
    def __init__(self) -> None:
        self._plugins: List[ContextPlugin] = []

    def register_plugin(self, plugin: ContextPlugin) -> None:
        """Registers a new plugin with the context engine."""
        self._plugins.append(plugin)
        logging.debug(f"Registered plugin: {plugin.__class__.__name__}")

    def retrieve_context(self, query: str, user_id: int, base_retrieval_func: Callable[[str, int], List[Any]]) -> List[Any]:
        """
        Retrieves context by executing lifecycle hooks before and after
        calling the base retrieval function.
        """
        current_query = query
        for plugin in self._plugins:
            current_query = plugin.before_retrieval(current_query, user_id)

        context = base_retrieval_func(current_query, user_id)

        for plugin in self._plugins:
            context = plugin.after_retrieval(context, current_query, user_id)

        return context

    def update_context(self, new_context: Any, user_id: int) -> None:
        """Triggers the on_context_update hook for all registered plugins."""
        for plugin in self._plugins:
            plugin.on_context_update(new_context, user_id)

    async def compact(self, entries: List[Any], context_params: dict) -> List[Any]:
        """Compacts entries when memory limit is reached (used by working memory)."""
        limit = context_params.get("limit", len(entries))
        # Default behavior: remove the oldest entry to satisfy limit
        # Plugins can override this behavior in the future via hooks.
        if len(entries) > limit:
            return entries[1:]
        return entries
