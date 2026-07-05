import logging
from typing import List, Protocol, Any, Dict, Optional, Callable
from magda_agent.architecture.context_hooks_v5 import HookRegistry

class ContextPluginV1(Protocol):
    """Protocol defining the lifecycle hooks for a Context Engine V1 plugin."""

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

    def before_write(self, context: Any, user_id: int) -> Any:
        """Called before context is written. Can modify the context."""
        ...

    def after_write(self, context: Any, user_id: int) -> None:
        """Called after context is written."""
        ...

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """Called when the overall context is updated."""
        ...

class ContextEngineV1:
    """
    ContextEngineV1 manages context dynamically using a plugin architecture
    with lifecycle hooks. Uses HookRegistry.
    """
    def __init__(self, plugins: Optional[List[ContextPluginV1]] = None, llm: Optional[Any] = None) -> None:
        self._plugins: List[ContextPluginV1] = []
        self.llm = llm
        self.hook_registry = HookRegistry()

        if plugins:
            for plugin in plugins:
                self.register_plugin(plugin)

    def register_plugin(self, plugin: ContextPluginV1) -> None:
        """Registers a new plugin with the context engine."""
        self._plugins.append(plugin)

        # Sync hooks
        for hook_name in ['before_retrieval', 'after_retrieval', 'on_context_update', 'before_write', 'after_write']:
            if hasattr(plugin, hook_name):
                self.hook_registry.register_hook(hook_name, getattr(plugin, hook_name))

        # Async and common lifecycle hooks
        for hook_name in ['bootstrap', 'ingest', 'assemble', 'compact']:
            if hasattr(plugin, hook_name):
                self.hook_registry.register_hook(hook_name, getattr(plugin, hook_name))

        logging.debug(f"Registered plugin: {plugin.__class__.__name__}")

    def add_plugin(self, plugin: ContextPluginV1) -> None:
        """Alias for register_plugin for compatibility."""
        self.register_plugin(plugin)

    async def bootstrap_all(self, config: Dict[str, Any]) -> None:
        """Initialize all plugins using the hook registry."""
        config_with_hooks: Dict[str, Any] = dict(config)
        config_with_hooks["hook_registry"] = self.hook_registry
        await self.hook_registry.trigger_broadcast_async('bootstrap', config_with_hooks)

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Run content through ingest hook of all plugins using the hook registry."""
        return await self.hook_registry.trigger_hook_async('ingest', content, metadata)

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble context using all plugins via the hook registry."""
        if 'assemble' not in self.hook_registry._hooks or not self.hook_registry._hooks['assemble']:
            return "\n".join([str(item) for item in context_items])
        return await self.hook_registry.trigger_broadcast_async('assemble', context_items, metadata)

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Compact context through all plugins using the hook registry."""
        current_items = await self.hook_registry.trigger_hook_async('compact', context_items, metadata)

        limit = metadata.get("limit", 10)
        if current_items and len(current_items) > limit and self.llm is not None:
            logging.info("Context length exceeds limit after plugins. Using ContextEngine built-in fallback compression.")
            to_summarize = current_items[:2]
            remaining = current_items[2:]

            combined_text = "\n".join([f"- {getattr(e, 'content', str(e))}" for e in to_summarize])
            prompt = f"Please summarize the following context:\n{combined_text}"

            try:
                summary_content = await self.llm.chat_completion([
                    {"role": "system", "content": "You compress memory context. Return only the summary text."},
                    {"role": "user", "content": prompt}
                ], temperature=0.3)

                # Mock memory entry to represent summarized data
                class MinimalMemoryEntry:
                    def __init__(self, content: str):
                        self.content = content

                summary_entry = MinimalMemoryEntry(content=summary_content.strip())
                current_items = [summary_entry] + remaining
            except Exception as e:
                logging.error(f"ContextEngine fallback compaction failed: {e}")
                current_items = current_items[1:]
        elif current_items and len(current_items) > limit:
            logging.warning("No LLM available for fallback compaction, dropping oldest item.")
            current_items = current_items[1:]

        return current_items if current_items is not None else []

    def retrieve_context(self, query: str, user_id: int, base_retrieval_func: Callable[[str, int], List[Any]]) -> List[Any]:
        """Retrieves context by executing lifecycle hooks before and after."""
        current_query: str = self.hook_registry.trigger_hook('before_retrieval', query, user_id)
        if current_query is None:
            current_query = query

        context: List[Any] = base_retrieval_func(current_query, user_id)

        updated_context = self.hook_registry.trigger_hook('after_retrieval', context, current_query, user_id)
        if updated_context is None:
            updated_context = context

        return updated_context

    def update_context(self, new_context: Any, user_id: int) -> None:
        """Triggers the on_context_update hook for all registered plugins."""
        self.hook_registry.trigger_hook('on_context_update', new_context, user_id)

    def write_context(self, context: Any, user_id: int) -> None:
        """Writes context by executing lifecycle hooks before and after."""
        current_context: Any = self.hook_registry.trigger_hook('before_write', context, user_id)
        if current_context is None:
            current_context = context

        self.update_context(current_context, user_id)
        self.hook_registry.trigger_hook('after_write', current_context, user_id)
