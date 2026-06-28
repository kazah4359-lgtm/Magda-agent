import logging
from typing import List, Protocol, Any, Callable, Dict, Optional
from magda_agent.architecture.context_hooks import HookRegistry

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

    def before_write(self, context: Any, user_id: int) -> Any:
        """Called before context is written. Can modify the context."""
        ...

    def after_write(self, context: Any, user_id: int) -> None:
        """Called after context is written."""
        ...

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """Called when the overall context is updated."""
        ...


class ContextEngine:
    """
    ContextEngine manages context dynamically using a plugin architecture
    with lifecycle hooks.
    """
    def __init__(self, plugins: Optional[List[ContextPlugin]] = None, llm: Optional[Any] = None) -> None:
        self._plugins: List[ContextPlugin] = []
        self.llm = llm
        self.hook_registry = HookRegistry()

        if plugins:
            for plugin in plugins:
                self.register_plugin(plugin)

    def register_plugin(self, plugin: ContextPlugin) -> None:
        """Registers a new plugin with the context engine."""
        self._plugins.append(plugin)

        if hasattr(plugin, 'before_retrieval'):
            self.hook_registry.register_hook('before_retrieval', plugin.before_retrieval)
        if hasattr(plugin, 'after_retrieval'):
            self.hook_registry.register_hook('after_retrieval', plugin.after_retrieval)
        if hasattr(plugin, 'on_context_update'):
            self.hook_registry.register_hook('on_context_update', plugin.on_context_update)
        if hasattr(plugin, 'before_write'):
            self.hook_registry.register_hook('before_write', plugin.before_write)
        if hasattr(plugin, 'after_write'):
            self.hook_registry.register_hook('after_write', plugin.after_write)

        logging.debug(f"Registered plugin: {plugin.__class__.__name__}")

    def add_plugin(self, plugin: ContextPlugin) -> None:
        """Alias for register_plugin for compatibility."""
        self.register_plugin(plugin)

    async def bootstrap_all(self, config: Dict[str, Any]) -> None:
        """Initialize all plugins."""
        config_with_hooks: Dict[str, Any] = dict(config)
        config_with_hooks["hook_registry"] = self.hook_registry
        plugin: ContextPlugin
        for plugin in self._plugins:
            if hasattr(plugin, 'bootstrap'):
                await plugin.bootstrap(config_with_hooks)

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
        """Compact context through all plugins, with a built-in fallback using LLM."""
        current_items = context_items
        for plugin in self._plugins:
            if hasattr(plugin, 'compact'):
                current_items = await plugin.compact(current_items, metadata)

        limit = metadata.get("limit", 10)
        if len(current_items) > limit and self.llm is not None:
            logging.info("Context length exceeds limit after plugins. Using ContextEngine built-in fallback compression.")
            to_summarize = current_items[:2]
            remaining = current_items[2:]

            combined_text = "\n".join([f"- {getattr(e, 'content', str(e))}" for e in to_summarize])
            prompt = f"Please summarize the following short-term memory context into a concise summary while maintaining key facts and semantic links:\n{combined_text}"

            try:
                summary_content = await self.llm.chat_completion([
                    {"role": "system", "content": "You compress memory context. Return only the summary text."},
                    {"role": "user", "content": prompt}
                ], temperature=0.3)

                # Import MemoryEntry to create a valid compressed item, avoiding circular imports at top level
                from magda_agent.memory.working import MemoryEntry

                first = to_summarize[0] if to_summarize else None
                avg_importance = sum(getattr(e, 'importance', 0.5) for e in to_summarize) / max(1, len(to_summarize))

                summary_entry = MemoryEntry(
                    content=summary_content.strip(),
                    importance=avg_importance,
                    emotional_state=getattr(first, 'emotional_state', None) if first else None,
                    tags=list(set(t for e in to_summarize if isinstance(getattr(e, 'tags', []), list) for t in getattr(e, 'tags', []))),
                    user_id=getattr(first, 'user_id', None) if first else None
                )
                current_items = [summary_entry] + remaining
            except Exception as e:
                logging.error(f"ContextEngine fallback compaction failed: {e}")
                # Fallback to dropping oldest item
                current_items = current_items[1:]
        elif len(current_items) > limit:
            logging.warning("No LLM available for fallback compaction, dropping oldest item.")
            current_items = current_items[1:]

        return current_items

    def retrieve_context(self, query: str, user_id: int, base_retrieval_func: Callable[[str, int], List[Any]]) -> List[Any]:
        """
        Retrieves context by executing lifecycle hooks before and after
        calling the base retrieval function.
        """
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

# Context Engine lifecycle complete
