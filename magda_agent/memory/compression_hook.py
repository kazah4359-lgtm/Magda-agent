import logging
from typing import List, Any, Dict, Optional
from magda_agent.memory.context_engine import ContextPlugin

class CompressionHookPlugin(ContextPlugin):
    """
    A ContextPlugin that automatically compresses short-term memory during
    the before_retrieval phase to maintain a compact context window,
    inspired by OpenClaw trends.
    """

    def __init__(self, memory_system: Optional[Any] = None) -> None:
        """
        Initialize the CompressionHookPlugin.

        Args:
            memory_system: Optional reference to the overarching MemorySystem
                           which gives access to the short-term working memory.
        """
        self.memory_system = memory_system
        logging.info("CompressionHookPlugin initialized.")

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.
        """
        if "memory_system" in config:
            self.memory_system = config["memory_system"]

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
        """
        Automatically compresses short-term memory before context retrieval.
        It uses the referenced MemorySystem to enforce compact limits in-place.
        """
        logging.debug("CompressionHookPlugin: before_retrieval executing memory compaction.")

        if self.memory_system and hasattr(self.memory_system, 'working_memory'):
            working_memory = self.memory_system.working_memory

            if hasattr(working_memory, 'get_entries') and hasattr(working_memory, 'remove'):
                entries = working_memory.get_entries(user_id)
                # Ensure short term memory does not exceed a certain length to maintain compact context.
                # Assuming standard working memory limit is around 10, we proactively compress if near limit.
                limit = getattr(working_memory, 'limit', 10)

                # If memory is at or above limit, trigger dropping of oldest or least important
                if len(entries) >= limit:
                    # Heuristically, we sort by importance, but if we just want a strict length:
                    entries_to_remove = len(entries) - limit + 1

                    # Sort entries by importance (lowest first)
                    # We remove the least important entries to compress the active window
                    sorted_by_importance = sorted(
                        entries,
                        key=lambda x: getattr(x, 'importance', 0.0)
                    )

                    for i in range(entries_to_remove):
                        entry_to_remove = sorted_by_importance[i]
                        working_memory.remove(getattr(entry_to_remove, 'id'), user_id)
                        logging.debug(f"CompressionHookPlugin compressed memory entry {getattr(entry_to_remove, 'id')}")

        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """Called after context is retrieved. Can modify the retrieved context."""
        return context

    def before_write(self, context: Any, user_id: int) -> Any:
        """Called before context is written."""
        return context

    def after_write(self, context: Any, user_id: int) -> None:
        """Called after context is written."""
        pass

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """Called when the overall context is updated."""
        pass
