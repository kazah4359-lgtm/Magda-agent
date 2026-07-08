import logging
from typing import List, Any, Dict

class EvictionPolicy:
    """
    EvictionPolicy is a ContextPlugin that manages gracefully discarding
    low-priority items from working memory when approaching size limits.
    It prioritizes items with higher 'importance' values and falls back
    to discarding the oldest items.
    """

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.
        """
        pass

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Process incoming content before it is stored or used.
        """
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """
        Assemble the context string from retrieved items for the LLM.
        """
        return "\n".join([str(item) for item in context_items])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        Compact or summarize the context when limits are reached by dropping
        the lowest importance items first.

        Args:
            context_items: List of memory entries to filter.
            metadata: Contains 'limit' denoting max number of items allowed.

        Returns:
            A list of the remaining context items.
        """
        limit = metadata.get("limit", 10)

        if len(context_items) <= limit:
            return context_items

        logging.info(f"EvictionPolicy checking {len(context_items)} items against limit {limit}.")

        # Sort the items based on importance.
        # Ensure older items are dropped if importance is the same.
        # We assume context_items is a chronological list (oldest first).

        # Enumerate to keep original index for tie-breaking (drop oldest first means keep newer).
        # We want to KEEP items with highest importance.
        # If importance is equal, keep the newer ones (higher index).

        def sort_key(enumerated_item: tuple[int, Any]) -> tuple[float, int]:
            index, item = enumerated_item
            importance = getattr(item, 'importance', 0.5)
            # Default importance if not specified is 0.5
            return (importance, index)

        sorted_items = sorted(enumerate(context_items), key=sort_key, reverse=True)

        # Keep the top `limit` items
        kept_items_with_index = sorted_items[:limit]

        # Restore chronological order
        kept_items_with_index.sort(key=lambda x: x[0])

        remaining_items = [item for index, item in kept_items_with_index]
        return remaining_items

    def before_retrieval(self, query: str, user_id: int) -> str:
        """Called before context is retrieved."""
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """Called after context is retrieved."""
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
