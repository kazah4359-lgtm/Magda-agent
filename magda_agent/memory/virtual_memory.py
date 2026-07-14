import logging
import math
from typing import List, Dict, Any, Optional
from magda_agent.memory.context_engine import ContextPlugin
from magda_agent.memory.working import MemoryEntry
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.emotions.engine import PADState

class VirtualMemoryPlugin(ContextPlugin):
    """
    OpenClaw Virtual Memory Pattern Integration.
    An explicit swapping mechanism to transition large context windows (WorkingMemory)
    into structured memory layers (EpisodicMemory) to prevent token limit/item count exhaustion.
    """

    def __init__(self, episodic_memory: Optional[EpisodicMemory] = None, token_limit: int = 2000) -> None:
        """
        Initializes the VirtualMemoryPlugin.

        Args:
            episodic_memory: Optional EpisodicMemory instance. If not provided, it will be initialized.
            token_limit: Heuristic token limit before swapping.
        """
        self.episodic_memory = episodic_memory or EpisodicMemory(persist_directory=":memory:")
        self.token_limit = token_limit
        self.config: Dict[str, Any] = {}
        logging.debug("Initialized VirtualMemoryPlugin")

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """
        Bootstrap lifecycle hook. Initializes plugin state from config.

        Args:
            config: Configuration dictionary injected by ContextEngine.
        """
        self.config = config
        self.token_limit = config.get("token_limit", self.token_limit)
        logging.info(f"VirtualMemoryPlugin bootstrapped with token_limit={self.token_limit}")

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Ingest lifecycle hook. Process raw content before storing.

        Args:
            content: Raw string content.
            metadata: Metadata associated with the content.

        Returns:
            Processed content.
        """
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """
        Assemble lifecycle hook. Constructs context prompt.

        Args:
            context_items: List of retrieved context items.
            metadata: Metadata controlling assembly format.

        Returns:
            Assembled context string.
        """
        if not context_items:
            return ""
        items_str = "\n".join([f"- {getattr(item, 'content', str(item))}" for item in context_items])
        return f"VIRTUAL CONTEXT:\n{items_str}"

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        Compact lifecycle hook. Swaps out excess context items to EpisodicMemory to stay within token limits.

        Args:
            context_items: List of MemoryEntry items in working memory.
            metadata: Metadata controlling compaction rules (like 'limit', 'user_id').

        Returns:
            The remaining working memory items after swapping.
        """
        user_id = metadata.get("user_id")
        limit = metadata.get("limit", 10)

        # First, check item count limit
        if len(context_items) > limit:
            excess_count = len(context_items) - limit
            context_items = await self.swap_out(context_items, excess_count, user_id)

        # Second, check token heuristic limit
        while self._estimate_tokens(context_items) > self.token_limit and len(context_items) > 1:
            context_items = await self.swap_out(context_items, 1, user_id)

        return context_items

    def before_retrieval(self, query: str, user_id: int) -> str:
        """
        Hook called before retrieval.

        Args:
            query: Searching query.
            user_id: User ID.

        Returns:
            Query string.
        """
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """
        Hook called after retrieval.

        Args:
            context: Retrieved context elements.
            query: Searching query.
            user_id: User ID.

        Returns:
            Retrieved context elements.
        """
        return context

    def before_write(self, context: Any, user_id: int) -> Any:
        """
        Hook called before writing context.

        Args:
            context: Context item.
            user_id: User ID.

        Returns:
            Context item.
        """
        return context

    def after_write(self, context: Any, user_id: int) -> None:
        """
        Hook called after writing context.

        Args:
            context: Context item.
            user_id: User ID.
        """
        pass

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """
        Hook called on context updates.

        Args:
            new_context: New context state.
            user_id: User ID.
        """
        pass

    # --- Custom Swap Methods ---

    async def swap_out(self, context_items: List[Any], count: int, user_id: Optional[int] = None) -> List[Any]:
        """
        Transition/swap a given count of oldest context items to EpisodicMemory.

        Args:
            context_items: Working memory context items.
            count: Number of oldest items to swap out.
            user_id: Optional user ID filter.

        Returns:
            Remaining context items after swapping.
        """
        if not context_items or count <= 0:
            return context_items

        to_swap = context_items[:count]
        remaining = context_items[count:]

        for item in to_swap:
            content = getattr(item, "content", str(item))
            importance = getattr(item, "importance", 0.5)
            emotional_state = getattr(item, "emotional_state", None)

            metadata: Dict[str, Any] = {
                "swapped_from_virtual": True,
                "importance": importance,
            }

            if emotional_state and isinstance(emotional_state, PADState):
                metadata.update({
                    "pad_p": emotional_state.pleasure,
                    "pad_a": emotional_state.arousal,
                    "pad_d": emotional_state.dominance
                })

            self.episodic_memory.store_event(
                text=content,
                metadata=metadata,
                user_id=user_id
            )

        logging.info(f"Swapped out {len(to_swap)} oldest items to episodic memory.")
        return remaining

    async def swap_in(self, context_items: List[Any], query: str, top_k: int = 3, user_id: Optional[int] = None) -> List[Any]:
        """
        Retrieve relevant past swapped memories from EpisodicMemory and load them back into WorkingMemory.

        Args:
            context_items: Current working memory context items.
            query: Semantic search query for retrieval.
            top_k: Number of items to retrieve.
            user_id: Optional user ID filter.

        Returns:
            Updated context list containing swapped-in items.
        """
        events = self.episodic_memory.recall_events(query=query, top_k=top_k, user_id=user_id)
        swapped_in_entries = []
        for event in events:
            # Avoid duplicating content already in context_items
            if any(getattr(item, "content", "") == event for item in context_items):
                continue
            entry = MemoryEntry(
                content=event,
                importance=0.5,
                emotional_state=PADState(0.0, 0.0, 0.0),
                user_id=user_id
            )
            swapped_in_entries.append(entry)

        logging.info(f"Swapped in {len(swapped_in_entries)} relevant items.")
        return context_items + swapped_in_entries

    def _estimate_tokens(self, entries: List[Any]) -> int:
        """
        Heuristic method to calculate token counts of entries.

        Args:
            entries: List of memory entries.

        Returns:
            Estimated token count.
        """
        total_words = sum(len(getattr(e, "content", str(e)).split()) for e in entries)
        return int(total_words * 1.3)
