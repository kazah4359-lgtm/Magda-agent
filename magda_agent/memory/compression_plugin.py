import logging
from typing import Any, Dict, List, Optional
from magda_agent.memory.context_engine import ContextPlugin

class CompressionPlugin(ContextPlugin):
    """
    Plugin for Context Engine that compresses older context items
    when limits are reached.
    """
    def __init__(self, llm: Optional[Any] = None) -> None:
        """
        Initializes the CompressionPlugin.

        Args:
            llm: Optional language model integration for intelligent compaction.
        """
        self.llm = llm

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Process incoming content before it is stored or used."""
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble the context string from retrieved items for the LLM."""
        return "\n".join([f"- {getattr(item, 'content', str(item))}" for item in context_items])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Compact or summarize the context when limits are reached."""
        limit = metadata.get("limit", 10)
        if len(context_items) <= limit:
            return context_items

        if not self.llm:
            logging.warning("No LLM available for compaction, dropping oldest item.")
            return context_items[1:]

        to_summarize = context_items[:2]
        remaining = context_items[2:]

        combined_text = "\n".join([f"- {getattr(e, 'content', str(e))}" for e in to_summarize])
        prompt = f"Please summarize the following short-term memory context into a single concise bullet point:\n{combined_text}"

        try:
            summary_content = await self.llm.chat_completion([
                {"role": "system", "content": "You compress memory context. Return only the summary text."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)

            from magda_agent.memory.working import MemoryEntry

            first = to_summarize[0] if to_summarize else None
            avg_importance = sum(getattr(e, 'importance', 0.5) for e in to_summarize) / max(1, len(to_summarize))

            tags = []
            for e in to_summarize:
                e_tags = getattr(e, 'tags', [])
                if isinstance(e_tags, list):
                    tags.extend(e_tags)
            tags = list(set(tags))

            summary_entry = MemoryEntry(
                content=summary_content.strip(),
                importance=avg_importance,
                emotional_state=getattr(first, 'emotional_state', None) if first else None,
                tags=tags,
                user_id=getattr(first, 'user_id', None) if first else None
            )

            return [summary_entry] + remaining

        except Exception as e:
            logging.error(f"CompressionPlugin compaction failed: {e}")
            return context_items[1:]

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
