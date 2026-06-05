import logging
from typing import Any, Dict, List, Optional
from magda_agent.context.plugin import ContextPlugin

class DefaultContextPlugin(ContextPlugin):
    """
    Default plugin implementing Magda's current context behavior.
    """
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Standard context assembly from memory entries."""
        return "\n".join([f"- {item.content}" for item in context_items if hasattr(item, 'content')])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        Implements summarization logic when compacting.
        Inspired by the existing MemorySystem/WorkingMemory logic.
        """
        # If the number of items is within limit, don't compact
        limit = metadata.get("limit", 10)
        if len(context_items) <= limit:
            return context_items

        if not self.llm:
            logging.warning("No LLM available for compaction, dropping oldest item.")
            return context_items[1:]

        # Compact by summarizing the two oldest items
        to_summarize = context_items[:2]
        remaining = context_items[2:]

        combined_text = "\n".join([f"- {e.content}" for e in to_summarize if hasattr(e, 'content')])
        prompt = f"Please summarize the following short-term memory context into a single concise bullet point:\n{combined_text}"

        try:
            summary_content = await self.llm.chat_completion([
                {"role": "system", "content": "You compress memory context. Return only the summary text."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)

            # Import here to avoid circular imports if any
            from magda_agent.memory.working import MemoryEntry

            # Use attributes from first entry
            first = to_summarize[0]
            avg_importance = sum(e.importance for e in to_summarize if hasattr(e, 'importance')) / len(to_summarize)

            summary_entry = MemoryEntry(
                content=summary_content.strip(),
                importance=avg_importance,
                emotional_state=getattr(first, 'emotional_state', None),
                tags=getattr(first, 'tags', []),
                user_id=getattr(first, 'user_id', None)
            )
            return [summary_entry] + remaining
        except Exception as e:
            logging.error(f"DefaultContextPlugin compaction failed: {e}")
            return context_items[1:] # Fallback to dropping
