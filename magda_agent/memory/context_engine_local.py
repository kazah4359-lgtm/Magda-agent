import logging
from typing import List, Any, Dict, Optional
from magda_agent.memory.context_engine import ContextEngine, ContextPlugin
from magda_agent.memory.working import MemoryEntry

class LocalFirstContextEngine(ContextEngine):
    """
    LocalFirstContextEngine extends ContextEngine by gracefully falling back
    to a local LLM if the primary external LLM fails during context compaction.
    """
    def __init__(self, plugins: Optional[List[ContextPlugin]] = None, llm: Optional[Any] = None, local_llm: Optional[Any] = None) -> None:
        """
        Initialize the LocalFirstContextEngine with optional plugins, primary LLM, and fallback local LLM.
        """
        super().__init__(plugins=plugins, llm=llm)
        self.local_llm = local_llm

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        Compact context through all plugins using the hook registry.
        If limits are exceeded, attempt compression using the primary LLM.
        If the primary LLM fails, fall back to the local LLM.
        If the local LLM fails or is absent, truncate the context.
        """
        current_items = await self.hook_registry.trigger_hook_async('compact', context_items, metadata)

        limit = metadata.get("limit", 10)
        if len(current_items) <= limit:
            return current_items

        # We need compaction
        to_summarize = current_items[:2]
        remaining = current_items[2:]
        combined_text = "\n".join([f"- {getattr(e, 'content', str(e))}" for e in to_summarize])
        prompt = f"Please summarize the following short-term memory context into a concise summary while maintaining key facts and semantic links:\n{combined_text}"

        first = to_summarize[0] if to_summarize else None
        avg_importance = sum(getattr(e, 'importance', 0.5) for e in to_summarize) / max(1, len(to_summarize))
        tags = list(set(t for e in to_summarize if isinstance(getattr(e, 'tags', []), list) for t in getattr(e, 'tags', [])))

        async def _attempt_compression(target_llm: Any) -> Optional[MemoryEntry]:
            if target_llm is None:
                return None
            try:
                summary_content = await target_llm.chat_completion([
                    {"role": "system", "content": "You compress memory context. Return only the summary text."},
                    {"role": "user", "content": prompt}
                ], temperature=0.3)

                return MemoryEntry(
                    content=summary_content.strip(),
                    importance=avg_importance,
                    emotional_state=getattr(first, 'emotional_state', None) if first else None,
                    tags=tags,
                    user_id=getattr(first, 'user_id', None) if first else None
                )
            except Exception as e:
                logging.error(f"Compression attempt failed: {e}")
                return None

        logging.info("Context length exceeds limit. Using LocalFirstContextEngine fallback compression.")

        # Try primary LLM
        summary_entry = await _attempt_compression(self.llm)

        # Try local LLM if primary failed
        if summary_entry is None and self.local_llm is not None:
            logging.info("Primary LLM failed or absent. Falling back to local LLM.")
            summary_entry = await _attempt_compression(self.local_llm)

        # Truncate if both failed
        if summary_entry is None:
            logging.warning("Both external and local LLMs failed or were absent. Dropping oldest item.")
            return current_items[1:]

        return [summary_entry] + remaining
