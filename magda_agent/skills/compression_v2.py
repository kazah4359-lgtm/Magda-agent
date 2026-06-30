import logging
from typing import Any, Dict, List, Optional
from magda_agent.memory.context_engine import ContextPlugin
from magda_agent.llm_client import LLMClient
from magda_agent.memory.working import MemoryEntry

class OpenClawContextCompressorV2(ContextPlugin):
    """
    OpenClaw-inspired context compressor that automatically compresses
    short-term memory during the retrieval phase or when explicitly requested.
    """
    def __init__(self, llm: LLMClient, threshold: int = 10):
        self.llm = llm
        self.threshold = threshold
        logging.info(f"OpenClawContextCompressorV2 initialized with threshold {threshold}")

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        pass

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        return "\n".join([str(item.content) if hasattr(item, 'content') else str(item) for item in context_items])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        Explicitly compress the context if it exceeds the limit.
        """
        limit = metadata.get("limit", self.threshold)
        if len(context_items) <= limit:
            return context_items

        logging.info(f"OpenClawContextCompressorV2: Compacting {len(context_items)} items to limit {limit}")
        return await self._compress_entries(context_items, limit)

    def before_retrieval(self, query: str, user_id: int) -> str:
        """
        Hook called before retrieval. Can be used for query refinement.
        """
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """
        Hook called after retrieval. If context is too large, we could trigger
        async compression, but for now we just return it.
        """
        return context

    async def _compress_entries(self, entries: List[Any], limit: int) -> List[Any]:
        """
        Helper to compress entries using the LLM.
        """
        if not entries:
            return []

        # Keep the most recent entries and compress the older ones
        # For simplicity, we compress the first half if we are over limit
        to_compress = entries[:len(entries) - limit + 1]
        remaining = entries[len(entries) - limit + 1:]

        combined_text = "\n".join([getattr(e, 'content', str(e)) for e in to_compress])
        prompt = f"Summarize the following memory context into a single concise summary while maintaining key facts:\n{combined_text}"

        try:
            summary_content = await self.llm.chat_completion([
                {"role": "system", "content": "You are a context compression engine. Return only the summary text."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)

            first = to_compress[0]
            avg_importance = sum(getattr(e, 'importance', 0.5) for e in to_compress) / len(to_compress)

            summary_entry = MemoryEntry(
                content=summary_content.strip(),
                importance=avg_importance,
                emotional_state=getattr(first, 'emotional_state', None),
                tags=list(set(t for e in to_compress if hasattr(e, 'tags') for t in e.tags)),
                user_id=getattr(first, 'user_id', None)
            )
            return [summary_entry] + remaining
        except Exception as e:
            logging.error(f"OpenClawContextCompressorV2: Compression failed: {e}")
            return entries[1:] # Fallback to dropping the oldest

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        pass
