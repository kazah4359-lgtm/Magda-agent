import logging
from typing import List
from magda_agent.llm_client import LLMClient
from magda_agent.memory.working import MemoryEntry

class ContextCompressorV2:
    """
    Context compressor v2 that selectively retrieves memories
    and compresses them to fit within context limits.
    """
    def __init__(self, llm: LLMClient):
        """
        Initialize the ContextCompressorV2.

        Args:
            llm: The LLM client used for summarizing text.
        """
        self.llm = llm

    async def compress_and_retrieve(self, memories: List[MemoryEntry], query: str, max_tokens: int = 1000) -> List[MemoryEntry]:
        """
        Filters memories based on the query and compresses them if they exceed the limit.

        Args:
            memories: A list of memory entries to process.
            query: The search query to filter relevance.
            max_tokens: Simulated token limit.

        Returns:
            A list of compressed and filtered memory entries.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        filtered = []
        for m in memories:
            content_lower = m.content.lower()
            # Basic exact match or word match
            if query_lower in content_lower or any(len(word) > 3 and word in content_lower for word in query_words):
                filtered.append(m)

        if not filtered:
            return []

        combined_text = "\n".join(m.content for m in filtered)
        if len(combined_text) <= max_tokens * 4:
            return filtered

        logging.info("Compressing text to fit context limits v2.")
        prompt = f"Summarize the following text related to '{query}', maintaining key points:\n\n{combined_text}"
        messages = [
            {"role": "system", "content": "You are a context compression engine. Return only the compressed text."},
            {"role": "user", "content": prompt}
        ]

        try:
            compressed_text = await self.llm.chat_completion(messages, temperature=0.3)
            compressed_text = compressed_text.strip()
        except Exception as e:
            logging.error(f"Failed to compress text v2: {e}")
            compressed_text = combined_text

        avg_importance = sum(m.importance for m in filtered) / len(filtered)
        first = filtered[0]

        summary_entry = MemoryEntry(
            content=compressed_text,
            importance=avg_importance,
            emotional_state=first.emotional_state,
            tags=first.tags,
            user_id=first.user_id
        )
        return [summary_entry]
