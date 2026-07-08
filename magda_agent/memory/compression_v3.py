import logging
from typing import List
from magda_agent.llm_client import LLMClient
from magda_agent.memory.working import MemoryEntry

class OpenClawContextCompressorV3:
    """
    OpenClaw-inspired context compressor v3 that selectively retrieves memories.
    It preserves the most important items (critical memories) and compresses
    the remaining ones when the context size exceeds the limit.
    """
    def __init__(self, llm: LLMClient, threshold: float = 0.8) -> None:
        """
        Initialize the OpenClawContextCompressorV3.

        Args:
            llm: The LLM client used for summarizing text.
            threshold: The importance threshold. Memories with importance >= threshold
                       are considered critical and will not be compressed.
        """
        self.llm = llm
        self.threshold = threshold

    async def compress_and_retrieve(self, memories: List[MemoryEntry], query: str, max_tokens: int = 1000) -> List[MemoryEntry]:
        """
        Filters memories based on the query and selectively compresses them.
        Critical memories (importance >= threshold) are preserved as is.
        Non-critical memories are compressed if the total context size exceeds max_tokens.

        Args:
            memories: A list of memory entries to process.
            query: The search query to filter relevance.
            max_tokens: Simulated token limit for the context.

        Returns:
            A list containing critical memories and a compressed summary of non-critical memories (if any).
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

        critical_memories = [m for m in filtered if m.importance >= self.threshold]
        non_critical_memories = [m for m in filtered if m.importance < self.threshold]

        combined_text = "\n".join(m.content for m in filtered)

        # If the total text is within the limit, return all filtered memories
        if len(combined_text) <= max_tokens * 4:
            return filtered

        result = []
        result.extend(critical_memories)

        if non_critical_memories:
            non_critical_text = "\n".join(m.content for m in non_critical_memories)

            logging.info("Compressing non-critical text to fit context limits v3.")
            prompt = f"Summarize the following text related to '{query}', maintaining key points:\n\n{non_critical_text}"
            messages = [
                {"role": "system", "content": "You are a context compression engine. Return only the compressed text."},
                {"role": "user", "content": prompt}
            ]

            try:
                compressed_text = await self.llm.chat_completion(messages, temperature=0.3)
                compressed_text = compressed_text.strip()
            except Exception as e:
                logging.error(f"Failed to compress text v3: {e}")
                compressed_text = non_critical_text

            avg_importance = sum(m.importance for m in non_critical_memories) / len(non_critical_memories)
            first = non_critical_memories[0]

            # Combine tags from non-critical entries
            all_tags = set()
            for m in non_critical_memories:
                if m.tags:
                    all_tags.update(m.tags)

            summary_entry = MemoryEntry(
                content=compressed_text,
                importance=avg_importance,
                emotional_state=first.emotional_state,
                tags=list(all_tags),
                user_id=first.user_id
            )
            result.append(summary_entry)

        return result
