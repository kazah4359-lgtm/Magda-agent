import logging
from typing import List, Optional
from magda_agent.llm_client import LLMClient
from magda_agent.memory.working import MemoryEntry

class ContextCompressor:
    """
    Context compressor that selectively summarizes older memory entries to fit within
    limited context windows and filters low-salience memories.
    """
    def __init__(self, llm: LLMClient):
        """
        Initialize the ContextCompressor.

        Args:
            llm: The LLM client used for summarizing text.
        """
        self.llm = llm

    async def compress_text(self, text: str, max_tokens: int = 1000) -> str:
        """
        Compresses a given text to reduce its token size while maintaining key points.

        Args:
            text: The input text to compress.
            max_tokens: A simulated max token limit based on text length.

        Returns:
            The compressed text, or the original text if it's within limits or if compression fails.
        """
        # Roughly estimating tokens by length (1 token ~ 4 chars)
        if len(text) <= max_tokens * 4:
            return text

        logging.info("Compressing text to fit context limits.")
        prompt = f"Summarize the following text, maintaining key points, to fit within a smaller context window:\n\n{text}"
        messages = [
            {"role": "system", "content": "You are a context compression engine. Return only the compressed text."},
            {"role": "user", "content": prompt}
        ]

        try:
            compressed = await self.llm.chat_completion(messages, temperature=0.3)
            return compressed.strip()
        except Exception as e:
            logging.error(f"Failed to compress text: {e}")
            return text

    def filter_low_salience(self, memories: List[MemoryEntry], threshold: float = 0.5) -> List[MemoryEntry]:
        """
        Selectively filters out low-salience memories.

        Args:
            memories: A list of memory entries.
            threshold: The importance threshold. Memories below this are filtered out.

        Returns:
            A list of memory entries with importance >= threshold.
        """
        return [m for m in memories if m.importance >= threshold]

    async def compress_memories(self, memories: List[MemoryEntry], max_tokens: int = 1000, threshold: float = 0.5) -> List[MemoryEntry]:
        """
        Filters low-salience memories and compresses the remaining content if it exceeds the limit.

        Args:
            memories: A list of memory entries.
            max_tokens: The simulated token limit for the total text.
            threshold: The importance threshold for filtering.

        Returns:
            A list of compressed memory entries. If compressed, it will be a single summarized entry.
        """
        filtered = self.filter_low_salience(memories, threshold)
        if not filtered:
            return []

        combined_text = "\n".join(m.content for m in filtered)
        if len(combined_text) <= max_tokens * 4:
            return filtered

        compressed_text = await self.compress_text(combined_text, max_tokens)

        # Average importance of the summarized memories
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
