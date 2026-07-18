import logging
from typing import List
from magda_agent.llm_client import LLMClient
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

class SelectiveContextCompressor:
    """
    Context compressor module for virtual context management that selectively
    compresses older working memory entries when the token limit is exceeded,
    prior to semantic storage.
    """
    def __init__(self, llm: LLMClient):
        """
        Initialize the SelectiveContextCompressor.

        Args:
            llm: The LLM client used for summarizing text.
        """
        self.llm = llm

    def _estimate_tokens(self, text: str) -> int:
        """
        Calculates a heuristic token length for the given text.
        Roughly 1.3 tokens per word.
        """
        return int(len(text.split()) * 1.3)

    def _estimate_memories_tokens(self, memories: List[MemoryEntry]) -> int:
        """
        Calculates a heuristic token length for a list of memory entries.
        """
        total_words = sum(len(e.content.split()) for e in memories)
        return int(total_words * 1.3)

    async def compress_old_memories(self, memories: List[MemoryEntry], max_tokens: int = 1000) -> List[MemoryEntry]:
        """
        Checks if the list of memories exceeds the token limit. If so, selectively
        compresses the oldest memories while keeping newer ones intact.

        Args:
            memories: A list of memory entries, assumed to be chronologically ordered (oldest first).
            max_tokens: The simulated maximum token limit.

        Returns:
            A list of memory entries, potentially with a compressed entry replacing older ones.
        """
        if not memories:
            return []

        current_tokens = self._estimate_memories_tokens(memories)
        if current_tokens <= max_tokens:
            return memories

        logging.info(f"Context length ({current_tokens} tokens) exceeds limit ({max_tokens} tokens). Selectively compressing old memories...")

        # Divide memories: Keep half of them (the newer ones), compress the older half
        keep_count = len(memories) // 2
        to_compress = memories[:-keep_count] if keep_count > 0 else memories
        to_keep = memories[-keep_count:] if keep_count > 0 else []

        if len(to_compress) <= 1 and not to_keep:
            # Nothing meaningful to compress if only 1 item and we are keeping 0
            # Just compress it if it's too large, but typically an entry shouldn't be larger than max_tokens
            to_compress = memories
            to_keep = []

        combined_text = "\n".join(m.content for m in to_compress)
        prompt = f"Please summarize the following older working memory context into a single concise bullet point:\n{combined_text}"
        messages = [
            {"role": "system", "content": "You are a context compression engine. Return only the compressed text."},
            {"role": "user", "content": prompt}
        ]

        try:
            compressed_text = await self.llm.chat_completion(messages, temperature=0.3)
            compressed_text = compressed_text.strip()
        except Exception as e:
            logging.error(f"Failed to selectively compress text: {e}")
            compressed_text = combined_text

        # Average properties for the summary entry
        avg_importance = sum(m.importance for m in to_compress) / len(to_compress)

        # Calculate emotional state averages (using pleasure as proxy or full PAD)
        avg_p = 0.0
        avg_a = 0.0
        avg_d = 0.0
        has_state = False

        for m in to_compress:
            if m.emotional_state:
                avg_p += m.emotional_state.pleasure
                avg_a += m.emotional_state.arousal
                avg_d += m.emotional_state.dominance
                has_state = True

        if has_state:
            state = PADState(avg_p / len(to_compress), avg_a / len(to_compress), avg_d / len(to_compress))
        else:
            state = PADState(0, 0, 0)

        first = to_compress[0]

        # Merge tags safely
        all_tags = set()
        for m in to_compress:
            if m.tags:
                all_tags.update(m.tags)

        summary_entry = MemoryEntry(
            content=compressed_text,
            importance=avg_importance,
            emotional_state=state,
            tags=list(all_tags),
            user_id=first.user_id
        )

        return [summary_entry] + to_keep
