import logging
from typing import TYPE_CHECKING, List, Optional
from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import PADState
from magda_agent.memory.working import MemoryEntry

if TYPE_CHECKING:
    from magda_agent.memory.working import WorkingMemory
    from magda_agent.memory.episodic import EpisodicMemory

class VirtualContextManagerV3:
    """
    VirtualContextManagerV3 implements advanced virtual context management for
    explicitly paging out old short-term memory (WorkingMemory) into EpisodicMemory,
    and paging it back in when requested, inspired by the MemGPT/Letta patterns.
    """
    def __init__(self, llm_client: Optional['LLMClient'] = None) -> None:
        """
        Initializes the VirtualContextManagerV3.

        Args:
            llm_client: An optional LLMClient for advanced summarization during compression.
        """
        self.llm_client = llm_client

    async def compress_context(self, entries: List['MemoryEntry']) -> 'MemoryEntry':
        """
        Compresses multiple memory entries into a single summary entry using the LLM
        or a fallback summarization technique.

        Args:
            entries: A list of MemoryEntry objects to compress.

        Returns:
            A new MemoryEntry containing the summarized context.

        Raises:
            ValueError: If the entries list is empty.
        """
        if not entries:
            raise ValueError("No entries to compress")

        combined_text = "\n".join(e.content for e in entries)

        if self.llm_client:
            prompt = [{"role": "system", "content": "Summarize these memory entries concisely."},
                      {"role": "user", "content": combined_text}]
            summary = await self.llm_client.chat_completion(prompt)
        else:
            summary = f"Summary of {len(entries)} items: {combined_text[:50]}..."

        user_id = entries[0].user_id
        avg_importance = sum(e.importance for e in entries) / len(entries)

        # Calculate emotional state averages
        avg_p = sum(e.emotional_state.pleasure for e in entries if e.emotional_state) / len(entries)
        avg_a = sum(e.emotional_state.arousal for e in entries if e.emotional_state) / len(entries)
        avg_d = sum(e.emotional_state.dominance for e in entries if e.emotional_state) / len(entries)
        state = PADState(avg_p, avg_a, avg_d)

        return MemoryEntry(content=summary, importance=avg_importance, emotional_state=state, user_id=user_id)

    async def page_out_explicit(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: int, count: int = 1) -> None:
        """
        Explicitly moves the oldest `count` entries from WorkingMemory to EpisodicMemory.
        Uses compression if more than one entry is specified.

        Args:
            working_memory: The WorkingMemory instance to page out from.
            episodic_memory: The EpisodicMemory instance to page out into.
            user_id: The ID of the user.
            count: Number of oldest entries to page out.
        """
        entries = working_memory.get_entries(user_id=user_id)
        if not entries:
            return

        to_remove = entries[:count]
        original_to_remove = entries[:count]

        if len(to_remove) > 1:
            try:
                compressed_entry = await self.compress_context(to_remove)
                to_remove = [compressed_entry]
            except Exception as e:
                logging.error(f"Context compression failed during page_out_explicit: {e}")

        for entry in to_remove:
            metadata = {
                "paged_out_explicitly": True,
                "importance": entry.importance,
            }
            if entry.emotional_state:
                metadata["pad_p"] = entry.emotional_state.pleasure
                metadata["pad_a"] = entry.emotional_state.arousal
                metadata["pad_d"] = entry.emotional_state.dominance
            if entry.tags:
                metadata["tags"] = ",".join(entry.tags)

            episodic_memory.store_event(
                text=entry.content,
                metadata=metadata,
                user_id=user_id
            )
            logging.debug(f"Paged out memory entry {entry.id} for user {user_id}")

        for orig_entry in original_to_remove:
            working_memory.remove(orig_entry.id, user_id=user_id)

    async def page_in_explicit(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: int, query: str, top_k: int = 5) -> None:
        """
        Explicitly recalls relevant events from EpisodicMemory based on semantic similarity
        and loads them into WorkingMemory.

        Args:
            working_memory: The WorkingMemory instance to page in to.
            episodic_memory: The EpisodicMemory instance to page in from.
            user_id: The ID of the user.
            query: The semantic search query string.
            top_k: The number of top results to fetch.
        """
        events = episodic_memory.recall_events(query=query, top_k=top_k, user_id=user_id)
        for event_text in events:
            entry = MemoryEntry(
                content=event_text,
                importance=0.5,
                emotional_state=PADState(0, 0, 0),
                user_id=user_id
            )
            await working_memory.add(entry)
            logging.debug(f"Paged in explicit memory entry for user {user_id}: {event_text[:30]}...")

    async def paginate_explicit_memory_blocks(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: int, block_size: int = 2) -> None:
        """
        Divides the working memory into discrete blocks of a given size and pages them out
        to episodic memory explicitly. Avoids paging out incomplete blocks.

        Args:
            working_memory: The WorkingMemory instance to page out from.
            episodic_memory: The EpisodicMemory instance to page out into.
            user_id: The ID of the user.
            block_size: The number of entries per block to paginate.
        """
        entries = working_memory.get_entries(user_id=user_id)
        if not entries:
            return

        to_remove = entries[:- (len(entries) % block_size)] if len(entries) % block_size != 0 else entries
        if not to_remove:
            return

        blocks = [to_remove[i:i + block_size] for i in range(0, len(to_remove), block_size)]

        for block in blocks:
            for entry in block:
                metadata = {
                    "paged_out_explicitly": True,
                    "importance": entry.importance,
                }
                if entry.emotional_state:
                    metadata["pad_p"] = entry.emotional_state.pleasure
                    metadata["pad_a"] = entry.emotional_state.arousal
                    metadata["pad_d"] = entry.emotional_state.dominance
                if entry.tags:
                    metadata["tags"] = ",".join(entry.tags)

                episodic_memory.store_event(
                    text=entry.content,
                    metadata=metadata,
                    user_id=user_id
                )
                logging.debug(f"Paged out memory entry {entry.id} in block explicitly for user {user_id}")

            for entry in block:
                working_memory.remove(entry.id, user_id=user_id)

    def get_token_length(self, entries: List['MemoryEntry']) -> int:
        """
        Calculates a heuristic token length for the given memory entries.

        Args:
            entries: A list of MemoryEntry objects.

        Returns:
            The estimated token count.
        """
        total_words = sum(len(e.content.split()) for e in entries)
        return int(total_words * 1.3)

    async def maintain_working_memory_limits(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: int, max_tokens: int = 4000) -> None:
        """
        Calculates token length of current working context and transparently pages out
        older memories explicitly if the limit is exceeded.

        Args:
            working_memory: The WorkingMemory instance to manage.
            episodic_memory: The EpisodicMemory instance to page into.
            user_id: The ID of the user.
            max_tokens: The maximum token length allowed before paging out.
        """
        entries = working_memory.get_entries(user_id=user_id)
        if not entries:
            return

        current_tokens = self.get_token_length(entries)
        if current_tokens <= max_tokens:
            return

        logging.info(f"Working memory context length ({current_tokens} tokens) exceeds limit ({max_tokens} tokens). Paging out...")

        count_to_remove = max(1, len(entries) // 2)
        await self.page_out_explicit(working_memory, episodic_memory, user_id, count=count_to_remove)
