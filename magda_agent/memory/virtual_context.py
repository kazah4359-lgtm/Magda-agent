import logging
from typing import TYPE_CHECKING
from magda_agent.emotions.engine import PADState
from magda_agent.memory.working import MemoryEntry

if TYPE_CHECKING:
    from magda_agent.memory.working import WorkingMemory
    from magda_agent.memory.episodic import EpisodicMemory

class VirtualContextManager:
    """
    VirtualContextManager handles paging out old short-term memory (WorkingMemory)
    into EpisodicMemory, and paging it back in when requested via semantic search.
    """

    async def page_out(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: int, count: int = 1) -> None:
        """
        Move the oldest `count` entries from WorkingMemory to EpisodicMemory.
        """
        entries = working_memory.get_entries(user_id=user_id)
        if not entries:
            return

        to_remove = entries[:count]
        for entry in to_remove:
            metadata = {
                "paged_out": True,
                "importance": entry.importance,
                "pad_p": entry.emotional_state.pleasure,
                "pad_a": entry.emotional_state.arousal,
                "pad_d": entry.emotional_state.dominance
            }
            if entry.tags:
                metadata["tags"] = ",".join(entry.tags)

            episodic_memory.store_event(
                text=entry.content,
                metadata=metadata,
                user_id=user_id
            )
            working_memory.remove(entry.id, user_id=user_id)
            logging.debug(f"Paged out memory entry {entry.id} for user {user_id}")


    async def page_in(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: int, query: str) -> None:
        """
        Recall relevant events from EpisodicMemory and load them into WorkingMemory.
        """
        events = episodic_memory.recall_events(query=query, top_k=5, user_id=user_id)
        for event_text in events:
            # We recreate a MemoryEntry. For a more sophisticated system, we would retrieve
            # the exact original metadata. But as per acceptance criteria, we restore the content.
            entry = MemoryEntry(
                content=event_text,
                importance=0.5,
                emotional_state=PADState(0, 0, 0),
                user_id=user_id
            )
            # Add to working memory. Note: this might trigger another page_out if limit is exceeded!
            await working_memory.add(entry)
            logging.debug(f"Paged in memory entry for user {user_id}: {event_text[:30]}...")
