import time
import logging
from typing import List, Dict, Optional, Callable, Awaitable, Any
from magda_agent.emotions.engine import PADState

class MemoryEntry:
    """A single memory entry for short or long-term storage."""
    def __init__(self, content: str, importance: float, emotional_state: PADState, tags: List[str] = None, user_id: Optional[int] = None):
        self.content = content
        self.timestamp = time.time()
        self.importance = importance
        self.emotional_state = emotional_state
        self.tags = tags or []
        self.id = int(time.time() * 1000)
        self.user_id = user_id

class WorkingMemory:
    """
    Working Memory stores bounded, short-term context for active tasks.
    It does not use persistent storage and does not use ChromaDB.
    """
    def __init__(self, limit: int = 10, context_engine: Optional[Any] = None):
        self.limit = limit
        self.context_engine = context_engine
        self._entries_by_user: Dict[int, List[MemoryEntry]] = {}

    async def add(self, entry: MemoryEntry, summarizer: Optional[Callable[[List['MemoryEntry']], Awaitable['MemoryEntry']]] = None) -> None:
        """Add a memory entry to the active working memory."""
        u_id = entry.user_id if entry.user_id is not None else -1
        user_entries = self._entries_by_user.setdefault(u_id, [])
        user_entries.append(entry)

        # Enforce bounded limit by removing oldest entries if exceeded
        while len(user_entries) > self.limit:
            if self.context_engine:
                # Use ContextEngine compact lifecycle hook
                user_entries = await self.context_engine.compact(user_entries, {"limit": self.limit, "user_id": u_id})
            elif summarizer:
                # Take oldest two entries to summarize, so we compress context and reduce length by 1
                to_summarize = user_entries[:2]
                user_entries = user_entries[2:]

                try:
                    summary_entry = await summarizer(to_summarize)
                    # Insert summary at the beginning
                    user_entries.insert(0, summary_entry)
                except Exception as e:
                    logging.error(f"Summarizer failed, falling back to drop: {e}")
                    # Revert to plain dropping if summarizer fails, to ensure limit is enforced
                    user_entries.insert(0, to_summarize[1]) # Keep the 2nd oldest if 1 is dropped, so len reduces by 1
            else:
                user_entries.pop(0)

        self._entries_by_user[u_id] = user_entries

    def get_entries(self, user_id: Optional[int] = None) -> List[MemoryEntry]:
        """Get the current working memory entries for a user."""
        u_id = user_id if user_id is not None else -1
        return self._entries_by_user.get(u_id, [])

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all flattened memory entries across all users."""
        return [entry for user_list in self._entries_by_user.values() for entry in user_list]

    def remove(self, entry_id: int, user_id: Optional[int] = None) -> None:
        """Remove a memory entry by ID."""
        u_id = user_id if user_id is not None else -1
        if u_id in self._entries_by_user:
            self._entries_by_user[u_id] = [
                e for e in self._entries_by_user[u_id] if e.id != entry_id
            ]

    def clear(self, user_id: Optional[int] = None) -> None:
        """Clear the working memory for a user."""
        u_id = user_id if user_id is not None else -1
        if u_id in self._entries_by_user:
            self._entries_by_user[u_id] = []
