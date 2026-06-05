import time
import logging
from typing import List, Dict, Optional
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
    def __init__(self, limit: int = 10):
        self.limit = limit
        self._entries_by_user: Dict[int, List[MemoryEntry]] = {}

    def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry to the active working memory."""
        u_id = entry.user_id if entry.user_id is not None else -1
        user_entries = self._entries_by_user.setdefault(u_id, [])
        user_entries.append(entry)

        # Enforce bounded limit by removing oldest entries if exceeded
        while len(user_entries) > self.limit:
            user_entries.pop(0)

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
