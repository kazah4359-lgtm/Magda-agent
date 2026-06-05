import time
import json
import math
import logging
import chromadb
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from magda_agent.emotions.engine import PADState

@dataclass
class MemoryEntry:
    content: str
    timestamp: float
    importance: float  # 0.0 to 1.0
    emotional_state: PADState
    tags: List[str] = field(default_factory=list)
    id: int = field(default_factory=lambda: int(time.time() * 1000))
    user_id: Optional[int] = None

class MemorySystem:
    """
    Hierarchical Memory System with Short-Term and Long-Term storage.
    Includes emotional coloring, importance-based decay, and ChromaDB vector search.
    """
    def __init__(self, short_term_limit: int = 10):
        self._short_term_by_user: Dict[int, List[MemoryEntry]] = {}
        self._long_term_by_user: Dict[int, List[MemoryEntry]] = {}
        self.short_term_limit = short_term_limit

        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(name="working_memory")
        self._entries: Dict[str, MemoryEntry] = {}

    @property
    def short_term(self) -> List[MemoryEntry]:
        """Flattened list of all short-term memories across all users."""
        return [entry for user_list in self._short_term_by_user.values() for entry in user_list]

    @property
    def long_term(self) -> List[MemoryEntry]:
        """Flattened list of all long-term memories across all users."""
        return [entry for user_list in self._long_term_by_user.values() for entry in user_list]

    def add_memory(self, content: str, importance: float, emotional_state: PADState, tags: List[str] = None, user_id: int = None):
        """Add a new entry to short-term memory and index it in ChromaDB."""
        entry = MemoryEntry(
            content=content,
            timestamp=time.time(),
            importance=importance,
            emotional_state=emotional_state,
            tags=tags or [],
            user_id=user_id
        )

        u_id = user_id if user_id is not None else -1
        user_short_term = self._short_term_by_user.setdefault(u_id, [])
        user_short_term.append(entry)

        entry_id_str = str(entry.id)
        self._entries[entry_id_str] = entry

        meta = {"importance": importance, "user_id": u_id}

        try:
            self.collection.add(
                documents=[content],
                metadatas=[meta],
                ids=[entry_id_str]
            )
        except Exception as e:
            logging.error(f"Failed to add memory to ChromaDB: {e}")

        # If short-term memory is full, consolidate
        if len(user_short_term) > self.short_term_limit:
            # Consolidate only this user's memory, pass u_id directly or wrap in logic
            self._consolidate_user(u_id)

    def consolidate(self, user_id: Optional[int] = None):
        """
        Move important or emotionally significant memories to long-term storage.
        Less important memories in short-term are eventually discarded.
        If user_id is None, it consolidates all tracked users.
        """
        if user_id is None:
            # None implies decaying all tracked states (for background jobs)
            for u in list(self._short_term_by_user.keys()):
                self._consolidate_user(u)
        else:
            self._consolidate_user(user_id)

    def _consolidate_user(self, u_id: int):
        user_short_term = self._short_term_by_user.setdefault(u_id, [])
        user_long_term = self._long_term_by_user.setdefault(u_id, [])

        # Sort by importance and emotional intensity
        user_short_term.sort(key=lambda x: x.importance + self._calc_emotional_intensity(x.emotional_state), reverse=True)

        # Move the top memory to long-term
        if user_short_term:
            most_important = user_short_term.pop(0)
            if most_important.importance > 0.3: # Minimum threshold for long-term storage
                user_long_term.append(most_important)
            else:
                self._remove_from_index(str(most_important.id))

        # Trim short-term memory
        while len(user_short_term) > self.short_term_limit:
            discarded = user_short_term.pop()
            self._remove_from_index(str(discarded.id))

    def _remove_from_index(self, entry_id_str: str) -> None:
        """Remove a discarded memory from the internal dict and ChromaDB index."""
        if entry_id_str in self._entries:
            del self._entries[entry_id_str]
        try:
            self.collection.delete(ids=[entry_id_str])
        except Exception as e:
            logging.error(f"Failed to delete memory from ChromaDB: {e}")

    def retrieve_relevant(self, query: str, limit: int = 5, user_id: int = None) -> List[MemoryEntry]:
        """
        Retrieve relevant memories using ChromaDB vector search.
        """
        try:
            # Prevent querying if the collection is empty
            if self.collection.count() == 0:
                return []

            u_id = user_id if user_id is not None else -1
            query_kwargs = {
                "query_texts": [query],
                "n_results": min(limit, self.collection.count()),
                "where": {"user_id": u_id}
            }

            results = self.collection.query(**query_kwargs)

            entries = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                for entry_id in results["ids"][0]:
                    if entry_id in self._entries:
                        entries.append(self._entries[entry_id])

            # Sort by importance
            return sorted(entries, key=lambda x: x.importance, reverse=True)

        except Exception as e:
            logging.error(f"Failed to retrieve relevant memories: {e}")
            return []

    def _calc_emotional_intensity(self, state: PADState) -> float:
        return math.sqrt(state.pleasure**2 + state.arousal**2 + state.dominance**2)

    def get_summary(self) -> str:
        return f"Memory: {len(self.short_term)} Short-term, {len(self.long_term)} Long-term entries."

    def close(self):
        """Clean up the EphemeralClient on shutdown."""
        try:
            self.client.clear_system_cache()
            logging.info("MemorySystem ChromaDB client gracefully closed.")
        except Exception as e:
            logging.error(f"Failed to close MemorySystem ChromaDB client: {e}")
