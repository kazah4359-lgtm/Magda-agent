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
        self.short_term: List[MemoryEntry] = []
        self.long_term: List[MemoryEntry] = []
        self.short_term_limit = short_term_limit

        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(name="working_memory")
        self._entries: Dict[str, MemoryEntry] = {}

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
        self.short_term.append(entry)

        entry_id_str = str(entry.id)
        self._entries[entry_id_str] = entry

        meta = {"importance": importance}
        if user_id is not None:
            meta["user_id"] = user_id

        try:
            self.collection.add(
                documents=[content],
                metadatas=[meta],
                ids=[entry_id_str]
            )
        except Exception as e:
            logging.error(f"Failed to add memory to ChromaDB: {e}")

        # If short-term memory is full, consolidate
        if len(self.short_term) > self.short_term_limit:
            self.consolidate()

    def consolidate(self):
        """
        Move important or emotionally significant memories to long-term storage.
        Less important memories in short-term are eventually discarded.
        """
        # Sort by importance and emotional intensity
        self.short_term.sort(key=lambda x: x.importance + self._calc_emotional_intensity(x.emotional_state), reverse=True)

        # Move the top memory to long-term
        if self.short_term:
            most_important = self.short_term.pop(0)
            if most_important.importance > 0.3: # Minimum threshold for long-term storage
                self.long_term.append(most_important)
            else:
                self._remove_from_index(str(most_important.id))

        # Trim short-term memory
        while len(self.short_term) > self.short_term_limit:
            discarded = self.short_term.pop()
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

            query_kwargs = {
                "query_texts": [query],
                "n_results": min(limit, self.collection.count())
            }
            if user_id is not None:
                query_kwargs["where"] = {"user_id": user_id}

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
