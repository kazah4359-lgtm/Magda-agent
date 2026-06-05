import time
import json
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from magda_agent.emotions.engine import PADState
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.memory.episodic import EpisodicMemory

class MemorySystem:
    """
    Hierarchical Memory System with Short-Term and Long-Term storage.
    Includes emotional coloring and importance-based decay.
    Uses WorkingMemory for short-term and EpisodicMemory for long-term consolidation.
    """
    def __init__(self, short_term_limit: int = 10, persist_directory: str = "./memory_db"):
        self.short_term_limit = short_term_limit
        self.working_memory = WorkingMemory(limit=short_term_limit)
        self.episodic_memory = EpisodicMemory(persist_directory=persist_directory)

        # For backwards compatibility and testing
        self._long_term_by_user: Dict[int, List[MemoryEntry]] = {}

    @property
    def short_term(self) -> List[MemoryEntry]:
        """Flattened list of all short-term memories across all users."""
        return self.working_memory.get_all_entries()

    @property
    def long_term(self) -> List[MemoryEntry]:
        """Flattened list of all long-term memories across all users."""
        return [entry for user_list in self._long_term_by_user.values() for entry in user_list]

    def add_memory(self, content: str, importance: float, emotional_state: PADState, tags: List[str] = None, user_id: int = None):
        """Add a new entry to short-term working memory."""
        entry = MemoryEntry(
            content=content,
            importance=importance,
            emotional_state=emotional_state,
            tags=tags or [],
            user_id=user_id
        )

        self.working_memory.add(entry)
        u_id = user_id if user_id is not None else -1

        user_short_term = self.working_memory.get_entries(u_id)

        # If short-term memory is full, consolidate
        if len(user_short_term) >= self.short_term_limit:
            self._consolidate_user(u_id)

    def consolidate(self, user_id: Optional[int] = None):
        """
        Move important or emotionally significant memories to long-term storage.
        Less important memories in short-term are eventually discarded.
        If user_id is None, it consolidates all tracked users.
        """
        if user_id is None:
            # None implies decaying all tracked states (for background jobs)
            # Find all users currently in working memory
            all_users = list(self.working_memory._entries_by_user.keys())
            for u in all_users:
                self._consolidate_user(u)
        else:
            u_id = user_id if user_id is not None else -1
            self._consolidate_user(u_id)

    def _consolidate_user(self, u_id: int):
        user_short_term = self.working_memory.get_entries(u_id)
        if not user_short_term:
            return

        user_long_term = self._long_term_by_user.setdefault(u_id, [])

        # Sort by importance and emotional intensity
        user_short_term.sort(key=lambda x: x.importance + self._calc_emotional_intensity(x.emotional_state), reverse=True)

        # Move the top memory to long-term
        if user_short_term:
            most_important = user_short_term[0]
            if most_important.importance > 0.3: # Minimum threshold for long-term storage
                user_long_term.append(most_important)
                # Store in episodic memory for true long-term retrieval
                meta = {"importance": most_important.importance, "tags": ",".join(most_important.tags)}
                self.episodic_memory.store_event(
                    text=most_important.content,
                    metadata=meta,
                    user_id=most_important.user_id
                )

            # Remove the consolidated item from working memory RAM
            self.working_memory.remove(most_important.id, u_id)
            user_short_term = self.working_memory.get_entries(u_id)

        # Trim short-term memory if still over limit
        while len(user_short_term) > self.short_term_limit:
            discarded = user_short_term[-1]
            self.working_memory.remove(discarded.id, u_id)
            user_short_term = self.working_memory.get_entries(u_id)

    def retrieve_relevant(self, query: str, limit: int = 5, user_id: int = None) -> List[MemoryEntry]:
        """
        Retrieve relevant memories. Since WorkingMemory is small, we use keyword matching.
        For robust semantic search, external modules should query SemanticMemory/EpisodicMemory.
        """
        try:
            u_id = user_id if user_id is not None else -1
            entries = self.working_memory.get_entries(u_id)

            if not entries:
                return []

            query_lower = query.lower()
            query_words = set(query_lower.split())

            scored_entries = []
            for entry in entries:
                score = 0
                content_lower = entry.content.lower()
                # Basic exact match
                if query_lower in content_lower:
                    score += 5
                # Word match
                for word in query_words:
                    if len(word) > 3 and word in content_lower:
                        score += 1

                if score > 0:
                    scored_entries.append((score, entry))

            if not scored_entries:
                # Fallback: just return the most important recent ones
                return sorted(entries, key=lambda x: x.importance, reverse=True)[:limit]

            # Sort by score descending, then importance
            scored_entries.sort(key=lambda x: (x[0], x[1].importance), reverse=True)
            return [e[1] for e in scored_entries[:limit]]

        except Exception as e:
            logging.error(f"Failed to retrieve relevant memories: {e}")
            return []

    def _calc_emotional_intensity(self, state: PADState) -> float:
        return math.sqrt(state.pleasure**2 + state.arousal**2 + state.dominance**2)

    def get_summary(self) -> str:
        return f"Memory: {len(self.short_term)} Short-term, {len(self.long_term)} Long-term entries."

    def close(self):
        """Clean up memory systems on shutdown."""
        try:
            if hasattr(self.episodic_memory.client, "clear_system_cache"):
                self.episodic_memory.client.clear_system_cache()
            logging.info("MemorySystem gracefully closed.")
        except Exception as e:
            logging.error(f"Failed to close MemorySystem: {e}")
