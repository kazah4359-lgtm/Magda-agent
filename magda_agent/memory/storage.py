import time
import json
import math
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
    Includes emotional coloring and importance-based decay.
    """
    def __init__(self, short_term_limit: int = 10):
        self.short_term: List[MemoryEntry] = []
        self.long_term: List[MemoryEntry] = []
        self.short_term_limit = short_term_limit

    def add_memory(self, content: str, importance: float, emotional_state: PADState, tags: List[str] = None, user_id: int = None):
        """Add a new entry to short-term memory."""
        entry = MemoryEntry(
            content=content,
            timestamp=time.time(),
            importance=importance,
            emotional_state=emotional_state,
            tags=tags or [],
            user_id=user_id
        )
        self.short_term.append(entry)

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

        # Trim short-term memory
        while len(self.short_term) > self.short_term_limit:
            self.short_term.pop()

    def retrieve_relevant(self, query: str, limit: int = 5, user_id: int = None) -> List[MemoryEntry]:
        """
        Retrieve relevant memories based on tags or simple keyword matching.
        In a real scenario, this would use vector search.
        """
        all_memories = self.short_term + self.long_term
        if user_id is not None:
            all_memories = [m for m in all_memories if m.user_id == user_id]

        # Simple keyword matching for demonstration
        results = [m for m in all_memories if query.lower() in m.content.lower() or any(query.lower() in t.lower() for t in m.tags)]
        return sorted(results, key=lambda x: x.importance, reverse=True)[:limit]

    def _calc_emotional_intensity(self, state: PADState) -> float:
        return math.sqrt(state.pleasure**2 + state.arousal**2 + state.dominance**2)

    def get_summary(self) -> str:
        return f"Memory: {len(self.short_term)} Short-term, {len(self.long_term)} Long-term entries."
