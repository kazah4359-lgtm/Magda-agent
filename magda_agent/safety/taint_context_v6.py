"""MCP Kernel Taint Tracking Context Integration.

Integrates TaintTrackerV2 with WorkingMemory.
"""
from typing import List, Dict, Optional, Callable, Awaitable, Any
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.safety.taint_tracking_v2 import TaintTrackerV2

class TaintedWorkingMemory(WorkingMemory):
    """Working memory that tracks tainted data."""
    def __init__(self, limit: int = 10, context_engine: Optional[Any] = None, tracker: Optional[TaintTrackerV2] = None):
        super().__init__(limit=limit, context_engine=context_engine)
        self.tracker = tracker if tracker is not None else TaintTrackerV2()

    async def add(self, entry: MemoryEntry, summarizer: Optional[Callable[[List[MemoryEntry]], Awaitable[MemoryEntry]]] = None) -> None:
        """Add a memory entry to the active working memory, propagating taint."""
        if self.tracker.is_tainted(entry.content):
            if "tainted" not in entry.tags:
                entry.tags.append("tainted")
        await super().add(entry, summarizer)
