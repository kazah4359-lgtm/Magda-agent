import logging
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import PADState
from magda_agent.memory.working import MemoryEntry

if TYPE_CHECKING:
    from magda_agent.memory.working import WorkingMemory
    from magda_agent.memory.episodic import EpisodicMemory

class CoreMemory:
    """
    Explicit, multi-layered core memory context inspired by MemGPT.
    Contains persona (agent's identity), human (facts about the user),
    and task (current high-level goals).
    """
    def __init__(self, persona: str = "", human: str = "", task: str = ""):
        self.persona = persona
        self.human = human
        self.task = task

    def to_dict(self) -> Dict[str, str]:
        return {
            "persona": self.persona,
            "human": self.human,
            "task": self.task
        }

    def assemble(self) -> str:
        """Assembles the core memory into a single formatted string."""
        parts = []
        if self.persona:
            parts.append(f"CORE MEMORY (PERSONA):\n{self.persona}")
        if self.human:
            parts.append(f"CORE MEMORY (HUMAN):\n{self.human}")
        if self.task:
            parts.append(f"CORE MEMORY (TASK):\n{self.task}")
        return "\n\n".join(parts)

class VirtualContextManager:
    """
    VirtualContextManager handles multi-layered core memory and paging out
    old short-term memory (WorkingMemory) into EpisodicMemory.
    """
    def __init__(self, llm_client: Optional['LLMClient'] = None, section_limit: int = 1000) -> None:
        """
        Initializes the VirtualContextManager.

        Args:
            llm_client: Optional LLM client for summarization.
            section_limit: Heuristic word limit for core memory sections.
        """
        self.llm_client = llm_client
        self.core_memories: Dict[int, CoreMemory] = {}
        self.section_limit = section_limit

    def get_core_memory(self, user_id: int) -> CoreMemory:
        """Retrieves or creates the core memory for a specific user."""
        u_id = user_id if user_id is not None else -1
        if u_id not in self.core_memories:
            self.core_memories[u_id] = CoreMemory()
        return self.core_memories[u_id]

    async def update_core_section(self, user_id: int, section: str, content: str) -> None:
        """
        Updates a specific section of the core memory and enforces size limits.

        Args:
            user_id: The ID of the user.
            section: One of 'persona', 'human', 'task'.
            content: The new content for the section.
        """
        core = self.get_core_memory(user_id)
        if section == "persona":
            core.persona = content
        elif section == "human":
            core.human = content
        elif section == "task":
            core.task = content
        else:
            raise ValueError(f"Unknown core memory section: {section}")

        await self._maintain_core_size(user_id, section)

    async def _maintain_core_size(self, user_id: int, section: str) -> None:
        """Maintains the size of a core memory section to prevent overflow."""
        core = self.get_core_memory(user_id)
        content = getattr(core, section)
        if not content:
            return

        words = content.split()
        if len(words) > self.section_limit:
            logging.info(f"Core memory section '{section}' for user {user_id} exceeded limit ({len(words)} words). Compressing...")
            if self.llm_client:
                prompt = [
                    {"role": "system", "content": f"You are a context manager. Summarize the following '{section}' memory section to stay under {self.section_limit} words while preserving essential facts."},
                    {"role": "user", "content": content}
                ]
                summary = await self.llm_client.chat_completion(prompt)
                setattr(core, section, summary.strip())
            else:
                # Fallback: simple truncation
                setattr(core, section, " ".join(words[:self.section_limit]) + "...")

    async def compress_context(self, entries: List['MemoryEntry']) -> 'MemoryEntry':
        """
        Compresses multiple memory entries into a single summary entry.
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
        avg_p = sum(e.emotional_state.pleasure for e in entries) / len(entries)
        avg_a = sum(e.emotional_state.arousal for e in entries) / len(entries)
        avg_d = sum(e.emotional_state.dominance for e in entries) / len(entries)
        state = PADState(avg_p, avg_a, avg_d)

        return MemoryEntry(content=summary, importance=avg_importance, emotional_state=state, user_id=user_id)


    async def page_out(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: int, count: int = 1) -> None:
        """
        Move the oldest `count` entries from WorkingMemory to EpisodicMemory.
        """
        u_id = user_id if user_id is not None else -1
        entries = working_memory.get_entries(user_id=u_id)
        if not entries:
            return

        to_remove = entries[:count]
        if len(to_remove) > 1:
            try:
                # Attempt to compress context before paging out
                compressed_entry = await self.compress_context(to_remove)
                to_remove = [compressed_entry]
            except Exception as e:
                logging.error(f"Context compression failed during page_out: {e}")

        # Keep track of IDs to remove from working memory
        original_ids = [e.id for e in entries[:count]]

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
                user_id=u_id
            )
            logging.debug(f"Paged out memory entry {entry.id} for user {u_id}")

        for eid in original_ids:
            working_memory.remove(eid, user_id=u_id)


    async def page_in(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: int, query: str) -> None:
        """
        Recall relevant events from EpisodicMemory and load them into WorkingMemory.
        """
        u_id = user_id if user_id is not None else -1
        events = episodic_memory.recall_events(query=query, top_k=5, user_id=u_id)
        for event_text in events:
            # We recreate a MemoryEntry.
            entry = MemoryEntry(
                content=event_text,
                importance=0.5,
                emotional_state=PADState(0, 0, 0),
                user_id=u_id
            )
            # Add to working memory. Note: this might trigger another page_out if limit is exceeded!
            await working_memory.add(entry)
            logging.debug(f"Paged in memory entry for user {u_id}: {event_text[:30]}...")

# Implemented multi-layered core memory and virtual context management
