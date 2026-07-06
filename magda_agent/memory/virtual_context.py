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
    Implements ContextPlugin protocol for integration with ContextEngine.
    """
    def __init__(self, llm_client: Optional['LLMClient'] = None, section_limit: int = 1000) -> None:
        """
        Initializes the VirtualContextManager.

        Args:
            llm_client: Optional LLM client for summarization.
            section_limit: Heuristic word limit for core memory sections.
        """
        self.llm_client = llm_client
        self.core_memories: Dict[str, CoreMemory] = {}
        self.section_limit = section_limit

    # --- ContextPlugin Protocol Hooks ---

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Process incoming content before it is stored or used."""
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble the context string including core memory for the LLM."""
        user_id = metadata.get("user_id")
        # Ensure user_id is string for key lookup if present
        u_id = str(user_id) if user_id is not None else "default"
        core = self.get_core_memory(u_id)

        parts = []
        core_str = core.assemble()
        if core_str:
            parts.append(core_str)

        working_str = "\n".join([f"- {item.content}" for item in context_items if hasattr(item, 'content')])
        if working_str:
            parts.append(f"WORKING MEMORY:\n{working_str}")

        return "\n\n".join(parts)

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Compact context when limits are reached by compressing oldest items."""
        limit = metadata.get("limit", 10)
        if len(context_items) <= limit:
            return context_items

        # Basic compaction: compress all to a single entry
        try:
            compressed = await self.compress_context(context_items)
            return [compressed]
        except Exception as e:
            logging.error(f"Compaction failed in VirtualContextManager: {e}")
            return context_items[1:] # Fallback: drop oldest

    def on_context_update(self, new_context: Any, user_id: Optional[str]) -> None:
        pass

    # --- Core Memory Management ---

    def get_core_memory(self, user_id: Optional[str]) -> CoreMemory:
        """Retrieves or creates the core memory for a specific user."""
        u_id = str(user_id) if user_id is not None else "default"
        if u_id not in self.core_memories:
            self.core_memories[u_id] = CoreMemory()
        return self.core_memories[u_id]

    async def update_core_section(self, user_id: Optional[str], section: str, content: str) -> None:
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

    async def _maintain_core_size(self, user_id: Optional[str], section: str) -> None:
        """Maintains the size of a core memory section to prevent overflow."""
        core = self.get_core_memory(user_id)
        content = getattr(core, section)
        if not content:
            return

        words = content.split()
        if len(words) > self.section_limit:
            logging.info(f"Core memory section '{section}' exceeded limit. Compressing...")
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


    def get_token_length(self, entries: List['MemoryEntry']) -> int:
        """Calculates a heuristic token length for the given memory entries."""
        total_words = sum(len(e.content.split()) for e in entries)
        return int(total_words * 1.3)

    async def maintain_working_memory_limits(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: Optional[str], max_tokens: int = 4000) -> None:
        """Pages out older memories if the token limit is exceeded."""
        entries = working_memory.get_entries(user_id=user_id)
        if not entries:
            return

        current_tokens = self.get_token_length(entries)
        if current_tokens <= max_tokens:
            return

        logging.info(f"Working memory context length ({current_tokens} tokens) exceeds limit. Paging out...")
        count_to_remove = max(1, len(entries) // 2)
        await self.page_out_explicit(working_memory, episodic_memory, user_id, count=count_to_remove)


    async def compress_context(self, entries: List['MemoryEntry']) -> 'MemoryEntry':
        """Compresses multiple memory entries into a single summary entry."""
        if not entries:
            raise ValueError("No entries to compress")

        combined_text: str = "\n".join(e.content for e in entries)

        if self.llm_client:
            prompt: List[Dict[str, str]] = [
                {"role": "system", "content": "Summarize these memory entries concisely."},
                {"role": "user", "content": combined_text}
            ]
            summary = await self.llm_client.chat_completion(prompt)
        else:
            summary = f"Summary of {len(entries)} items: {combined_text[:50]}..."

        first = entries[0]
        avg_importance: float = sum(e.importance for e in entries) / len(entries)

        # Aggregate emotional state (averaging pleasure as a proxy for simplicity)
        avg_p = 0.0
        if first.emotional_state:
            avg_p = sum(e.emotional_state.pleasure for e in entries if e.emotional_state) / len(entries)

        state: PADState = PADState(avg_p, 0.0, 0.0)

        return MemoryEntry(
            content=summary,
            importance=avg_importance,
            emotional_state=state,
            user_id=first.user_id
        )

    # --- Explicit Paging Methods ---

    async def page_out_explicit(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: Optional[str], count: int = 1) -> None:
        """
        Explicitly move the oldest `count` entries from WorkingMemory to EpisodicMemory.
        """
        entries: List['MemoryEntry'] = working_memory.get_entries(user_id=user_id)
        if not entries:
            return

        to_remove: List['MemoryEntry'] = entries[:count]
        original_ids: List[str] = [e.id for e in to_remove]

        if len(to_remove) > 1:
            try:
                compressed_entry: 'MemoryEntry' = await self.compress_context(to_remove)
                storage_entries = [compressed_entry]
            except Exception as e:
                logging.error(f"Context compression failed during page_out_explicit: {e}")
                storage_entries = to_remove
        else:
            storage_entries = to_remove

        for entry in storage_entries:
            metadata: Dict[str, Any] = {
                "paged_out_explicitly": True,
                "importance": entry.importance,
            }
            if entry.emotional_state:
                metadata.update({
                    "pad_p": entry.emotional_state.pleasure,
                    "pad_a": entry.emotional_state.arousal,
                    "pad_d": entry.emotional_state.dominance
                })

            episodic_memory.store_event(
                text=entry.content,
                metadata=metadata,
                user_id=user_id
            )

        for eid in original_ids:
            working_memory.remove(eid, user_id=user_id)


    async def page_in_explicit(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: Optional[str], query: str, top_k: int = 5) -> None:
        """
        Recall relevant events from EpisodicMemory and load them into WorkingMemory.
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

    # --- Compatibility Aliases ---

    async def page_out(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: Optional[str], count: int = 1) -> None:
        await self.page_out_explicit(working_memory, episodic_memory, user_id, count)

    async def page_in(self, working_memory: 'WorkingMemory', episodic_memory: 'EpisodicMemory', user_id: Optional[str], query: str) -> None:
        await self.page_in_explicit(working_memory, episodic_memory, user_id, query)
