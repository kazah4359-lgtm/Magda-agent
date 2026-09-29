import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Union

from magda_agent.llm_client import LLMClient
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.memory.semantic import SemanticMemory
from magda_agent.memory.working import MemoryEntry, WorkingMemory

logger = logging.getLogger(__name__)


class MemGPTVirtualContextSemanticCompressorV5:
    """
    MemGPT-inspired virtual context semantic compressor (V5).

    Periodically processes older episodic memory chunks and working memory entries,
    clustering them into concise semantic concepts/facts using an LLM.
    Extracted facts are stored into SemanticMemory, while original episodic events
    are decayed and working memory space is freed.
    """

    def __init__(
        self,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        working_memory: Optional[WorkingMemory] = None,
        llm_client: Optional[LLMClient] = None,
        llm: Optional[LLMClient] = None,
        batch_size: int = 5,
        min_events_threshold: int = 2,
        cluster_prompt: Optional[str] = None,
    ) -> None:
        self.episodic_memory = episodic_memory or EpisodicMemory(persist_directory=":memory:")
        self.semantic_memory = semantic_memory or SemanticMemory(persist_directory=":memory:")
        self.working_memory = working_memory
        self.llm = llm_client or llm or LLMClient()
        self.batch_size = batch_size
        self.min_events_threshold = min_events_threshold

        self.cluster_prompt = cluster_prompt or (
            "Analyze the following raw episodic memory events and extract the core semantic facts. "
            "Cluster related events and provide a concise list of key facts learned. "
            "Return only the extracted facts, one fact per line, formatted as clean bullet points or plain text.\n\n"
            "Events:\n{events}"
        )

        self._background_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self.total_compressions: int = 0
        self.total_facts_extracted: int = 0
        self.total_events_compacted: int = 0

    async def _call_llm(self, prompt: str) -> str:
        """Helper to invoke LLMClient via generate or chat_completion."""
        if hasattr(self.llm, "generate"):
            res = await self.llm.generate(prompt, temperature=0.2)
        elif hasattr(self.llm, "chat_completion"):
            res = await self.llm.chat_completion([{"role": "user", "content": prompt}], temperature=0.2)
        else:
            res = ""
        return res or ""

    def _parse_facts(self, raw_llm_response: str) -> List[str]:
        """Parse raw LLM response into clean individual fact strings."""
        if not raw_llm_response or raw_llm_response.startswith("Error:"):
            return []

        facts = []
        for line in raw_llm_response.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            # Remove leading bullet points or list numbers like "-", "*", "•", "1.", "1)"
            cleaned = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line_str).strip()
            if cleaned:
                facts.append(cleaned)
        return facts

    async def compress_episodic_to_semantic(
        self, user_id: Optional[int] = None, max_events: Optional[int] = None
    ) -> List[str]:
        """
        Retrieves raw episodic memory events, clusters them into semantic facts via LLM,
        stores the facts into SemanticMemory, decays original episodic events,
        and clears old corresponding working memory entries if present.

        Returns:
            List of extracted semantic fact strings.
        """
        limit = max_events or (self.batch_size * 2)
        events = self.episodic_memory.get_all_events(user_id=user_id, include_decayed=False, limit=limit)

        if not events or len(events) < self.min_events_threshold:
            logger.info(
                f"VirtualCompressionV5: Insufficient events ({len(events)}) for threshold ({self.min_events_threshold})."
            )
            return []

        events_to_process = events[: self.batch_size]
        event_texts = [f"- {e['text']}" for e in events_to_process]
        events_formatted = "\n".join(event_texts)

        prompt = self.cluster_prompt.format(events=events_formatted)

        try:
            raw_response = await self._call_llm(prompt)
            facts = self._parse_facts(raw_response)

            if not facts:
                logger.warning("VirtualCompressionV5: LLM returned no valid facts.")
                return []

            # Store facts in semantic memory
            for fact in facts:
                self.semantic_memory.store_fact(
                    fact,
                    metadata={"source": "virtual_compression_v5", "events_count": len(events_to_process)},
                    user_id=user_id,
                )

            # Decay processed episodic events
            for event in events_to_process:
                self.episodic_memory.decay_event(event["id"])

            # Optionally free space in working memory if provided
            if self.working_memory is not None:
                await self._free_working_memory_space(user_id=user_id)

            self.total_compressions += 1
            self.total_facts_extracted += len(facts)
            self.total_events_compacted += len(events_to_process)

            logger.info(
                f"VirtualCompressionV5: Compressed {len(events_to_process)} events into {len(facts)} semantic facts."
            )
            return facts

        except Exception as e:
            logger.error(f"VirtualCompressionV5 error: {e}", exc_info=True)
            return []

    async def _free_working_memory_space(self, user_id: Optional[int] = None) -> None:
        """Free working memory space by removing or compacting older entries."""
        try:
            if hasattr(self.working_memory, "get_entries"):
                entries = self.working_memory.get_entries(user_id=user_id)
                if len(entries) > 2:
                    # Keep newest entries, remove older ones to free space
                    if hasattr(self.working_memory, "_entries_by_user"):
                        uid = user_id if user_id is not None else -1
                        if uid in self.working_memory._entries_by_user:
                            self.working_memory._entries_by_user[uid] = entries[-2:]
        except Exception as e:
            logger.warning(f"Error freeing working memory space: {e}")

    async def run_periodic_compression(
        self, interval_seconds: float = 60.0, user_id: Optional[int] = None
    ) -> None:
        """Periodically run virtual compression in an async loop."""
        self._running = True
        try:
            while self._running:
                await self.compress_episodic_to_semantic(user_id=user_id)
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            self._running = False
            logger.info("Virtual compression periodic loop cancelled.")

    def start_background_loop(
        self, interval_seconds: float = 60.0, user_id: Optional[int] = None
    ) -> asyncio.Task:
        """Starts background periodic compression task."""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(
                self.run_periodic_compression(interval_seconds=interval_seconds, user_id=user_id)
            )
        return self._background_task

    async def stop_background_loop(self) -> None:
        """Stops background periodic compression task."""
        self._running = False
        if self._background_task is not None and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None


MemGPTVirtualCompressionV5 = MemGPTVirtualContextSemanticCompressorV5
