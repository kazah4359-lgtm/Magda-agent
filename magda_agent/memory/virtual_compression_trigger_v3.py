import logging
from typing import List, Optional, Tuple, Union

from magda_agent.llm_client import LLMClient
from magda_agent.memory.working import MemoryEntry, WorkingMemory
from magda_agent.memory.virtual_context_compression_v1 import OpenClawVirtualContextCompressionHook

logger = logging.getLogger(__name__)


class MemGPTVirtualCompressionTriggerV3:
    """
    A subagent trigger responsible for periodic and threshold-based context compression
    of long dialogue history in working memory, inspired by MemGPT dynamic virtual context management.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        token_threshold: int = 2000,
        compression_hook: Optional[OpenClawVirtualContextCompressionHook] = None,
    ) -> None:
        """
        Initialize the virtual compression trigger V3.

        Args:
            llm: Optional LLMClient used for context summarization.
            token_threshold: The maximum estimated token count before triggering context compression.
            compression_hook: Optional custom compression hook instance. If None, one will be created.
        """
        self.llm = llm
        self.token_threshold = token_threshold
        self.compression_hook = compression_hook or OpenClawVirtualContextCompressionHook(
            llm=llm, token_threshold=token_threshold
        )
        self.trigger_count: int = 0
        self.total_compressions: int = 0
        self.last_compressed_tokens: int = 0

    def estimate_tokens(self, entries: List[MemoryEntry]) -> int:
        """
        Estimate token length for a list of MemoryEntry instances (~1.3 tokens per word).

        Args:
            entries: List of memory entries.

        Returns:
            Estimated total token count.
        """
        if not entries:
            return 0
        total_words = sum(len(e.content.split()) for e in entries)
        return int(total_words * 1.3)

    def _extract_entries(self, entries_or_memory: Union[List[MemoryEntry], WorkingMemory], user_id: Optional[int] = None) -> List[MemoryEntry]:
        if isinstance(entries_or_memory, WorkingMemory):
            if user_id is not None:
                return entries_or_memory.get_entries(user_id=user_id)
            all_entries = entries_or_memory.get_all_entries()
            return all_entries if all_entries else entries_or_memory.get_entries()
        return entries_or_memory

    def should_trigger(
        self, entries_or_memory: Union[List[MemoryEntry], WorkingMemory], user_id: Optional[int] = None
    ) -> bool:
        """
        Checks if current memory content exceeds the token threshold.

        Args:
            entries_or_memory: List of MemoryEntry items or WorkingMemory instance.
            user_id: Optional user ID filter if entries_or_memory is a WorkingMemory instance.

        Returns:
            True if compression should be triggered, False otherwise.
        """
        entries = self._extract_entries(entries_or_memory, user_id=user_id)
        token_count = self.estimate_tokens(entries)
        return token_count > self.token_threshold

    async def check_and_compress(
        self, entries_or_memory: Union[List[MemoryEntry], WorkingMemory], user_id: Optional[int] = None
    ) -> Tuple[bool, List[MemoryEntry]]:
        """
        Evaluates memory and performs context compression if threshold is exceeded.

        Args:
            entries_or_memory: List of MemoryEntry items or WorkingMemory instance.
            user_id: Optional user ID filter if entries_or_memory is a WorkingMemory instance.

        Returns:
            Tuple of (was_compressed: bool, updated_entries: List[MemoryEntry]).
        """
        self.trigger_count += 1

        is_working_memory = isinstance(entries_or_memory, WorkingMemory)
        entries = self._extract_entries(entries_or_memory, user_id=user_id)

        if not self.should_trigger(entries):
            return False, entries

        initial_tokens = self.estimate_tokens(entries)
        logger.info(
            f"MemGPTVirtualCompressionTriggerV3: Triggering compression. "
            f"Tokens ({initial_tokens}) > threshold ({self.token_threshold})"
        )

        compressed_entries = await self.compression_hook.compress(entries)
        self.total_compressions += 1
        self.last_compressed_tokens = initial_tokens

        if is_working_memory:
            u_id = user_id if user_id is not None else (entries[0].user_id if entries and entries[0].user_id is not None else -1)
            entries_or_memory._entries_by_user[u_id] = compressed_entries.copy()

        return True, compressed_entries
