import logging
import time
from typing import Any, Dict, List, Optional
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

class ContextSelectiveRetrievalV3:
    """
    ContextSelectiveRetrievalV3 implements advanced context compression and selective
    retrieval for Context Engine. It dynamically prunes context windows based on
    token limits and prioritizes memories based on importance, query relevance, and temporal recency.
    """
    def __init__(self, llm: Optional[Any] = None, importance_threshold: float = 0.7) -> None:
        """
        Initializes the ContextSelectiveRetrievalV3.

        Args:
            llm: Optional language model integration for intelligent compaction.
            importance_threshold: Importance threshold above which memory entries are considered critical.
        """
        self.llm = llm
        self.importance_threshold = importance_threshold

    def get_token_length(self, entries: List[MemoryEntry]) -> int:
        """
        Calculates estimated token length of given memory entries.

        Args:
            entries: A list of MemoryEntry objects.

        Returns:
            The estimated token count.
        """
        total_words = sum(len(e.content.split()) for e in entries)
        return int(total_words * 1.3)

    def calculate_relevance(self, content: str, query: Optional[str]) -> float:
        """
        Calculates a simple relevance score based on word overlap.

        Args:
            content: The text content of the memory entry.
            query: The search query.

        Returns:
            A relevance score between 0.0 and 1.0.
        """
        if not query:
            return 0.0
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        if not query_words:
            return 0.0
        overlap = query_words.intersection(content_words)
        return len(overlap) / len(query_words)

    async def prune_context(self, entries: List[MemoryEntry], max_tokens: int, query: Optional[str] = None) -> List[MemoryEntry]:
        """
        Intelligently prunes the list of memory entries to fit within max_tokens.
        Prioritizes critical entries (importance >= threshold) and highly relevant entries.
        Compresses less relevant entries if LLM is available, or drops them.

        Args:
            entries: The full list of MemoryEntry items.
            max_tokens: The token limit for the context.
            query: Optional semantic or keyword query.

        Returns:
            The pruned and prioritized list of MemoryEntry items.
        """
        if not entries:
            return []

        current_tokens = self.get_token_length(entries)
        if current_tokens <= max_tokens:
            return entries

        logging.info(f"Context length ({current_tokens} tokens) exceeds limit ({max_tokens} tokens). Pruning dynamically...")

        # Calculate composite score for each entry
        scored_entries = []
        n_entries = len(entries)
        for idx, entry in enumerate(entries):
            importance = entry.importance
            relevance = self.calculate_relevance(entry.content, query)
            recency = idx / max(1, n_entries - 1)  # Later items are more recent

            # Composite prioritization score
            score = (importance * 0.5) + (relevance * 0.3) + (recency * 0.2)
            is_critical = importance >= self.importance_threshold or relevance >= 0.5
            scored_entries.append((score, is_critical, entry))

        # Separate critical and non-critical entries
        critical = [item[2] for item in scored_entries if item[1]]
        non_critical = [item[2] for item in scored_entries if not item[1]]

        # If critical entries alone exceed the limit, truncate the lowest scoring critical entries
        if self.get_token_length(critical) > max_tokens:
            logging.warning("Critical entries alone exceed limit, truncating lowest scored critical entries.")
            sorted_critical = sorted(
                [(item[0], item[2]) for item in scored_entries if item[1]],
                key=lambda x: x[0],
                reverse=True
            )
            retained_critical = []
            for _, entry in sorted_critical:
                if self.get_token_length(retained_critical + [entry]) <= max_tokens:
                    retained_critical.append(entry)
                else:
                    break
            return sorted(retained_critical, key=lambda e: entries.index(e))

        # We have some room left for non-critical entries
        retained_entries = list(critical)
        if not non_critical:
            return retained_entries

        # Try to compress non-critical entries if LLM is present
        if self.llm:
            try:
                non_critical_text = "\n".join(e.content for e in non_critical)
                prompt = (
                    f"Summarize the following non-critical background information concisely "
                    f"to fit within the remaining context limit:\n\n{non_critical_text}"
                )
                messages = [
                    {"role": "system", "content": "You are a precise context compression engine. Output only the summarized text."},
                    {"role": "user", "content": prompt}
                ]
                summary_content = await self.llm.chat_completion(messages, temperature=0.3)

                avg_importance = sum(e.importance for e in non_critical) / len(non_critical)
                first = non_critical[0]
                tags = list(set(tag for e in non_critical for tag in e.tags))

                summary_entry = MemoryEntry(
                    content=summary_content.strip(),
                    importance=avg_importance,
                    emotional_state=getattr(first, 'emotional_state', PADState(0, 0, 0)),
                    tags=tags,
                    user_id=first.user_id
                )

                # Check if adding the summary entry fits in token limit
                if self.get_token_length(retained_entries + [summary_entry]) <= max_tokens:
                    retained_entries.append(summary_entry)
                else:
                    logging.warning("Compressed summary too large, dropping non-critical entries.")
            except Exception as e:
                logging.error(f"LLM compression failed: {e}. Falling back to dynamic truncation.")
                self._truncate_non_critical(retained_entries, non_critical, max_tokens, scored_entries)
        else:
            self._truncate_non_critical(retained_entries, non_critical, max_tokens, scored_entries)

        # Re-sort to maintain original order of entries (except for generated summaries placed at the end/correct spots)
        return sorted(retained_entries, key=lambda e: entries.index(e) if e in entries else len(entries))

    def _truncate_non_critical(self, retained: List[MemoryEntry], non_critical: List[MemoryEntry], max_tokens: int, scored_entries: List[Any]) -> None:
        """Helper to append non-critical entries sorted by priority score until limit is reached."""
        # Sort non-critical entries by composite score descending
        sorted_non_critical = sorted(
            [(item[0], item[2]) for item in scored_entries if not item[1]],
            key=lambda x: x[0],
            reverse=True
        )
        for _, entry in sorted_non_critical:
            if self.get_token_length(retained + [entry]) <= max_tokens:
                retained.append(entry)
            else:
                break

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        ContextPlugin compliant compact lifecycle hook.
        Compacts the context items to fit within token bounds.
        """
        max_tokens = metadata.get("max_tokens", 1000)
        query = metadata.get("query", None)
        return await self.prune_context(context_items, max_tokens, query=query)
