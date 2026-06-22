import logging
from typing import List, Optional
from magda_agent.memory.working import MemoryEntry

class ContextCompressorSelective:
    """
    Handles compression of working memory prior to semantic conversion, and supports selective
    retrieval from compressed blocks.
    """
    def __init__(self, llm_client=None) -> None:
        """
        Initializes the context compressor.

        Args:
            llm_client: Optional LLM client to use for summarization.
        """
        self.llm = llm_client

    async def compress_entries(self, entries: List[MemoryEntry]) -> MemoryEntry:
        """
        Compresses a list of memory entries into a single entry.

        Args:
            entries: A list of MemoryEntry objects to compress.

        Returns:
            A new MemoryEntry object representing the compressed content.
        """
        if not entries:
            raise ValueError("No entries to compress")

        combined_text = "\n".join(e.content for e in entries)
        compressed_text = combined_text

        if self.llm:
            try:
                logging.info("Compressing old working memory entries.")
                prompt = f"Summarize the following text, maintaining key facts:\n\n{combined_text}"
                messages = [
                    {"role": "system", "content": "You compress memory context. Return only the summary text."},
                    {"role": "user", "content": prompt}
                ]
                compressed_text = await self.llm.chat_completion(messages, temperature=0.3)
                compressed_text = compressed_text.strip()
            except Exception as e:
                logging.error(f"Compression failed: {e}")

        avg_importance = sum(e.importance for e in entries) / len(entries)
        first = entries[0]

        # Combine tags from all entries
        all_tags = set()
        for e in entries:
            if e.tags:
                all_tags.update(e.tags)

        return MemoryEntry(
            content=compressed_text,
            importance=avg_importance,
            emotional_state=first.emotional_state,
            tags=list(all_tags),
            user_id=first.user_id
        )


    async def compress_workflow(self, workflow_context: str, token_limit: int) -> str:
        """
        Compresses a long-running workflow context to fit within a specified token limit.
        Uses a heuristic of 4 characters per token if an exact tokenizer is unavailable.

        Args:
            workflow_context: The text of the workflow context.
            token_limit: The maximum allowed tokens.

        Returns:
            The original string if within limits, or a compressed summary.
        """
        char_limit = token_limit * 4
        if len(workflow_context) <= char_limit:
            return workflow_context

        if self.llm:
            try:
                logging.info(f"Compressing workflow context exceeding {token_limit} tokens.")
                prompt = f"Summarize the following workflow context to fit within {token_limit} tokens, retaining critical state and paths:\n\n{workflow_context}"
                messages = [
                    {"role": "system", "content": "You compress workflow context. Return only the summary text."},
                    {"role": "user", "content": prompt}
                ]
                compressed_text = await self.llm.chat_completion(messages, temperature=0.3)
                return compressed_text.strip()
            except Exception as e:
                logging.error(f"Workflow compression failed: {e}")
                # Fallback to simple truncation
                return workflow_context[:char_limit] + "... [TRUNCATED]"
        else:
            logging.warning("No LLM available for workflow compression, falling back to truncation.")
            return workflow_context[:char_limit] + "... [TRUNCATED]"

    def selective_retrieval(self, entries: List[MemoryEntry], query: str) -> List[MemoryEntry]:
        """
        Retrieves relevant entries based on a keyword match, to handle compressed blocks effectively.

        Args:
            entries: A list of MemoryEntry objects.
            query: The keyword query to filter by.

        Returns:
            A list of filtered MemoryEntry objects.
        """
        query_words = set(query.lower().split())
        filtered = []
        for e in entries:
            content_lower = e.content.lower()
            if any(word in content_lower for word in query_words):
                filtered.append(e)
        return filtered
