import logging
from typing import Any, Dict, List, Optional
import math
import heapq

logger = logging.getLogger(__name__)

class LargeContextWindow:
    """Handles efficient memory indexing for up to 1M tokens, modeled after Claude SDK."""

    def __init__(self, max_tokens: int = 1000000) -> None:
        """
        Initialize the large context window.

        Args:
            max_tokens: The maximum number of tokens to support (default 1M).
        """
        self.max_tokens = max_tokens
        self.chunks: List[Dict[str, Any]] = []
        self.current_tokens: int = 0
        self.index: Dict[str, List[int]] = {}

    def _rebuild_index(self) -> None:
        self.index.clear()
        for idx, chunk in enumerate(self.chunks):
            words = set(self._tokenize(chunk["content"]))
            for word in words:
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append(idx)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer for indexing."""
        return text.lower().split()

    def add_chunk(self, content: str, tokens: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a chunk of content to the context window and index it.

        Args:
            content: The string content to add.
            tokens: The estimated number of tokens in the content.
            metadata: Optional metadata for the chunk.
        """
        while self.current_tokens + tokens > self.max_tokens and self.chunks:
            # Evict the oldest chunk
            oldest = self.chunks.pop(0)
            self.current_tokens -= oldest["tokens"]
            # We don't remove from inverted index for performance, as chunk indices will shift.
            # Instead, we should ideally rebuild index or use UUIDs.
            # For simplicity, we just clear and rebuild the whole index if we ever evict.
            # A more robust solution would track valid ranges.

            # Since index uses list indices, popping(0) shifts all subsequent indices by -1.
            # We must rebuild the index to avoid corruption.
            self._rebuild_index()

        chunk_idx = len(self.chunks)
        self.chunks.append({
            "content": content,
            "tokens": tokens,
            "metadata": metadata or {}
        })
        self.current_tokens += tokens

        # Build inverted index
        words = set(self._tokenize(content))
        for word in words:
            if word not in self.index:
                self.index[word] = []
            self.index[word].append(chunk_idx)

    def retrieve(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks efficiently from the large token window using TF-IDF style scoring.

        Args:
            query: The query string to search for.
            max_results: The maximum number of results to return.

        Returns:
            A list of relevant chunks matching the query.
        """
        query_words = self._tokenize(query)
        if not query_words:
            return []

        # Score chunks
        scores: Dict[int, float] = {}
        num_docs = len(self.chunks)

        for word in query_words:
            if word in self.index:
                doc_freq = len(self.index[word])
                # Simple IDF
                idf = math.log((num_docs + 1) / (doc_freq + 1)) + 1
                for chunk_idx in self.index[word]:
                    if chunk_idx not in scores:
                        scores[chunk_idx] = 0.0
                    # TF is 1 for simplicity (boolean presence)
                    scores[chunk_idx] += idf

        # Get top chunks
        top_indices = heapq.nlargest(max_results, scores, key=scores.get)
        return [self.chunks[idx] for idx in top_indices]
