"""
SemanticClusterPagerV2 module.

Implements MemGPT-inspired semantic clustering and virtual context paging on
episodic memory records using local embeddings and concurrent cluster eviction/summarization.
"""

import asyncio
import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r"\w+", text.lower())


def default_cosine_similarity(text1: str, text2: str) -> float:
    """Computes a localized lexical cosine similarity based on word frequencies."""
    words1 = _tokenize(text1)
    words2 = _tokenize(text2)
    if not words1 or not words2:
        return 0.0

    freq1: Dict[str, int] = {}
    freq2: Dict[str, int] = {}
    for w in words1:
        freq1[w] = freq1.get(w, 0) + 1
    for w in words2:
        freq2[w] = freq2.get(w, 0) + 1

    all_words = set(freq1.keys()).union(freq2.keys())
    dot_product = sum(freq1.get(w, 0) * freq2.get(w, 0) for w in all_words)

    mag1 = math.sqrt(sum(val**2 for val in freq1.values()))
    mag2 = math.sqrt(sum(val**2 for val in freq2.values()))

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    return dot_product / (mag1 * mag2)


def vector_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


def _get_entry_text(entry: Any) -> str:
    """Extract text/content from string, dict, or object entry."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("content") or entry.get("text") or str(entry)
    for attr in ("content", "text"):
        if hasattr(entry, attr):
            val = getattr(entry, attr)
            if isinstance(val, str):
                return val
    return str(entry)


def _estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.3))


class SemanticClusterPagerV2:
    """
    Context Engine plugin / Virtual Context Pager using local embeddings and clustering.

    Identifies redundant thematic clusters across episodic memories, and evicts or
    summarizes related records concurrently when token or item count thresholds are hit.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_tokens: int = 1000,
        max_items: Optional[int] = None,
        mode: str = "evict",  # "evict" or "summarize"
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        summarize_fn: Optional[Callable[[List[Any]], Any]] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.max_items = max_items
        self.mode = mode
        self.embedding_fn = embedding_fn
        self.similarity_fn = similarity_fn
        self.summarize_fn = summarize_fn
        self.llm = llm

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Initialize plugin configuration."""
        if "similarity_threshold" in config:
            self.similarity_threshold = config["similarity_threshold"]
        if "max_tokens" in config:
            self.max_tokens = config["max_tokens"]
        if "max_items" in config:
            self.max_items = config["max_items"]
        if "mode" in config:
            self.mode = config["mode"]
        if "embedding_fn" in config:
            self.embedding_fn = config["embedding_fn"]
        if "similarity_fn" in config:
            self.similarity_fn = config["similarity_fn"]

    def calculate_similarity(self, item1: Any, item2: Any) -> float:
        """Compute similarity between two items using embedding_fn or similarity_fn."""
        text1 = _get_entry_text(item1)
        text2 = _get_entry_text(item2)

        if self.embedding_fn is not None:
            vec1 = self.embedding_fn(text1)
            vec2 = self.embedding_fn(text2)
            return vector_cosine_similarity(vec1, vec2)

        if self.similarity_fn is not None:
            return self.similarity_fn(text1, text2)

        return default_cosine_similarity(text1, text2)

    def cluster_memories(self, memories: List[Any]) -> List[List[Any]]:
        """
        Group memories into thematic clusters based on semantic similarity.
        Returns a list of clusters, where each cluster is a list of memory items.
        """
        if not memories:
            return []

        n = len(memories)
        visited = [False] * n
        clusters: List[List[Any]] = []

        # Build adjacency graph
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_similarity(memories[i], memories[j])
                if sim >= self.similarity_threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        # Connected components discovery
        for i in range(n):
            if not visited[i]:
                cluster_indices = []
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    cluster_indices.append(curr)
                    for neighbor in adj[curr]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                clusters.append([memories[idx] for idx in cluster_indices])

        return clusters

    def total_tokens(self, memories: List[Any]) -> int:
        """Calculate estimated total tokens in memories."""
        return sum(_estimate_tokens(_get_entry_text(m)) for m in memories)

    def is_threshold_exceeded(
        self,
        memories: List[Any],
        max_tokens: Optional[int] = None,
        max_items: Optional[int] = None,
    ) -> bool:
        """Check if total tokens or item count exceeds specified or configured limits."""
        limit_tokens = max_tokens if max_tokens is not None else self.max_tokens
        limit_items = max_items if max_items is not None else self.max_items

        if limit_items is not None and len(memories) > limit_items:
            return True
        if limit_tokens is not None and self.total_tokens(memories) > limit_tokens:
            return True
        return False

    async def _summarize_cluster(self, cluster: List[Any]) -> Any:
        """Asynchronously summarize a single cluster of items."""
        if len(cluster) == 1:
            return cluster[0]

        if self.summarize_fn is not None:
            if asyncio.iscoroutinefunction(self.summarize_fn):
                return await self.summarize_fn(cluster)
            return self.summarize_fn(cluster)

        texts = [_get_entry_text(m) for m in cluster]
        combined = "\n".join(texts)

        if self.llm is not None:
            prompt = f"Summarize the following related memory records into a single concise record:\n{combined}"
            try:
                res = await self.llm.chat_completion(
                    [{"role": "user", "content": prompt}]
                )
                if isinstance(cluster[0], dict):
                    return {"content": str(res).strip(), "summarized": True}
                return f"[Cluster Summary]: {str(res).strip()}"
            except Exception as e:
                logging.error(f"Error calling LLM for cluster summary: {e}")

        # Fallback local summary
        summary_text = f"[Summary of {len(cluster)} records]: {texts[0]}"
        if isinstance(cluster[0], dict):
            res_dict = dict(cluster[0])
            res_dict["content"] = summary_text
            res_dict["summarized_count"] = len(cluster)
            return res_dict
        return summary_text

    async def _evict_or_compress_cluster(self, cluster: List[Any]) -> List[Any]:
        """
        Evict or compress a single cluster concurrently.
        For redundant cluster (>1 items):
        - If mode == 'summarize': return [summary]
        - If mode == 'evict': return [cluster[-1]] (keep most recent)
        For single item cluster (size 1): return cluster as is.
        """
        if len(cluster) <= 1:
            return cluster

        if self.mode == "summarize":
            summary = await self._summarize_cluster(cluster)
            return [summary]
        else:
            # Evict redundant older items in cluster, keep the last one
            return [cluster[-1]]

    async def compact_clusters_concurrently(self, clusters: List[List[Any]]) -> List[Any]:
        """Concurrently evict or summarize clusters."""
        tasks = [self._evict_or_compress_cluster(cluster) for cluster in clusters]
        results = await asyncio.gather(*tasks)
        compacted: List[Any] = []
        for res in results:
            compacted.extend(res)
        return compacted

    async def page_and_compact(
        self,
        memories: List[Any],
        max_tokens: Optional[int] = None,
        max_items: Optional[int] = None,
    ) -> List[Any]:
        """
        Main entry point for paging and compacting memory records based on semantic clustering.
        If thresholds are exceeded, identifies clusters and concurrently evicts or summarizes redundant ones.
        """
        if not memories:
            return []

        if not self.is_threshold_exceeded(memories, max_tokens, max_items):
            return list(memories)

        clusters = self.cluster_memories(memories)
        compacted = await self.compact_clusters_concurrently(clusters)

        limit_tokens = max_tokens if max_tokens is not None else self.max_tokens
        limit_items = max_items if max_items is not None else self.max_items

        while (
            compacted
            and len(compacted) > 1
            and (
                (limit_items is not None and len(compacted) > limit_items)
                or (limit_tokens is not None and self.total_tokens(compacted) > limit_tokens)
            )
        ):
            compacted.pop(0)

        return compacted

    # --- ContextPlugin protocol implementation ---

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        return "\n".join([_get_entry_text(item) for item in context_items])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Context Engine lifecycle hook for compacting context items."""
        max_tokens = metadata.get("max_tokens", self.max_tokens)
        max_items = metadata.get("max_items", metadata.get("limit", self.max_items))
        return await self.page_and_compact(context_items, max_tokens=max_tokens, max_items=max_items)

    def before_retrieval(self, query: str, user_id: int) -> str:
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        return context

    def before_write(self, context: Any, user_id: int) -> Any:
        return context

    def after_write(self, context: Any, user_id: int) -> None:
        pass

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        pass

    async def pre_process(self, content: str, metadata: Dict[str, Any]) -> str:
        return content

    async def post_process(self, response: str, metadata: Dict[str, Any]) -> str:
        return response
