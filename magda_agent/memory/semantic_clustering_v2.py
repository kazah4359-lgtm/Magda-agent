"""
SemanticClusteringV2 module.

Inspired by MemGPT/Letta patterns: Context Engine plugin that semantically clusters
and compresses episodic memory records based on semantic tags and content similarity,
effectively reducing working context token size while offloading evicted items to long-term/episodic memory stores.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Union, Tuple


def _extract_text(item: Any) -> str:
    """Extract text or content string from string, dict, or entry object."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("content") or item.get("text") or item)
    for attr in ("content", "text"):
        if hasattr(item, attr):
            val = getattr(item, attr)
            if isinstance(val, str):
                return val
    return str(item)


def _estimate_tokens(text_or_items: Any) -> int:
    """Estimate token count for a text string, entry item, or list of items."""
    if isinstance(text_or_items, list):
        return sum(_estimate_tokens(i) for i in text_or_items)
    text = _extract_text(text_or_items)
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.3))


def _extract_tags(item: Any) -> List[str]:
    """
    Extract semantic tags from item (dict, object, or text content).
    Checks item.tags, item.metadata.get('tags'), dict keys, and inline hashtags/#tags or [tag] patterns.
    """
    tags: List[str] = []

    if isinstance(item, dict):
        raw_tags = item.get("tags")
        if not raw_tags and isinstance(item.get("metadata"), dict):
            raw_tags = item["metadata"].get("tags")
        if isinstance(raw_tags, list):
            tags.extend([str(t).strip().lower() for t in raw_tags])
        elif isinstance(raw_tags, str):
            tags.append(raw_tags.strip().lower())
    elif hasattr(item, "tags") and getattr(item, "tags"):
        raw_tags = getattr(item, "tags")
        if isinstance(raw_tags, list):
            tags.extend([str(t).strip().lower() for t in raw_tags])
        elif isinstance(raw_tags, str):
            tags.append(raw_tags.strip().lower())
    elif hasattr(item, "metadata") and isinstance(getattr(item, "metadata"), dict):
        raw_tags = getattr(item, "metadata").get("tags")
        if isinstance(raw_tags, list):
            tags.extend([str(t).strip().lower() for t in raw_tags])
        elif isinstance(raw_tags, str):
            tags.append(raw_tags.strip().lower())

    text = _extract_text(item)
    # Extract #hashtags
    hashtags = re.findall(r"#(\w+)", text)
    tags.extend([h.lower() for h in hashtags])

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            deduped.append(tag)

    return deduped if deduped else ["general"]


class MemoryClusterEntry:
    """Represents a compressed/summarized cluster of episodic memories."""

    def __init__(self, content: str, tags: List[str], importance: float = 0.5, items_count: int = 1) -> None:
        self.content = content
        self.tags = tags
        self.importance = importance
        self.items_count = items_count

    def __repr__(self) -> str:
        return f"<MemoryClusterEntry tags={self.tags} items={self.items_count} content={self.content[:30]}...>"


class SemanticClusteringV2:
    """
    Context Engine plugin that clusters related episodic memories semantically using semantic tags
    and compresses working context to achieve significant token reduction.
    """

    def __init__(
        self,
        episodic_memory: Optional[Any] = None,
        llm: Optional[Any] = None,
        similarity_threshold: float = 0.6,
        max_tokens: int = 1000,
        max_items: Optional[int] = None,
        mode: str = "summarize",
    ) -> None:
        """
        Initialize the SemanticClusteringV2 plugin.

        Args:
            episodic_memory: Optional memory store to offload original uncompressed entries.
            llm: Optional LLM client for intelligent summary generation.
            similarity_threshold: Threshold for semantic clustering.
            max_tokens: Target token limit for context compaction.
            max_items: Optional max item count limit.
            mode: Compaction mode ("summarize" or "evict").
        """
        self.episodic_memory = episodic_memory
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.max_items = max_items
        self.mode = mode
        self.config: Dict[str, Any] = {}

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Bootstrap lifecycle hook. Initializes plugin parameters from config."""
        self.config = config
        if "similarity_threshold" in config:
            self.similarity_threshold = float(config["similarity_threshold"])
        if "max_tokens" in config:
            self.max_tokens = int(config["max_tokens"])
        if "max_items" in config:
            self.max_items = config["max_items"]
        if "mode" in config:
            self.mode = str(config["mode"])
        if "episodic_memory" in config:
            self.episodic_memory = config["episodic_memory"]
        if "llm" in config:
            self.llm = config["llm"]

    def extract_tags(self, item: Any) -> List[str]:
        """Extract semantic tags from item."""
        return _extract_tags(item)

    def extract_text(self, item: Any) -> str:
        """Extract text content from item."""
        return _extract_text(item)

    def estimate_tokens(self, text_or_items: Any) -> int:
        """Estimate token count for a text string or list of items."""
        return _estimate_tokens(text_or_items)

    def cluster_by_tags(self, memories: List[Any]) -> Dict[str, List[Any]]:
        """
        Group episodic memories into clusters keyed by semantic tags.
        Items sharing a tag are grouped together.
        """
        clusters: Dict[str, List[Any]] = {}
        for item in memories:
            tags = self.extract_tags(item)
            primary_tag = tags[0] if tags else "general"
            if primary_tag not in clusters:
                clusters[primary_tag] = []
            clusters[primary_tag].append(item)
        return clusters

    async def _summarize_cluster(self, tag: str, cluster: List[Any]) -> Any:
        """Summarize a cluster of memories into a single compressed entry."""
        if len(cluster) == 1:
            return cluster[0]

        texts = [self.extract_text(item) for item in cluster]
        combined_text = "\n".join(texts)

        if self.llm is not None:
            prompt = f"Summarize the following episodic memories related to tag '{tag}':\n{combined_text}"
            try:
                summary_text = await self.llm.chat_completion(
                    [
                        {"role": "system", "content": "You compress and summarize episodic memory clusters concisely."},
                        {"role": "user", "content": prompt},
                    ]
                )
                summary = str(summary_text).strip()
            except Exception as e:
                logging.error(f"Error calling LLM for semantic cluster summary: {e}")
                summary = f"[Cluster '{tag}' ({len(cluster)} events)]: {texts[0]} ... [and {len(cluster) - 1} more]"
        else:
            summary = f"[Cluster '{tag}' ({len(cluster)} events)]: {texts[0]} ... [and {len(cluster) - 1} more]"

        all_tags = set()
        for item in cluster:
            for t in self.extract_tags(item):
                all_tags.add(t)

        return MemoryClusterEntry(
            content=summary,
            tags=list(all_tags),
            importance=0.5,
            items_count=len(cluster),
        )

    def _offload_to_episodic_store(self, items: List[Any], store: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Offload uncompressed or evicted memory entries to the provided episodic memory store."""
        user_id = metadata.get("user_id") if metadata else None
        for item in items:
            text = self.extract_text(item)
            tags = self.extract_tags(item)
            if hasattr(store, "store_event"):
                store.store_event(text, metadata={"tags": tags, "clustered": True}, user_id=user_id)
            elif hasattr(store, "add_memory"):
                store.add_memory(text)
            elif hasattr(store, "append"):
                store.append(text)

    async def cluster_and_compress(
        self,
        memories: List[Any],
        max_tokens: Optional[int] = None,
        max_items: Optional[int] = None,
        episodic_store: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Main method to cluster episodic memories by semantic tags and compress working context.
        Verifies token reduction before and after compression.
        """
        if not memories:
            return []

        limit_tokens = max_tokens if max_tokens is not None else self.max_tokens
        limit_items = max_items if max_items is not None else self.max_items
        store = episodic_store or self.episodic_memory

        initial_tokens = self.estimate_tokens(memories)

        # Group by semantic tags
        clusters = self.cluster_by_tags(memories)

        compressed_items: List[Any] = []
        offloaded_items: List[Any] = []

        for tag, items in clusters.items():
            if len(items) > 1:
                offloaded_items.extend(items)
                if self.mode == "evict":
                    # Keep only the latest entry in the cluster
                    compressed_items.append(items[-1])
                else:
                    # Summarize cluster
                    summary_entry = await self._summarize_cluster(tag, items)
                    compressed_items.append(summary_entry)
            else:
                compressed_items.extend(items)

        if store is not None and offloaded_items:
            self._offload_to_episodic_store(offloaded_items, store, metadata)

        # Enforce token limit if still exceeding
        while (
            compressed_items
            and len(compressed_items) > 1
            and (
                (limit_items is not None and len(compressed_items) > limit_items)
                or (limit_tokens is not None and self.estimate_tokens(compressed_items) > limit_tokens)
            )
        ):
            evicted = compressed_items.pop(0)
            if store is not None:
                self._offload_to_episodic_store([evicted], store, metadata)

        final_tokens = self.estimate_tokens(compressed_items)
        logging.info(
            f"SemanticClusteringV2 compressed context: {len(memories)} -> {len(compressed_items)} items, "
            f"{initial_tokens} -> {final_tokens} estimated tokens."
        )

        return compressed_items

    # --- ContextPlugin Protocol Methods ---

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Ingest hook. Tags or pre-processes content."""
        tags = metadata.get("tags", [])
        if tags:
            tag_str = ",".join(tags)
            return f"[Tags:{tag_str}] {content}"
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble hook. Formats context items into prompt string."""
        if not context_items:
            return ""
        lines = []
        for item in context_items:
            text = self.extract_text(item)
            tags = self.extract_tags(item)
            lines.append(f"[{','.join(tags)}] {text}")
        return "\n".join(lines)

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Context Engine lifecycle compact hook."""
        max_tokens = metadata.get("max_tokens", metadata.get("limit_tokens", self.max_tokens))
        max_items = metadata.get("limit", metadata.get("max_items", self.max_items))
        store = metadata.get("episodic_memory") or metadata.get("memory_store") or self.episodic_memory

        return await self.cluster_and_compress(
            context_items,
            max_tokens=max_tokens,
            max_items=max_items,
            episodic_store=store,
            metadata=metadata,
        )

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
