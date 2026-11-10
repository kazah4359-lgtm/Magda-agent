"""
SemanticPruningHookV3 module.

Inspired by MemGPT/Letta context patterns: Implements a semantic pruning hook
and ContextPlugin for virtual context, evaluating short-term entries dynamically
against configurable exponential or linear decay metrics, importance, and semantic relevance,
and evicting decayed entries while optionally offloading them to episodic memory stores.
"""

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


def _extract_text(item: Any) -> str:
    """Extract plain text or content string from item (string, dict, or object)."""
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


def _extract_timestamp(item: Any, default_time: Optional[float] = None) -> float:
    """Extract creation or last updated timestamp from item, fallback to default_time or time.time()."""
    now = default_time if default_time is not None else time.time()
    if isinstance(item, dict):
        for key in ("timestamp", "created_at", "time", "date"):
            if key in item and isinstance(item[key], (int, float)):
                return float(item[key])
        if "metadata" in item and isinstance(item["metadata"], dict):
            for key in ("timestamp", "created_at", "time"):
                if key in item["metadata"] and isinstance(item["metadata"][key], (int, float)):
                    return float(item["metadata"][key])
    elif hasattr(item, "timestamp") and isinstance(getattr(item, "timestamp"), (int, float)):
        return float(getattr(item, "timestamp"))
    elif hasattr(item, "created_at") and isinstance(getattr(item, "created_at"), (int, float)):
        return float(getattr(item, "created_at"))
    elif hasattr(item, "metadata") and isinstance(getattr(item, "metadata"), dict):
        meta = getattr(item, "metadata")
        for key in ("timestamp", "created_at", "time"):
            if key in meta and isinstance(meta[key], (int, float)):
                return float(meta[key])
    return now


def _extract_importance(item: Any, default: float = 0.5) -> float:
    """Extract intrinsic importance score (0.0 to 1.0) from item."""
    if isinstance(item, dict):
        if "importance" in item and isinstance(item["importance"], (int, float)):
            return float(item["importance"])
        if "metadata" in item and isinstance(item["metadata"], dict):
            if "importance" in item["metadata"] and isinstance(item["metadata"]["importance"], (int, float)):
                return float(item["metadata"]["importance"])
    elif hasattr(item, "importance") and isinstance(getattr(item, "importance"), (int, float)):
        return float(getattr(item, "importance"))
    elif hasattr(item, "metadata") and isinstance(getattr(item, "metadata"), dict):
        meta = getattr(item, "metadata")
        if "importance" in meta and isinstance(meta["importance"], (int, float)):
            return float(meta["importance"])
    return default


def _extract_tags(item: Any) -> List[str]:
    """Extract semantic tags from item."""
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
    hashtags = re.findall(r"#(\w+)", text)
    tags.extend([h.lower() for h in hashtags])

    deduped: List[str] = []
    seen: Set[str] = set()
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped if deduped else ["general"]


@dataclass
class DecayConfigV3:
    """
    Configuration parameters for semantic pruning and context decay metrics.
    """
    decay_rate: float = 0.05
    half_life_seconds: float = 3600.0  # 1 hour default half-life
    min_utility_threshold: float = 0.35  # Entries below this composite score are pruned
    decay_mode: str = "exponential"  # "exponential" or "linear"
    weight_recency: float = 0.4
    weight_importance: float = 0.3
    weight_relevance: float = 0.3


class SemanticPruningHookV3:
    """
    Dynamic semantic pruning hook and Context Engine plugin for virtual context.
    Evaluates short-term working context entries using decay factors, importance,
    and semantic relevance, pruning decayed items and offloading them to episodic stores.
    """

    def __init__(
        self,
        config: Optional[DecayConfigV3] = None,
        episodic_memory: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        max_context_items: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Initialize the SemanticPruningHookV3.

        Args:
            config: Decay and utility threshold configuration.
            episodic_memory: Optional long-term / episodic memory store for offloading pruned entries.
            llm_client: Optional LLM client for semantic relevance scoring.
            max_context_items: Optional maximum number of items to keep in context.
            max_tokens: Optional token ceiling for context.
        """
        self.config = config or DecayConfigV3()
        self.episodic_memory = episodic_memory
        self.llm_client = llm_client
        self.max_context_items = max_context_items
        self.max_tokens = max_tokens
        self.plugin_config: Dict[str, Any] = {}
        self.pruning_history: List[Dict[str, Any]] = []

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Bootstrap lifecycle hook. Injects plugin configuration."""
        self.plugin_config = config
        if "decay_rate" in config:
            self.config.decay_rate = float(config["decay_rate"])
        if "half_life_seconds" in config:
            self.config.half_life_seconds = float(config["half_life_seconds"])
        if "min_utility_threshold" in config:
            self.config.min_utility_threshold = float(config["min_utility_threshold"])
        if "decay_mode" in config:
            self.config.decay_mode = str(config["decay_mode"])
        if "weight_recency" in config:
            self.config.weight_recency = float(config["weight_recency"])
        if "weight_importance" in config:
            self.config.weight_importance = float(config["weight_importance"])
        if "weight_relevance" in config:
            self.config.weight_relevance = float(config["weight_relevance"])
        if "max_context_items" in config:
            self.max_context_items = config["max_context_items"]
        if "max_tokens" in config:
            self.max_tokens = config["max_tokens"]
        if "episodic_memory" in config:
            self.episodic_memory = config["episodic_memory"]
        if "llm_client" in config:
            self.llm_client = config["llm_client"]
        logger.info("SemanticPruningHookV3 successfully bootstrapped.")

    def compute_recency_score(self, item_time: float, current_time: Optional[float] = None) -> float:
        """
        Computes time-decayed recency score between 0.0 and 1.0.
        Uses exponential or linear decay according to config.
        """
        now = current_time if current_time is not None else time.time()
        elapsed = max(0.0, now - item_time)

        if self.config.decay_mode == "linear":
            # Linear decay over 2 * half_life_seconds (or configured horizon)
            horizon = max(1.0, 2.0 * self.config.half_life_seconds)
            recency = max(0.0, 1.0 - (elapsed / horizon))
            return recency
        else:
            # Exponential decay: score = exp(-ln(2) * elapsed / half_life)
            half_life = max(1.0, self.config.half_life_seconds)
            lambda_param = math.log(2.0) / half_life
            return math.exp(-lambda_param * elapsed)

    def compute_semantic_relevance(self, item_text: str, query: Optional[str] = None) -> float:
        """
        Calculates semantic relevance score (0.0 to 1.0) using token overlap and keyword matches.
        """
        if not query or not query.strip():
            return 1.0  # Default to full relevance if no query constraint

        clean_query = query.lower()
        clean_text = item_text.lower()

        # Token set overlap (Jaccard-like or match fraction)
        query_words = set(re.findall(r"\w+", clean_query))
        text_words = set(re.findall(r"\w+", clean_text))

        if not query_words:
            return 1.0

        overlap = query_words.intersection(text_words)
        relevance = len(overlap) / len(query_words)
        return min(1.0, max(0.0, relevance))

    async def compute_semantic_relevance_llm(self, item_text: str, query: str) -> float:
        """
        Computes semantic relevance score using LLM when available, falling back to heuristic.
        """
        if not self.llm_client or not query:
            return self.compute_semantic_relevance(item_text, query)

        prompt = (
            f"Rate the relevance of the following memory entry to the task or query on a scale from 0.0 to 1.0.\n"
            f"Query: {query}\n"
            f"Memory Entry: {item_text}\n"
            f"Respond only with a single float number between 0.0 and 1.0."
        )
        try:
            response = await self.llm_client.chat_completion(
                [
                    {"role": "system", "content": "You are an evaluator scoring memory relevance."},
                    {"role": "user", "content": prompt},
                ]
            )
            val = float(re.search(r"([0-1](?:\.\d+)?)", str(response)).group(1))
            return min(1.0, max(0.0, val))
        except Exception as e:
            logger.debug(f"LLM relevance scoring failed, using heuristic: {e}")
            return self.compute_semantic_relevance(item_text, query)

    def calculate_utility(
        self,
        item: Any,
        query: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates composite retention utility score for an item.

        Returns:
            Tuple of (composite_utility, metrics_dict).
        """
        item_time = _extract_timestamp(item, default_time=current_time)
        recency = self.compute_recency_score(item_time, current_time=current_time)
        importance = _extract_importance(item)
        text = _extract_text(item)
        relevance = self.compute_semantic_relevance(text, query)

        w_rec = self.config.weight_recency
        w_imp = self.config.weight_importance
        w_rel = self.config.weight_relevance
        w_sum = w_rec + w_imp + w_rel

        if w_sum > 0:
            composite = (w_rec * recency + w_imp * importance + w_rel * relevance) / w_sum
        else:
            composite = (recency + importance + relevance) / 3.0

        metrics = {
            "recency": recency,
            "importance": importance,
            "relevance": relevance,
            "composite_utility": composite,
        }
        return composite, metrics

    def _offload_pruned_items(
        self,
        pruned_items: List[Any],
        store: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Offloads pruned items to the episodic memory store.
        Supports store_event, add_memory, and append.
        """
        user_id = metadata.get("user_id") if metadata else None
        for item in pruned_items:
            text = _extract_text(item)
            tags = _extract_tags(item)
            if hasattr(store, "store_event"):
                store.store_event(
                    text,
                    metadata={"tags": tags, "pruned_by": "SemanticPruningHookV3", "pruned": True},
                    user_id=user_id,
                )
            elif hasattr(store, "add_memory"):
                store.add_memory(text)
            elif hasattr(store, "append"):
                store.append(text)
            logger.debug(f"Offloaded pruned memory entry to episodic store: {text[:40]}...")

    async def prune_context(
        self,
        context_items: List[Any],
        query: Optional[str] = None,
        min_utility: Optional[float] = None,
        current_time: Optional[float] = None,
        episodic_store: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Dynamically prunes decayed and low-utility entries from context_items.

        Args:
            context_items: List of memory entries or text items.
            query: Optional semantic search query or current focus.
            min_utility: Custom minimum composite utility threshold override.
            current_time: Mocked or current timestamp.
            episodic_store: Optional target store override for pruned items.
            metadata: Context metadata (user_id, etc.).

        Returns:
            List of retained context items above the utility threshold.
        """
        if not context_items:
            return []

        threshold = min_utility if min_utility is not None else self.config.min_utility_threshold
        store = episodic_store or self.episodic_memory

        retained: List[Any] = []
        pruned: List[Any] = []

        for item in context_items:
            utility, metrics = self.calculate_utility(item, query=query, current_time=current_time)
            if utility >= threshold:
                retained.append(item)
            else:
                pruned.append(item)
                self.pruning_history.append({
                    "item": _extract_text(item),
                    "utility": utility,
                    "metrics": metrics,
                    "timestamp": current_time or time.time(),
                })

        # If max_context_items is set and retained exceeds it, prune lowest utility further
        limit = metadata.get("limit", self.max_context_items) if metadata else self.max_context_items
        if limit is not None and len(retained) > limit:
            scored_retained = [
                (self.calculate_utility(it, query=query, current_time=current_time)[0], idx, it)
                for idx, it in enumerate(retained)
            ]
            scored_retained.sort(key=lambda x: x[0], reverse=True)
            kept_tuples = sorted(scored_retained[:limit], key=lambda x: x[1])
            excess_tuples = scored_retained[limit:]
            retained = [t[2] for t in kept_tuples]
            for t in excess_tuples:
                pruned.append(t[2])

        if store is not None and pruned:
            self._offload_pruned_items(pruned, store, metadata)

        logger.info(
            f"SemanticPruningHookV3 pruned context: {len(context_items)} -> {len(retained)} items "
            f"({len(pruned)} items pruned, min_utility={threshold:.2f})."
        )
        return retained

    # --- ContextPlugin Protocol Methods ---

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        Context Engine lifecycle compact hook.
        Executes semantic pruning on incoming context_items.
        """
        query = metadata.get("query")
        min_utility = metadata.get("min_utility", self.config.min_utility_threshold)
        current_time = metadata.get("current_time")
        episodic_store = metadata.get("episodic_memory") or metadata.get("memory_store") or self.episodic_memory

        return await self.prune_context(
            context_items=context_items,
            query=query,
            min_utility=min_utility,
            current_time=current_time,
            episodic_store=episodic_store,
            metadata=metadata,
        )

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Ingest lifecycle hook. Tags content or records timestamp."""
        tags = metadata.get("tags", [])
        if tags:
            return f"[PruningTracked:{','.join(tags)}] {content}"
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble lifecycle hook. Assembles retained context into prompt representation."""
        if not context_items:
            return ""
        lines = [_extract_text(item) for item in context_items]
        return "\n".join(lines)

    def before_retrieval(self, query: str, user_id: int) -> str:
        """Synchronous hook before context retrieval."""
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """Synchronous hook after context retrieval."""
        return context

    def before_write(self, context: Any, user_id: int) -> Any:
        """Synchronous hook before context write."""
        return context

    def after_write(self, context: Any, user_id: int) -> None:
        """Synchronous hook after context write."""
        pass

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """Synchronous hook when context updates."""
        pass


# Aliases for backward and cross-architecture compatibility
SemanticPruningV3 = SemanticPruningHookV3
