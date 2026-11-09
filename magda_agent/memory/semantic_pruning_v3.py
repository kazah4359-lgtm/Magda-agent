import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from magda_agent.memory.context_engine import ContextPlugin
from magda_agent.memory.working import MemoryEntry

if TYPE_CHECKING:
    from magda_agent.llm_client import LLMClient
    from magda_agent.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


@dataclass
class DecayConfigV3:
    """
    Configuration for virtual context semantic pruning and decay metrics.
    """
    half_life_seconds: float = 3600.0  # 1 hour half-life for short-term decay
    decay_mode: str = "exponential"    # "exponential" or "linear"
    linear_decay_rate: float = 0.0001  # Weight loss per second in linear mode
    min_utility_threshold: float = 0.25 # Threshold below which decayed entries are pruned
    importance_weight: float = 0.45    # Composite score weight for entry importance
    relevance_weight: float = 0.35     # Composite score weight for query/semantic relevance
    recency_weight: float = 0.20       # Composite score weight for temporal position


class SemanticPruningHookV3(ContextPlugin):
    """
    SemanticPruningHookV3 implements Letta/MemGPT-inspired dynamic semantic pruning
    for virtual context streams and working memory.

    It dynamically calculates time-decayed utility scores for short-term entries,
    prioritizes high-relevance and critical entries, removes irrelevant or expired items,
    and optionally offloads pruned context entries into episodic memory.
    """

    def __init__(
        self,
        config: Optional[DecayConfigV3] = None,
        episodic_memory: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        default_max_tokens: int = 2000,
        user_id: Optional[int] = None,
    ) -> None:
        self.config = config or DecayConfigV3()
        self.episodic_memory = episodic_memory
        self.llm_client = llm_client
        self.default_max_tokens = default_max_tokens
        self.user_id = user_id
        self.pruned_history: List[Dict[str, Any]] = []

    def calculate_decay_factor(self, elapsed_seconds: float) -> float:
        """
        Calculates the decay multiplier (0.0 to 1.0) based on elapsed seconds.
        """
        if elapsed_seconds <= 0.0:
            return 1.0

        if self.config.decay_mode == "exponential":
            if self.config.half_life_seconds <= 0.0:
                return 0.0
            return 0.5 ** (elapsed_seconds / self.config.half_life_seconds)
        elif self.config.decay_mode == "linear":
            return max(0.0, 1.0 - (self.config.linear_decay_rate * elapsed_seconds))
        return 1.0

    def calculate_semantic_relevance(self, content: str, query: Optional[str]) -> float:
        """
        Calculates semantic/lexical overlap relevance score between content and a query string.
        """
        if not query or not query.strip():
            return 0.5  # Neutral relevance when no explicit query is provided

        query_tokens = set(query.lower().split())
        content_tokens = set(content.lower().split())

        if not query_tokens:
            return 0.5

        overlap = query_tokens.intersection(content_tokens)
        return len(overlap) / len(query_tokens)

    def get_token_length(self, entries: List[Any]) -> int:
        """
        Estimates the token length of memory entries.
        """
        total_words = sum(len(getattr(e, "content", str(e)).split()) for e in entries)
        return int(total_words * 1.3)

    def evaluate_entry_utility(
        self,
        entry: Any,
        idx: int,
        total_count: int,
        current_time: float,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates an individual memory entry's decayed utility score.
        """
        content = getattr(entry, "content", str(entry))
        initial_importance = float(getattr(entry, "importance", 0.5))
        entry_time = getattr(entry, "timestamp", current_time)

        elapsed = max(0.0, current_time - entry_time)
        decay_factor = self.calculate_decay_factor(elapsed)
        decayed_importance = initial_importance * decay_factor

        relevance = self.calculate_semantic_relevance(content, query)
        recency = idx / max(1, total_count - 1) if total_count > 1 else 1.0

        utility_score = (
            (decayed_importance * self.config.importance_weight)
            + (relevance * self.config.relevance_weight)
            + (recency * self.config.recency_weight)
        )

        return {
            "entry": entry,
            "utility_score": utility_score,
            "decay_factor": decay_factor,
            "decayed_importance": decayed_importance,
            "relevance": relevance,
            "recency": recency,
            "is_critical": initial_importance >= 0.8 or utility_score >= 0.7,
        }

    async def prune_context(
        self,
        entries: List[Any],
        max_tokens: Optional[int] = None,
        query: Optional[str] = None,
        current_time: Optional[float] = None,
        user_id: Optional[int] = None,
    ) -> List[Any]:
        """
        Prunes decayed and irrelevant entries from the virtual context stream.
        Offloads pruned items to episodic memory if available.
        """
        if not entries:
            return []

        now = current_time if current_time is not None else time.time()
        token_limit = max_tokens if max_tokens is not None else self.default_max_tokens
        uid = user_id if user_id is not None else self.user_id

        scored = [
            self.evaluate_entry_utility(entry, idx, len(entries), now, query)
            for idx, entry in enumerate(entries)
        ]

        # Filter out entries whose utility has decayed below min_utility_threshold
        # (Preserve critical items even if decayed slightly)
        retained_items: List[Dict[str, Any]] = []
        evicted_items: List[Dict[str, Any]] = []

        for item in scored:
            if item["is_critical"] or item["utility_score"] >= self.config.min_utility_threshold:
                retained_items.append(item)
            else:
                evicted_items.append(item)

        # If retained items exceed token_limit, prune lowest utility entries first
        while retained_items and self.get_token_length([it["entry"] for it in retained_items]) > token_limit:
            # Find item with lowest utility score to evict
            lowest_idx = min(range(len(retained_items)), key=lambda i: retained_items[i]["utility_score"])
            evicted_items.append(retained_items.pop(lowest_idx))

        # Offload evicted items to episodic memory if configured
        if evicted_items and self.episodic_memory is not None:
            await self._offload_to_episodic(evicted_items, uid)

        for evicted in evicted_items:
            self.pruned_history.append({
                "content": getattr(evicted["entry"], "content", str(evicted["entry"])),
                "utility_score": evicted["utility_score"],
                "decay_factor": evicted["decay_factor"],
                "timestamp": now,
                "user_id": uid,
            })

        logger.info(
            f"SemanticPruningHookV3: Pruned {len(evicted_items)} entries, retained {len(retained_items)} entries."
        )

        return [it["entry"] for it in retained_items]

    async def _offload_to_episodic(self, evicted: List[Dict[str, Any]], user_id: Optional[int]) -> None:
        """
        Offloads pruned context entries to episodic memory store.
        """
        for item in evicted:
            entry = item["entry"]
            text = getattr(entry, "content", str(entry))
            metadata = {
                "pruned_by": "SemanticPruningHookV3",
                "final_utility": item["utility_score"],
                "decay_factor": item["decay_factor"],
                "original_importance": getattr(entry, "importance", None),
            }
            try:
                if hasattr(self.episodic_memory, "store_event"):
                    self.episodic_memory.store_event(text=text, metadata=metadata, user_id=user_id)
                elif hasattr(self.episodic_memory, "add_memory"):
                    self.episodic_memory.add_memory(text)
                elif hasattr(self.episodic_memory, "append"):
                    self.episodic_memory.append(text)
            except Exception as e:
                logger.warning(f"Failed to offload pruned entry to episodic memory: {e}")

    # --- ContextPlugin Protocol Implementation ---

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Initialize plugin state from config."""
        if "half_life_seconds" in config:
            self.config.half_life_seconds = float(config["half_life_seconds"])
        if "decay_mode" in config:
            self.config.decay_mode = str(config["decay_mode"])
        if "min_utility_threshold" in config:
            self.config.min_utility_threshold = float(config["min_utility_threshold"])
        if "default_max_tokens" in config:
            self.default_max_tokens = int(config["default_max_tokens"])
        logger.info("SemanticPruningHookV3 bootstrapped.")

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Process incoming content before storage."""
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble context string from items."""
        return "\n".join([f"- {getattr(item, 'content', str(item))}" for item in context_items])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        Compact lifecycle hook: executes semantic pruning based on token limits and decay.
        """
        max_tokens = metadata.get("max_tokens", self.default_max_tokens)
        query = metadata.get("query")
        current_time = metadata.get("current_time")
        user_id = metadata.get("user_id", self.user_id)

        return await self.prune_context(
            entries=context_items,
            max_tokens=max_tokens,
            query=query,
            current_time=current_time,
            user_id=user_id,
        )

    def before_retrieval(self, query: str, user_id: int) -> str:
        """Called before context retrieval."""
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """Called after context retrieval."""
        return context

    def before_write(self, context: Any, user_id: int) -> Any:
        """Called before context is written."""
        return context

    def after_write(self, context: Any, user_id: int) -> None:
        """Called after context is written."""
        pass

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """Called when context is updated."""
        logger.debug(f"SemanticPruningHookV3 on_context_update for user {user_id}")

    async def pre_process(self, content: str, metadata: Dict[str, Any]) -> str:
        """Called to pre-process content before ingestion."""
        return content

    async def post_process(self, response: str, metadata: Dict[str, Any]) -> str:
        """Called to post-process response."""
        return response


# Aliases for naming conventions
SemanticPruningV3 = SemanticPruningHookV3
