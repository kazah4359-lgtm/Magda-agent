from typing import List, Any, Dict
from collections import deque
import logging
import time

from magda_agent.memory.context_engine import ContextPlugin

class TelemetryPlugin(ContextPlugin):
    """
    A ContextPlugin that tracks metrics for hook executions and context sizes.
    Includes comprehensive tracking for all hooks and prevents memory leaks using bounded deques.
    """

    def __init__(self) -> None:
        """Initialize the telemetry plugin and its bounded metrics storage."""
        self.metrics: Dict[str, Any] = {
            "before_retrieval_calls": 0,
            "after_retrieval_calls": 0,
            "total_retrieved_items": 0,
            "queries": deque(maxlen=1000),
            "retrieval_times": deque(maxlen=1000),
            "bootstrap_count": 0,
            "ingest_count": 0,
            "assemble_count": 0,
            "compact_count": 0,
            "on_context_update_count": 0
        }
        self._start_time: float = 0.0

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        self.metrics["bootstrap_count"] += 1
        logging.debug(f"TelemetryPlugin bootstrap count: {self.metrics['bootstrap_count']}")

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        self.metrics["ingest_count"] += 1
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        self.metrics["assemble_count"] += 1
        return "\n".join([str(item) for item in context_items])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        self.metrics["compact_count"] += 1
        return context_items

    def before_retrieval(self, query: str, user_id: int) -> str:
        self.metrics["before_retrieval_calls"] += 1
        self.metrics["queries"].append(query)
        self._start_time = time.time()
        logging.debug(f"TelemetryPlugin before_retrieval count: {self.metrics['before_retrieval_calls']}")
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        elapsed = time.time() - self._start_time
        self.metrics["after_retrieval_calls"] += 1
        self.metrics["total_retrieved_items"] += len(context)
        self.metrics["retrieval_times"].append(elapsed)
        logging.debug(
            f"TelemetryPlugin after_retrieval. Count: {self.metrics['after_retrieval_calls']}, "
            f"Items added: {len(context)}, Total items: {self.metrics['total_retrieved_items']}"
        )
        return context

    def before_write(self, context: Any, user_id: int) -> Any:
        return context

    def after_write(self, context: Any, user_id: int) -> None:
        pass

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        self.metrics["on_context_update_count"] += 1
        logging.debug(f"TelemetryPlugin on_context_update count: {self.metrics['on_context_update_count']}")
