import contextvars
import time
from typing import Any, Dict, List, Optional


class ThoughtChainTracer:
    """
    Traces cognitive steps per async context instead of using one global list.
    The per-context trace is capped to prevent unbounded memory growth.
    """

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._trace_context: contextvars.ContextVar[List[Dict[str, Any]]] = contextvars.ContextVar(
            f"thought_chain_trace_{id(self)}"
        )

    def start_trace(self) -> None:
        """Initializes an empty trace for the current context/request."""
        self._trace_context.set([])

    def add_step(self, step_name: str, data: Any = None) -> None:
        """Records a single cognitive step in the current context."""
        trace = self._get_or_create_trace()
        if len(trace) >= self.max_entries:
            return

        entry = {
            "timestamp": time.time(),
            "step": step_name
        }
        if data is not None:
            entry["data"] = data
        trace.append(entry)

    def get_trace(self) -> List[Dict[str, Any]]:
        """Returns a snapshot of the trace for the current context."""
        return list(self._get_or_create_trace())

    def clear(self) -> None:
        """Clears the trace for the current context."""
        self._trace_context.set([])

    def _get_or_create_trace(self) -> List[Dict[str, Any]]:
        try:
            return self._trace_context.get()
        except LookupError:
            trace: List[Dict[str, Any]] = []
            self._trace_context.set(trace)
            return trace
