import contextvars
import time
import uuid
from typing import Any, Dict, List, Optional


class A2AAsyncTracer:
    """
    A2A Async-first Tracing V4 for cross-agent workflows.
    Uses contextvars to maintain correlation IDs across asynchronous tasks.
    """

    def __init__(self) -> None:
        self._correlation_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
            f"a2a_correlation_id_{id(self)}", default=None
        )
        self.traces: Dict[str, List[Dict[str, Any]]] = {}

    def set_correlation_id(self, correlation_id: Optional[str] = None) -> str:
        """
        Sets the correlation ID for the current async context.
        If no ID is provided, generates a new UUID.
        """
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        self._correlation_id_ctx.set(correlation_id)
        if correlation_id not in self.traces:
            self.traces[correlation_id] = []
        return correlation_id

    def get_correlation_id(self) -> Optional[str]:
        """
        Retrieves the current correlation ID from the context.
        """
        return self._correlation_id_ctx.get()

    def clear_correlation_id(self) -> None:
        """
        Clears the correlation ID from the current context.
        """
        self._correlation_id_ctx.set(None)

    def log_action(self, action_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Logs an action under the current correlation ID.
        """
        correlation_id = self.get_correlation_id()
        if not correlation_id:
            return

        if correlation_id not in self.traces:
            self.traces[correlation_id] = []

        self.traces[correlation_id].append({
            "timestamp": time.time(),
            "action": action_name,
            "payload": payload or {}
        })

    def get_traces(self, correlation_id: str) -> List[Dict[str, Any]]:
        """
        Returns all traces associated with a specific correlation ID.
        """
        return self.traces.get(correlation_id, [])

    def clear_all_traces(self) -> None:
        """
        Clears all stored traces. Useful for testing.
        """
        self.traces.clear()
