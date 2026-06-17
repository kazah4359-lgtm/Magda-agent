import contextvars
import uuid
import logging
from typing import Optional, Dict

_trace_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("a2a_trace_id", default=None)

TRACE_HEADER = "X-A2A-Trace-ID"

class A2ATracer:
    """
    Provides distributed tracing capabilities for Agent-to-Agent (A2A) delegations.
    Ensures that a trace ID is propagated across multiple agents involved in a task.
    """

    @staticmethod
    def generate_trace_id() -> str:
        """
        Generates a new unique trace ID.

        Returns:
            str: A unique hexadecimal trace ID.
        """
        return uuid.uuid4().hex

    @staticmethod
    def get_current_trace_id() -> Optional[str]:
        """
        Retrieves the trace ID associated with the current asynchronous context.

        Returns:
            Optional[str]: The current trace ID, or None if not set.
        """
        return _trace_id_ctx.get()

    @staticmethod
    def set_trace_id(trace_id: Optional[str]) -> None:
        """
        Sets the trace ID for the current asynchronous context.

        Args:
            trace_id: The trace ID to set.
        """
        _trace_id_ctx.set(trace_id)
        if trace_id:
            logging.debug(f"[A2A TRACE] Context trace ID set to: {trace_id}")

    @staticmethod
    def get_or_create_trace_id() -> str:
        """
        Gets the current trace ID or creates a new one if none exists.

        Returns:
            str: The current or newly generated trace ID.
        """
        trace_id = A2ATracer.get_current_trace_id()
        if not trace_id:
            trace_id = A2ATracer.generate_trace_id()
            A2ATracer.set_trace_id(trace_id)
        return trace_id

    @staticmethod
    def inject_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """
        Injects the current trace ID into the provided headers dictionary.

        Args:
            headers: The headers dictionary to modify.

        Returns:
            Dict[str, str]: The modified headers dictionary.
        """
        trace_id = A2ATracer.get_or_create_trace_id()
        headers[TRACE_HEADER] = trace_id
        return headers

    @staticmethod
    def extract_from_headers(headers: Dict[str, str]) -> Optional[str]:
        """
        Extracts the trace ID from the provided headers.

        Args:
            headers: The headers dictionary (keys should be normalized or matched case-insensitively).

        Returns:
            Optional[str]: The extracted trace ID, or None if not found.
        """
        # Handle case-insensitive header lookup
        header_map = {k.lower(): v for k, v in headers.items()}
        return header_map.get(TRACE_HEADER.lower())
