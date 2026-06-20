import time
import copy
from collections import deque
from typing import Any, Deque, Dict, List, Optional


class AuditTrail:
    """
    Records and queries tool invocations: what, when, why (from plan/checkpoints),
    result, and duration. Includes recursive sanitization to prevent sensitive
    data logging and uses a bounded ring buffer to avoid memory leaks.
    Inspired by Prempti (Falco).
    """

    def __init__(self, max_capacity: int = 1000) -> None:
        """
        Initializes the AuditTrail with a fixed capacity.

        Args:
            max_capacity: Maximum number of entries to keep in the trail.
        """
        self.max_capacity = max_capacity
        self.trail: Deque[Dict[str, Any]] = deque(maxlen=max_capacity)

    def _sanitize(self, data: Any) -> Any:
        """
        Recursively sanitizes sensitive data from dictionaries and lists.
        Redacts values for keys that match sensitive patterns.
        """
        if isinstance(data, dict):
            sanitized = {}
            for k, v in data.items():
                k_l = k.lower()
                is_s = any(s in k_l for s in ["password", "api_key", "auth_header", "token", "access_key", "secret"])
                if not is_s:
                    is_s = k_l in {"key", "auth", "credential", "private"}

                if is_s:
                    if isinstance(v, dict):
                        # If a sensitive key points to a dict, we redact the whole thing
                        # to be safe, or we could continue to sanitize it.
                        # Redacting the whole thing is safer.
                        sanitized[k] = "***"
                    elif isinstance(v, list):
                        # Redact all primitives in the list, recurse for dicts
                        sanitized[k] = ["***" if not isinstance(item, (dict, list)) else self._sanitize(item) for item in v]
                    else:
                        sanitized[k] = "***"
                else:
                    sanitized[k] = self._sanitize(v)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        return copy.deepcopy(data)

    def log_call(self, tool_name: str, kwargs: Dict[str, Any], why: str, result: Any, duration: float = 0.0) -> None:
        """
        Logs a tool call with metadata and sanitization.

        Args:
            tool_name: Name of the tool or action.
            kwargs: Arguments passed to the tool.
            why: Reason for the call or outcome.
            result: Outcome of the execution or "allowed"/"blocked".
            duration: Time taken in seconds.
        """
        entry = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "kwargs": self._sanitize(kwargs),
            "why": why,
            "result": self._sanitize(result),
            "duration": duration
        }
        self.trail.append(entry)

    def query(self, tool_name: Optional[str] = None, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Queries the audit trail with optional filters.
        """
        results = list(self.trail)
        if tool_name:
            results = [e for e in results if e["tool_name"] == tool_name]
        if start_time is not None:
            results = [e for e in results if e["timestamp"] >= start_time]
        if end_time is not None:
            results = [e for e in results if e["timestamp"] <= end_time]
        return results

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all entries in the audit trail."""
        return list(self.trail)

    def clear(self) -> None:
        """Clears all entries from the audit trail."""
        self.trail.clear()
