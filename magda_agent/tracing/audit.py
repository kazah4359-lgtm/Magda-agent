import time
import copy
from collections import deque
from typing import Any, Deque, Dict, List, Optional


class AuditLogger:
    """
    Records and query tool invocations: what, when, why (from plan), result, duration.
    Includes sanitization to prevent sensitive data logging and uses a bounded ring
    buffer to avoid unbounded memory growth in long-running processes.
    """

    SENSITIVE_KEYS = {"password", "secret", "key", "token", "auth", "credential", "env"}

    def __init__(self, max_capacity: int = 1000) -> None:
        self.max_capacity = max_capacity
        self.trail: Deque[Dict[str, Any]] = deque(maxlen=max_capacity)

    def _sanitize(self, data: Any) -> Any:
        if isinstance(data, dict):
            sanitized = {}
            for k, v in data.items():
                if any(sensitive in k.lower() for sensitive in self.SENSITIVE_KEYS) and not isinstance(v, (dict, list)):
                    sanitized[k] = "***"
                else:
                    sanitized[k] = self._sanitize(v)
            return sanitized
        if isinstance(data, list):
            return [self._sanitize(item) for item in data]
        return copy.deepcopy(data)

    def log_call(self, tool_name: str, kwargs: Dict[str, Any], why: str, result: Any, duration: float) -> None:
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
        results = list(self.trail)
        if tool_name:
            results = [e for e in results if e["tool_name"] == tool_name]
        if start_time is not None:
            results = [e for e in results if e["timestamp"] >= start_time]
        if end_time is not None:
            results = [e for e in results if e["timestamp"] <= end_time]
        return results

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.trail)

    def clear(self) -> None:
        self.trail.clear()
