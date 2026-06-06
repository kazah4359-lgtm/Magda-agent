import time
import copy
from typing import Dict, Any, List, Optional

class AuditLogger:
    """
    Records and query tool invocations: what, when, why (from plan), result, duration.
    Includes sanitization to prevent sensitive data logging.
    """

    SENSITIVE_KEYS = {"password", "secret", "key", "token", "auth", "credential", "env"}

    def __init__(self) -> None:
        """
        Initializes the AuditLogger.
        """
        self.trail: List[Dict[str, Any]] = []

    def _sanitize(self, data: Any) -> Any:
        """
        Recursively sanitizes sensitive data from dictionaries.
        """
        if isinstance(data, dict):
            sanitized = {}
            for k, v in data.items():
                # If the key implies sensitivity AND the value is a string/primitive
                # We mask it. If it's a dict or list, we recurse to preserve structure
                # while masking inner sensitive keys.
                if any(sensitive in k.lower() for sensitive in self.SENSITIVE_KEYS) and not isinstance(v, (dict, list)):
                    sanitized[k] = "***"
                else:
                    sanitized[k] = self._sanitize(v)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        return copy.deepcopy(data)

    def log_call(self, tool_name: str, kwargs: Dict[str, Any], why: str, result: Any, duration: float) -> None:
        """
        Logs a tool call with metadata.
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
        Queries the audit log.
        """
        results = self.trail
        if tool_name:
            results = [e for e in results if e["tool_name"] == tool_name]
        if start_time is not None:
            results = [e for e in results if e["timestamp"] >= start_time]
        if end_time is not None:
            results = [e for e in results if e["timestamp"] <= end_time]
        return results

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Returns all log entries.
        """
        return self.trail

    def clear(self) -> None:
        """
        Clears the audit log.
        """
        self.trail.clear()
