import time
import copy
from typing import Dict, Any, List, Optional

class PremptiAuditLogger:
    """
    An audit logger for tool-call interceptions with sanitization.
    Inspired by Prempti (Falco).
    """

    def __init__(self, max_capacity: int = 1000):
        self.logs: List[Dict[str, Any]] = []
        self.max_capacity = max_capacity

        self.sensitive_keys = {
            "password", "api_key", "token", "auth_header", "secret_key",
            "api_keys" # handle lists
        }

    def _sanitize(self, data: Any, key_name: str = "", force_sanitize: bool = False) -> Any:
        """
        Recursively sanitize sensitive data.
        If force_sanitize is True, all primitive values are redacted.
        """
        is_sensitive = force_sanitize or any(sec in key_name.lower() for sec in self.sensitive_keys)

        if isinstance(data, dict):
            sanitized_dict = {}
            for k, v in data.items():
                sanitized_dict[k] = self._sanitize(v, k, force_sanitize=is_sensitive)
            return sanitized_dict
        elif isinstance(data, list):
            return [self._sanitize(item, key_name, force_sanitize=is_sensitive) for item in data]
        else:
             # Primitive value
             if is_sensitive:
                 return "***"
             return data


    def log_call(self, tool_name: str, kwargs: Dict[str, Any], why: str, result: str, duration: float) -> None:
        """Logs a tool call with sanitized kwargs."""

        entry = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "kwargs": self._sanitize(copy.deepcopy(kwargs)),
            "why": why,
            "result": result,
            "duration": duration
        }

        self.logs.append(entry)
        if len(self.logs) > self.max_capacity:
            self.logs.pop(0)

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all audit logs."""
        return self.logs

    def query(self, tool_name: Optional[str] = None, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Queries the audit logs based on filters."""
        results = []
        for log in self.logs:
            if tool_name and log["tool_name"] != tool_name:
                continue
            if start_time and log["timestamp"] < start_time:
                continue
            if end_time and log["timestamp"] > end_time:
                continue
            results.append(log)
        return results

    def clear(self) -> None:
        """Clears all audit logs."""
        self.logs = []
