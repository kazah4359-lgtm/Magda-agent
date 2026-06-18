"""MCPKernel Taint Tracking Sandbox v2."""
from typing import Any, Callable, Dict, Set, Union, List

class PolicyViolationError(Exception):
    """Raised when tainted data violates policy."""
    pass

class TaintedData:
    """Base class for tainted data."""
    def __init__(self, value: Any):
        self.value = value

class TaintedString(TaintedData, str):
    """A string subclass that marks data as tainted."""
    def __new__(cls, value: str):
        obj = str.__new__(cls, value)
        obj.value = value
        return obj

def mark_tainted(s: Any) -> Any:
    """Marks a string, dict, or list as tainted recursively."""
    if isinstance(s, str):
        return TaintedString(s)
    elif isinstance(s, list):
        return [mark_tainted(item) for item in s]
    elif isinstance(s, dict):
        return {mark_tainted(k): mark_tainted(v) for k, v in s.items()}
    return s

def is_tainted(s: Any) -> bool:
    """Checks if an object or any of its nested structures is tainted."""
    if isinstance(s, TaintedData):
        return True
    if isinstance(s, list):
        return any(is_tainted(item) for item in s)
    if isinstance(s, dict):
        return any(is_tainted(k) or is_tainted(v) for k, v in s.items())
    return False

def sanitize(s: Any) -> Any:
    """Sanitizes tainted data, returning regular primitives recursively."""
    if isinstance(s, TaintedData):
        return sanitize(s.value)
    elif isinstance(s, list):
        return [sanitize(item) for item in s]
    elif isinstance(s, dict):
        return {sanitize(k): sanitize(v) for k, v in s.items()}
    return s
