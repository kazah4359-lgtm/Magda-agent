class TaintedString(str):
    """A string subclass that marks data as tainted."""
    pass

def mark_tainted(s: str) -> TaintedString:
    """Marks a string as tainted."""
    return TaintedString(s)

def is_tainted(s: str) -> bool:
    """Checks if a string is tainted."""
    return isinstance(s, TaintedString)

def sanitize(s: str) -> str:
    """Sanitizes a tainted string, returning a regular string."""
    return str(s)
