import re
import time
import copy
import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

F = TypeVar('F', bound=Callable[..., Any])

class PremptiAuditLoggerV2:
    """
    An advanced audit logger inspired by Prempti (Falco).
    Intercepts tool calls to provide rigorous audit trails of invocations,
    arguments, and results, while preventing data leaks through robust
    sanitization/redaction of sensitive arguments.
    """

    # Expanded set of sensitive keys to redact
    SENSITIVE_KEYS = {
        "password", "secret", "key", "token", "auth", "credential",
        "env", "api_key", "access_key", "private", "private_key"
    }

    def __init__(self) -> None:
        """Initializes the PremptiAuditLoggerV2."""
        self.trail: List[Dict[str, Any]] = []

    def _sanitize(self, data: Any, memo: Optional[Dict[int, Any]] = None) -> Any:
        """
        Recursively sanitizes sensitive data from dictionaries and lists.
        Redacts values for keys that match sensitive patterns.
        Handles circular references via a memo dictionary.
        """
        if memo is None:
            memo = {}

        if id(data) in memo:
            return "<circular reference>"

        if inspect.isawaitable(data):
            return "<awaitable>"

        if isinstance(data, dict):
            memo[id(data)] = True
            sanitized = {}
            for k, v in data.items():
                k_lower = k.lower()

                # Check for exact matches or word boundaries to avoid over-redacting
                is_sensitive = False
                for s in self.SENSITIVE_KEYS:
                    # check if the sensitive key is present as a standalone word/stem,
                    # allowing for trailing 's' or being surrounded by underscores/etc.
                    # This avoids 'auth' matching 'author' but allows 'secret' matching 'secrets'
                    if re.search(rf"(^|[^a-zA-Z]){re.escape(s)}s?([^a-zA-Z]|$)", k_lower):
                        is_sensitive = True
                        break

                if is_sensitive:
                    sanitized[k] = "***"
                else:
                    sanitized[k] = self._sanitize(v, memo)
            return sanitized

        elif isinstance(data, list):
            memo[id(data)] = True
            return [self._sanitize(item, memo) for item in data]

        try:
            return copy.deepcopy(data)
        except Exception:
            return repr(data)


    def log_call(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        why: str,
        result: Any,
        duration: float = 0.0
    ) -> None:
        """
        Logs a tool call with metadata and sanitization.

        Args:
            tool_name: Name of the tool or action.
            kwargs: Arguments passed to the tool.
            why: Reason for the call or outcome context.
            result: Outcome of the execution.
            duration: Time taken in seconds.
        """
        entry = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "kwargs": self._sanitize(kwargs, None),
            "why": why,
            "result": self._sanitize(result, None),
            "duration": duration
        }
        self.trail.append(entry)

    def intercept(
        self,
        tool_name: Optional[str] = None,
        why: str = "intercepted call"
    ) -> Callable[[F], F]:
        """
        A decorator to intercept a function/tool call, log its arguments,
        execution time, and results securely.

        Args:
            tool_name (Optional[str]): The name of the tool. If not provided,
                                       the function's __name__ is used.
            why (str): The default reason or context for the call.

        Returns:
            Callable[[F], F]: The decorated function.
        """
        def decorator(func: F) -> F:
            name_to_use = tool_name if tool_name else func.__name__

            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.time()
                    all_args = self._extract_args(func, args, kwargs)
                    try:
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time
                        self.log_call(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            why=why,
                            result=result,
                            duration=duration
                        )
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        self.log_call(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            why=f"{why} (failed)",
                            result=str(e),
                            duration=duration
                        )
                        raise
                return cast(F, async_wrapper)
            else:
                @wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.time()
                    all_args = self._extract_args(func, args, kwargs)
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        self.log_call(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            why=why,
                            result=result,
                            duration=duration
                        )
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        self.log_call(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            why=f"{why} (failed)",
                            result=str(e),
                            duration=duration
                        )
                        raise
                return cast(F, sync_wrapper)

        return decorator

    def _extract_args(
        self,
        func: Callable[..., Any],
        args: Any,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extracts and normalizes arguments passed to a function based on its signature.

        Args:
            func (Callable): The function being called.
            args (tuple): Positional arguments.
            kwargs (dict): Keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary of all arguments bound to their names.
        """
        try:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return dict(bound.arguments)
        except Exception:
            # Fallback if signature parsing fails
            return {"args": list(args), "kwargs": kwargs}

    def query(
        self,
        tool_name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
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
        """Returns all log entries."""
        return self.trail

    def clear(self) -> None:
        """Clears the audit log."""
        self.trail.clear()
