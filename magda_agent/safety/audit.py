import time
import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from magda_agent.safety.audit_trail import AuditTrail

F = TypeVar('F', bound=Callable[..., Any])

class ToolInterceptor:
    """
    A decorator and wrapper class that intercepts tool calls to provide
    an audit trail of invocations, arguments, and results.
    Inspired by Prempti (Falco).
    """

    def __init__(self, audit_trail: AuditTrail) -> None:
        """
        Initializes the ToolInterceptor with an underlying AuditTrail instance.

        Args:
            audit_trail (AuditTrail): The audit trail instance to log events into.
        """
        self.audit_trail = audit_trail

    def intercept(self, tool_name: Optional[str] = None, why: str = "intercepted call") -> Callable[[F], F]:
        """
        A decorator to intercept a function/tool call.

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
                        self.audit_trail.log_call(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            why=why,
                            result=result,
                            duration=duration
                        )
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        self.audit_trail.log_call(
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
                        self.audit_trail.log_call(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            why=why,
                            result=result,
                            duration=duration
                        )
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        self.audit_trail.log_call(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            why=f"{why} (failed)",
                            result=str(e),
                            duration=duration
                        )
                        raise
                return cast(F, sync_wrapper)

        return decorator

    def _extract_args(self, func: Callable[..., Any], args: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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

    async def execute_async(self, func: Callable[..., Any], tool_name: str, why: str, *args: Any, **kwargs: Any) -> Any:
        """
        Explicitly executes and intercepts an async function without decorators.

        Args:
            func (Callable): The async function to execute.
            tool_name (str): The name of the tool.
            why (str): Context for the call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The function's result.
        """
        start_time = time.time()
        all_args = self._extract_args(func, args, kwargs)
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            self.audit_trail.log_call(
                tool_name=tool_name,
                kwargs=all_args,
                why=why,
                result=result,
                duration=duration
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.audit_trail.log_call(
                tool_name=tool_name,
                kwargs=all_args,
                why=f"{why} (failed)",
                result=str(e),
                duration=duration
            )
            raise

    def execute_sync(self, func: Callable[..., Any], tool_name: str, why: str, *args: Any, **kwargs: Any) -> Any:
        """
        Explicitly executes and intercepts a sync function without decorators.

        Args:
            func (Callable): The sync function to execute.
            tool_name (str): The name of the tool.
            why (str): Context for the call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The function's result.
        """
        start_time = time.time()
        all_args = self._extract_args(func, args, kwargs)
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self.audit_trail.log_call(
                tool_name=tool_name,
                kwargs=all_args,
                why=why,
                result=result,
                duration=duration
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.audit_trail.log_call(
                tool_name=tool_name,
                kwargs=all_args,
                why=f"{why} (failed)",
                result=str(e),
                duration=duration
            )
            raise
