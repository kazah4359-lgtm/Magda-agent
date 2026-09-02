import time
import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.safety.taint import is_tainted

F = TypeVar('F', bound=Callable[..., Any])

class PremptiToolLoggerV1:
    """
    Low-level logger inspired by Prempti (Falco).
    Intercepts dynamic tool executions and records pre-execution metadata
    into the AuditTrail prior to executing (or without executing) the tool.
    """

    def __init__(self, audit_trail: Optional[AuditTrail] = None) -> None:
        """
        Initializes the PremptiToolLoggerV1.

        Args:
            audit_trail: Optional AuditTrail instance for logging. If None, a default
                         in-memory AuditTrail instance is created.
        """
        if audit_trail is None:
            self.audit_trail = AuditTrail(db_path=None)
        else:
            self.audit_trail = audit_trail

    def _extract_args(self, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Dict[str, Any]:
        """Extracts and binds arguments passed to a function using its signature."""
        try:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return dict(bound.arguments)
        except Exception:
            extracted: Dict[str, Any] = {}
            for i, v in enumerate(args):
                extracted[f"arg_{i}"] = v
            extracted.update(kwargs)
            return extracted

    def log_pre_execution(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        why: str = "pre-execution interception"
    ) -> Dict[str, Any]:
        """
        Logs a tool call into the AuditTrail prior to execution.

        Args:
            tool_name: Name of the tool or action.
            kwargs: Arguments passed to the tool.
            why: Context or reason for interception.

        Returns:
            Dict[str, Any]: The pre-execution metadata dictionary logged to AuditTrail.
        """
        has_taint = any(is_tainted(v) for v in kwargs.values())
        log_args = dict(kwargs)
        log_args["_tainted_boundary_crossover"] = has_taint

        self.audit_trail.log_call(
            tool_name=tool_name,
            kwargs=log_args,
            why=why,
            result="pre_execution",
            duration=0.0
        )
        return {
            "tool_name": tool_name,
            "kwargs": log_args,
            "why": why,
            "status": "pre_execution",
            "timestamp": time.time()
        }

    def intercept(
        self,
        tool_name: Optional[str] = None,
        why: str = "intercepted call",
        execute_tool: bool = True
    ) -> Callable[[F], F]:
        """
        A decorator that intercepts dynamic tool execution, logs pre-execution metadata
        to AuditTrail, and conditionally executes the underlying tool.

        Args:
            tool_name: Name of the tool. If None, uses func.__name__.
            why: Reason or context for interception.
            execute_tool: If True, executes the tool and logs the result.
                          If False, logs pre-execution and skips execution.
        """
        def decorator(func: F) -> F:
            name_to_use = tool_name if tool_name else func.__name__

            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    all_args = self._extract_args(func, args, kwargs)
                    self.log_pre_execution(name_to_use, all_args, why=f"{why} (pre-execution)")

                    if not execute_tool:
                        return None

                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time
                        has_taint = any(is_tainted(v) for v in all_args.values()) or is_tainted(result)
                        log_args = dict(all_args)
                        log_args["_tainted_boundary_crossover"] = has_taint
                        self.audit_trail.log_call(name_to_use, log_args, why, result, duration)
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        has_taint = any(is_tainted(v) for v in all_args.values())
                        log_args = dict(all_args)
                        log_args["_tainted_boundary_crossover"] = has_taint
                        self.audit_trail.log_call(name_to_use, log_args, f"{why} (failed)", str(e), duration)
                        raise
                return cast(F, async_wrapper)
            else:
                @wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    all_args = self._extract_args(func, args, kwargs)
                    self.log_pre_execution(name_to_use, all_args, why=f"{why} (pre-execution)")

                    if not execute_tool:
                        return None

                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        has_taint = any(is_tainted(v) for v in all_args.values()) or is_tainted(result)
                        log_args = dict(all_args)
                        log_args["_tainted_boundary_crossover"] = has_taint
                        self.audit_trail.log_call(name_to_use, log_args, why, result, duration)
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        has_taint = any(is_tainted(v) for v in all_args.values())
                        log_args = dict(all_args)
                        log_args["_tainted_boundary_crossover"] = has_taint
                        self.audit_trail.log_call(name_to_use, log_args, f"{why} (failed)", str(e), duration)
                        raise
                return cast(F, sync_wrapper)

        return decorator
