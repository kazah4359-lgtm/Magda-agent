"""
OpenAI SDK Runtime Function Tool Concurrency V1 / V4.

Allows concurrent execution of independent function tool requests inside the runtime execution guardrail.
Supports both OpenAI tool call formats (with JSON-string arguments) and standard dictionary formats,
with thread-safe active task counting, per-execution guardrail hooks, and exception isolation.
"""

import asyncio
import inspect
import json
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class RuntimeFunctionConcurrencyV4:
    """
    Manages concurrent execution of independent function tool requests within a runtime execution guardrail.

    Features:
    - Concurrent function execution via `asyncio.gather` bounded by `asyncio.Semaphore`.
    - Handles OpenAI tool call schemas:
      `{"id": "call_123", "type": "function", "function": {"name": "func_name", "arguments": "{\\"x\\": 1}"}}`
      as well as standard dictionary formats:
      `{"id": "call_123", "name": "func_name", "kwargs": {"x": 1}}`.
    - Integrated guardrail hooks (`guardrail` callable or object with `validate_call` or `execute_with_guard`).
    - Sync skill offloading to worker threads via `asyncio.to_thread`.
    - Isolated error handling: failures in one function execution do not cancel or corrupt parallel tasks.
    - Active task tracking and thread-safe metrics inspection.
    """

    def __init__(
        self,
        registry: Optional[Any] = None,
        functions: Optional[Dict[str, Callable[..., Any]]] = None,
        max_concurrency: int = 10,
        guardrail: Optional[Union[Callable[..., Any], Any]] = None,
    ) -> None:
        """
        Initializes RuntimeFunctionConcurrencyV4.

        Args:
            registry (Optional[Any]): Optional SkillRegistry or registry instance containing skills.
            functions (Optional[Dict[str, Callable]]): Explicit mapping of function names to callable tools.
            max_concurrency (int): Maximum concurrent executions allowed.
            guardrail (Optional[Union[Callable, Any]]): Optional guardrail check hook or guard object.
        """
        self.registry = registry
        self.functions: Dict[str, Callable[..., Any]] = functions.copy() if functions else {}
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.guardrail = guardrail

        self._lock = threading.Lock()
        self._active_tasks_count = 0

    @property
    def active_tasks_count(self) -> int:
        """Returns the number of currently executing concurrent tool tasks."""
        with self._lock:
            return self._active_tasks_count

    def register_function(self, name: str, func: Callable[..., Any]) -> None:
        """
        Registers a function or tool callable directly.

        Args:
            name (str): Name of the function.
            func (Callable): Function or coroutine function.
        """
        self.functions[name] = func

    def set_guardrail(self, guardrail: Union[Callable[..., Any], Any]) -> None:
        """
        Sets or updates the active guardrail check.

        Args:
            guardrail (Union[Callable, Any]): Guardrail callable or object.
        """
        self.guardrail = guardrail

    def _parse_tool_call(self, call: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parses various tool call formats into (call_id, function_name, kwargs).

        Args:
            call (Dict[str, Any]): Input tool call dictionary.

        Returns:
            Tuple[str, str, Dict[str, Any]]: Extracted (call_id, function_name, kwargs)
        """
        call_id = str(call.get("id") or uuid.uuid4())
        name = ""
        kwargs: Dict[str, Any] = {}

        # Case 1: OpenAI tool call format: {"type": "function", "function": {"name": ..., "arguments": ...}}
        if call.get("type") == "function" or ("function" in call and isinstance(call["function"], dict)):
            fn_dict = call.get("function", {})
            name = fn_dict.get("name", "")
            raw_args = fn_dict.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    kwargs = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    kwargs = {"raw_arguments": raw_args}
            elif isinstance(raw_args, dict):
                kwargs = raw_args
        else:
            # Case 2: Standard dict format: {"name": ..., "kwargs": ...} or {"name": ..., "arguments": ...}
            name = call.get("name") or call.get("function") or ""
            if "kwargs" in call and isinstance(call["kwargs"], dict):
                kwargs = call["kwargs"]
            elif "arguments" in call:
                raw_args = call["arguments"]
                if isinstance(raw_args, str):
                    try:
                        kwargs = json.loads(raw_args) if raw_args.strip() else {}
                    except json.JSONDecodeError:
                        kwargs = {"raw_arguments": raw_args}
                elif isinstance(raw_args, dict):
                    kwargs = raw_args
            elif "params" in call and isinstance(call["params"], dict):
                kwargs = call["params"]

        return call_id, str(name), kwargs

    async def _resolve_and_check_guardrail(self, name: str, kwargs: Dict[str, Any]) -> Tuple[Optional[Callable[..., Any]], Optional[str]]:
        """
        Resolves callable function and evaluates guardrail policy check if set.

        Returns:
            Tuple[Optional[Callable], Optional[str]]: (func, error_message)
        """
        if not name:
            return None, "Error: Missing or empty function name."

        # Evaluate guardrail first if configured
        if self.guardrail:
            try:
                if callable(self.guardrail):
                    guard_res = self.guardrail(name, kwargs)
                    if inspect.isawaitable(guard_res):
                        guard_res = await guard_res
                    if guard_res is False:
                        return None, f"Error: Function '{name}' was blocked by runtime guardrail check."
                    if isinstance(guard_res, str) and guard_res.startswith("Error"):
                        return None, guard_res
                elif hasattr(self.guardrail, "validate_call"):
                    guard_res = self.guardrail.validate_call(name, kwargs)
                    if inspect.isawaitable(guard_res):
                        guard_res = await guard_res
                    if guard_res is False:
                        return None, f"Error: Function '{name}' failed guardrail validation."
                    if isinstance(guard_res, str) and guard_res.startswith("Error"):
                        return None, guard_res
                elif hasattr(self.guardrail, "check_policy"):
                    guard_res = self.guardrail.check_policy(name, kwargs)
                    if inspect.isawaitable(guard_res):
                        guard_res = await guard_res
                    if guard_res is False:
                        return None, f"Error: Policy violation for function '{name}'."
            except Exception as e:
                return None, f"Error: Guardrail validation failed: {str(e)}"

        # Resolve function from registered functions or skill registry
        if name in self.functions:
            return self.functions[name], None

        if self.registry and hasattr(self.registry, "skills") and name in self.registry.skills:
            skill = self.registry.skills[name]
            return skill, None

        return None, f"Error: Function or skill '{name}' not found."

    async def _execute_single_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a single function call with error isolation, guardrail enforcement, and sync offloading.

        Args:
            call (Dict[str, Any]): Function call request specification.

        Returns:
            Dict[str, Any]: Structured execution output dictionary.
        """
        call_id, name, kwargs = self._parse_tool_call(call)

        func, guard_err = await self._resolve_and_check_guardrail(name, kwargs)
        if guard_err or func is None:
            return {
                "id": call_id,
                "name": name,
                "status": "error",
                "error": guard_err or "Unknown function error",
                "output": None,
            }

        async with self.semaphore:
            with self._lock:
                self._active_tasks_count += 1
            try:
                # Check if func is coroutine function or callable in registry
                if inspect.iscoroutinefunction(func):
                    res = await func(**kwargs)
                elif self.registry and hasattr(self.registry, "execute_skill") and name in getattr(self.registry, "skills", {}):
                    # Use registry execute_skill if present
                    if inspect.iscoroutinefunction(getattr(self.registry, "skills", {}).get(name)):
                        res = await self.registry.execute_skill(name, **kwargs)
                    else:
                        res = await asyncio.to_thread(self.registry.execute_skill, name, **kwargs)
                        if inspect.isawaitable(res):
                            res = await res
                else:
                    # Sync function offloaded to thread
                    res = await asyncio.to_thread(func, **kwargs)
                    if inspect.isawaitable(res):
                        res = await res

                return {
                    "id": call_id,
                    "name": name,
                    "status": "success",
                    "output": res,
                    "error": None,
                }
            except Exception as e:
                return {
                    "id": call_id,
                    "name": name,
                    "status": "error",
                    "error": f"Error: Function execution failed: {str(e)}",
                    "output": None,
                }
            finally:
                with self._lock:
                    self._active_tasks_count -= 1

    async def execute_concurrently(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes multiple independent function tool calls concurrently using asyncio.gather.

        Args:
            calls (List[Dict[str, Any]]): List of function tool call dictionaries.

        Returns:
            List[Dict[str, Any]]: List of structured execution response dicts in matching order.
        """
        if not calls:
            return []

        tasks = [self._execute_single_call(call) for call in calls]
        return list(await asyncio.gather(*tasks))


# Class Aliases for compatibility and OpenAI Agents SDK integration
OpenAIRuntimeConcurrencyV4 = RuntimeFunctionConcurrencyV4
RuntimeFunctionConcurrencyManagerV4 = RuntimeFunctionConcurrencyV4
