import logging
import asyncio
from enum import Enum
from typing import Any, Tuple, Optional, Callable, Dict, Awaitable

from magda_agent.safety.policy import PolicyLayer


class FallbackStrategyV4(Enum):
    STOP_EXECUTION = "stop_execution"
    REQUEST_REVIEW = "request_review"
    WARN_AND_CONTINUE = "warn_and_continue"
    DYNAMIC_HANDLER = "dynamic_handler"
    NONE = "none"


class SecurityViolationError(Exception):
    pass


class RealtimeGuardrailV4:
    """
    Realtime guardrails v4 that trigger a safe fallback action immediately
    when a policy violation is detected mid-execution. Supports dynamic handlers
    and graceful degradation.
    """

    def __init__(self, policy_layer: PolicyLayer, default_strategy: FallbackStrategyV4 = FallbackStrategyV4.STOP_EXECUTION):
        """
        Initializes the RealtimeGuardrailV4.

        Args:
            policy_layer: The PolicyLayer to evaluate actions.
            default_strategy: The default fallback strategy if an action is denied.
        """
        self.policy_layer = policy_layer
        self.default_strategy = default_strategy
        self._dynamic_handlers: Dict[str, Callable] = {}

    def register_fallback_handler(self, tool_name: str, handler: Callable) -> None:
        """
        Registers a dynamic fallback handler for a specific tool.

        Args:
            tool_name: The name of the tool.
            handler: A callable that returns the fallback result. Can be async.
        """
        self._dynamic_handlers[tool_name] = handler

    def check_action(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str, FallbackStrategyV4]:
        """
        Checks an action against the policy and returns if it's allowed,
        an explanation, and the fallback strategy to apply if denied.

        Args:
            tool_name: The name of the tool to evaluate.
            **kwargs: The arguments to pass to the tool.

        Returns:
            A tuple containing a boolean indicating if the action is allowed,
            an explanation string, and the fallback strategy.
        """
        allow, explanation = self.policy_layer.evaluate(tool_name, **kwargs)

        if allow:
            return True, explanation, FallbackStrategyV4.NONE

        if tool_name in self._dynamic_handlers:
            strategy = FallbackStrategyV4.DYNAMIC_HANDLER
        else:
            strategy = self.default_strategy

        logging.warning(f"RealtimeGuardrailV4: Violation detected for '{tool_name}'. Strategy: {strategy.value}. Reason: {explanation}")

        return False, explanation, strategy

    def execute_with_guardrails(self, tool_func: Callable, tool_name: str, **kwargs: Any) -> Any:
        """
        Executes the given tool function if allowed by the guardrails policy.
        If blocked, executes the fallback strategy. Gracefully degrades on exceptions.
        Supports both synchronous and asynchronous tool functions.

        Args:
            tool_func (Callable): The tool function to execute.
            tool_name (str): The name of the tool.
            **kwargs: Arguments to pass to the tool function.

        Returns:
            Any: The result of the tool execution, or a fallback message if blocked.

        Raises:
            SecurityViolationError: If the fallback strategy is STOP_EXECUTION.
        """
        allow, explanation, strategy = self.check_action(tool_name, **kwargs)

        def handle_sync_fallback() -> Any:
            if strategy == FallbackStrategyV4.STOP_EXECUTION:
                raise SecurityViolationError(f"Action '{tool_name}' blocked: {explanation}")
            elif strategy == FallbackStrategyV4.REQUEST_REVIEW:
                return f"Review requested for action '{tool_name}': {explanation}"
            elif strategy == FallbackStrategyV4.WARN_AND_CONTINUE:
                return f"Warning: {explanation}. Action '{tool_name}' skipped."
            elif strategy == FallbackStrategyV4.DYNAMIC_HANDLER:
                handler = self._dynamic_handlers[tool_name]
                if asyncio.iscoroutinefunction(handler):
                    raise ValueError(f"Sync execution expected for tool '{tool_name}', but dynamic handler is async.")
                return handler(tool_name, explanation, **kwargs)
            return f"Action '{tool_name}' blocked: {explanation}"

        async def handle_async_fallback() -> Any:
            if strategy == FallbackStrategyV4.STOP_EXECUTION:
                raise SecurityViolationError(f"Action '{tool_name}' blocked: {explanation}")
            elif strategy == FallbackStrategyV4.REQUEST_REVIEW:
                return f"Review requested for action '{tool_name}': {explanation}"
            elif strategy == FallbackStrategyV4.WARN_AND_CONTINUE:
                return f"Warning: {explanation}. Action '{tool_name}' skipped."
            elif strategy == FallbackStrategyV4.DYNAMIC_HANDLER:
                handler = self._dynamic_handlers[tool_name]
                if asyncio.iscoroutinefunction(handler):
                    return await handler(tool_name, explanation, **kwargs)
                return handler(tool_name, explanation, **kwargs)
            return f"Action '{tool_name}' blocked: {explanation}"

        is_async = asyncio.iscoroutinefunction(tool_func)

        if not allow:
            if is_async:
                return handle_async_fallback()
            return handle_sync_fallback()

        if is_async:
            async def async_exec() -> Any:
                try:
                    return await tool_func(**kwargs)
                except Exception as e:
                    logging.error(f"Error executing tool '{tool_name}': {e}")
                    # Attempt graceful degradation fallback
                    if tool_name in self._dynamic_handlers:
                        handler = self._dynamic_handlers[tool_name]
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                return await handler(tool_name, f"Exception during execution: {e}", **kwargs)
                            return handler(tool_name, f"Exception during execution: {e}", **kwargs)
                        except Exception as fallback_e:
                            logging.error(f"Fallback handler for '{tool_name}' failed: {fallback_e}")
                    return f"Action '{tool_name}' failed during execution: {str(e)}"
            return async_exec()
        else:
            try:
                return tool_func(**kwargs)
            except Exception as e:
                logging.error(f"Error executing tool '{tool_name}': {e}")
                # Attempt graceful degradation fallback
                if tool_name in self._dynamic_handlers:
                    handler = self._dynamic_handlers[tool_name]
                    try:
                        if asyncio.iscoroutinefunction(handler):
                             raise ValueError(f"Sync execution failed for tool '{tool_name}', but dynamic handler is async.")
                        return handler(tool_name, f"Exception during execution: {e}", **kwargs)
                    except Exception as fallback_e:
                         logging.error(f"Fallback handler for '{tool_name}' failed: {fallback_e}")
                return f"Action '{tool_name}' failed during execution: {str(e)}"
