import logging
import asyncio
from enum import Enum
from typing import Any, Tuple, Optional, Callable
from magda_agent.safety.policy import PolicyLayer

class FallbackStrategy(Enum):
    STOP_EXECUTION = "stop_execution"
    REQUEST_REVIEW = "request_review"
    WARN_AND_CONTINUE = "warn_and_continue"
    NONE = "none"

class SecurityViolationError(Exception):
    pass

class RealtimeGuardrail:
    """
    Realtime guardrails that trigger a safe fallback action immediately
    when a policy violation is detected mid-execution.
    """

    def __init__(self, policy_layer: PolicyLayer, default_strategy: FallbackStrategy = FallbackStrategy.STOP_EXECUTION):
        """
        Initializes the RealtimeGuardrail.

        Args:
            policy_layer: The PolicyLayer to evaluate actions.
            default_strategy: The default fallback strategy if an action is denied.
        """
        self.policy_layer = policy_layer
        self.default_strategy = default_strategy

    def check_action(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str, FallbackStrategy]:
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
            return True, explanation, FallbackStrategy.NONE

        # Determine fallback strategy based on tool or context
        # For now, use the default strategy
        strategy = self.default_strategy

        logging.warning(f"RealtimeGuardrail: Violation detected for '{tool_name}'. Strategy: {strategy.value}. Reason: {explanation}")

        return False, explanation, strategy

    def execute_with_guardrails(self, tool_func: Callable, tool_name: str, **kwargs: Any) -> Any:
        """
        Executes the given tool function if allowed by the guardrails policy.
        If blocked, executes the fallback strategy.
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

        def handle_fallback() -> Any:
            if strategy == FallbackStrategy.STOP_EXECUTION:
                raise SecurityViolationError(f"Action '{tool_name}' blocked: {explanation}")
            elif strategy == FallbackStrategy.REQUEST_REVIEW:
                return f"Review requested for action '{tool_name}': {explanation}"
            elif strategy == FallbackStrategy.WARN_AND_CONTINUE:
                return f"Warning: {explanation}. Action '{tool_name}' skipped."
            return f"Action '{tool_name}' blocked: {explanation}"

        if not allow:
            if asyncio.iscoroutinefunction(tool_func):
                async def async_fallback() -> Any:
                    return handle_fallback()
                return async_fallback()
            return handle_fallback()

        if asyncio.iscoroutinefunction(tool_func):
            async def async_exec() -> Any:
                try:
                    return await tool_func(**kwargs)
                except Exception as e:
                    logging.error(f"Error executing tool '{tool_name}': {e}")
                    # Return sensible fallback on exception
                    return f"Action '{tool_name}' failed during execution: {str(e)}"
            return async_exec()
        else:
            try:
                return tool_func(**kwargs)
            except Exception as e:
                logging.error(f"Error executing tool '{tool_name}': {e}")
                # Return sensible fallback on exception
                return f"Action '{tool_name}' failed during execution: {str(e)}"
