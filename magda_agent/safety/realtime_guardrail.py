"""
MCP Realtime Guardrail Fallback module.
Implements a safety/execution fallback layer inspired by OpenAI Agents SDK realtime guardrails.
Intercepts policy violations and tool execution failures, producing dynamic reprompt prompts
to let the agent recover gracefully instead of hard-failing.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

from magda_agent.safety.policy import PolicyLayer


class MCPRealtimeGuardrailFallback:
    """
    Fallback layer for tool invocation that catches safety violations and execution
    failures, generating dynamic reprompt prompts for recovery.
    """

    def __init__(self, policy_layer: Optional[PolicyLayer] = None) -> None:
        """
        Initializes the MCPRealtimeGuardrailFallback.

        Args:
            policy_layer (Optional[PolicyLayer]): The policy evaluator to verify safety before execution.
        """
        self.policy_layer = policy_layer or PolicyLayer()

    async def execute_with_reprompt_fallback(
        self,
        tool_func: Callable[..., Any],
        tool_name: str,
        kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Executes the tool function with guardrails. If a violation is detected or an error occurs,
        intercepts the issue and returns a structured tuple containing False and a dynamic reprompt prompt.

        Args:
            tool_func (Callable[..., Any]): The synchronous or asynchronous function representing the tool call.
            tool_name (str): Name of the tool.
            kwargs (Dict[str, Any]): Arguments passed to the tool.

        Returns:
            Tuple[bool, str]: (Success, Result or Reprompt prompt).
        """
        # 1. Policy check before tool execution
        allow, explanation = self.policy_layer.evaluate(tool_name, **kwargs)
        if not allow:
            logging.warning(f"RealtimeGuardrail: Intercepted policy violation for '{tool_name}'.")
            fallback_prompt = (
                f"SAFETY ALERT: The execution of tool '{tool_name}' with arguments {kwargs} "
                f"was blocked due to a policy violation: {explanation}. "
                f"Please analyze the violation, dynamically revise your execution plan, and try a "
                f"different, safe approach without violating safety policies."
            )
            return False, fallback_prompt

        # 2. Execute the tool with try/except
        try:
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**kwargs)
            else:
                # Run synchronous tool in an executor to avoid blocking the asyncio event loop
                result = await asyncio.to_thread(tool_func, **kwargs)
            return True, str(result)
        except Exception as e:
            logging.error(f"RealtimeGuardrail: Intercepted tool execution failure for '{tool_name}'. Error: {e}")
            fallback_prompt = (
                f"EXECUTION FAILURE: Failed to execute tool '{tool_name}' with arguments {kwargs}. "
                f"Error: {str(e)}. "
                f"Please adjust your strategy, handle this error dynamically, and generate an "
                f"alternative plan or fallback steps."
            )
            return False, fallback_prompt
