"""A2A MCPKernel Guardrail Runtime Checkpoint V4.

Provides dynamic runtime execution hooks for A2A peer task delegations,
evaluating payload parameters against ACS safety checkpoints and MCPKernel
taint tracking sandboxes before executing tool calls.
"""

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

from magda_agent.safety.acs_checkpoints_v4 import ACSCheckpointsV4
from magda_agent.safety.mcpkernel_sandbox_v3 import MCPKernelSandboxV3, TaintTrackerV3
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.taint_tracking_v2 import PolicyViolationError

logger = logging.getLogger(__name__)


class A2AMCPKernelGuardV4:
    """A2A MCPKernel Guardrail V4.

    Evaluates A2A peer delegation tool call payloads against ACS 5-checkpoint
    validations and MCPKernel taint tracking policies dynamically during execution.
    """

    def __init__(
        self,
        checkpoints: Optional[ACSCheckpointsV4] = None,
        sandbox: Optional[MCPKernelSandboxV3] = None,
        tracker: Optional[TaintTrackerV3] = None,
        policy_layer: Optional[PolicyLayer] = None,
    ) -> None:
        """Initialize A2AMCPKernelGuardV4.

        Args:
            checkpoints: Optional ACSCheckpointsV4 instance.
            sandbox: Optional MCPKernelSandboxV3 instance.
            tracker: Optional TaintTrackerV3 instance.
            policy_layer: Optional PolicyLayer instance.
        """
        self.tracker = tracker or TaintTrackerV3()
        self.sandbox = sandbox or MCPKernelSandboxV3(tracker=self.tracker)
        self.policy_layer = policy_layer or PolicyLayer()
        self.checkpoints = checkpoints or ACSCheckpointsV4(
            policy_layer=self.policy_layer
        )

    def evaluate_payload(
        self,
        action_name: str,
        tool_name: str,
        kwargs: Dict[str, Any],
        state: str = "idle",
        next_state: Optional[str] = None,
        is_sensitive: bool = False,
        sensitive_args: Optional[Set[str]] = None,
    ) -> Tuple[bool, str]:
        """Evaluates payload parameters through safety checkpoints and taint checks.

        Args:
            action_name: The action name (e.g., 'execute', 'read', 'delegate').
            tool_name: The tool function name.
            kwargs: Input parameters for the tool call.
            state: Current state in agent workflow.
            next_state: Optional target state transition.
            is_sensitive: If True, tool is sensitive and any tainted input fails.
            sensitive_args: Optional set of argument names considered sensitive.

        Returns:
            Tuple of (is_passed, message).
        """
        action_data = {
            "action_name": action_name,
            "tool_name": tool_name,
            "kwargs": kwargs,
            "state": state,
            "next_state": next_state,
        }

        # 1. Pre-execution ACS Checkpoints (Input, Intent, Policy, State)
        passed, reason = self.checkpoints.validate_pre_execution(action_data)
        if not passed:
            logger.warning(f"A2A ACS Checkpoint failed for '{tool_name}': {reason}")
            return False, reason

        # 2. MCPKernel Taint Tracking Checks
        try:
            self.sandbox.check_sensitive_arguments(
                kwargs, is_sensitive=is_sensitive, sensitive_args=sensitive_args
            )
        except PolicyViolationError as e:
            logger.warning(f"A2A MCPKernel Taint check failed for '{tool_name}': {e}")
            return False, str(e)

        return True, "A2A payload evaluation passed"

    def execute_hook(
        self,
        tool_func: Callable[..., Any],
        action_name: str,
        tool_name: str,
        kwargs: Dict[str, Any],
        state: str = "idle",
        next_state: Optional[str] = None,
        is_sensitive: bool = False,
        sensitive_args: Optional[Set[str]] = None,
    ) -> Any:
        """Executes a synchronous tool call dynamically through A2A safety checkpoints.

        Args:
            tool_func: The tool function to execute.
            action_name: Action type (e.g. 'execute').
            tool_name: Name of tool being called.
            kwargs: Inputs passed to tool call.
            state: Current state.
            next_state: Target state transition.
            is_sensitive: Whether tool call is sensitive.
            sensitive_args: Optional set of sensitive argument names.

        Returns:
            The tool execution result.

        Raises:
            PolicyViolationError: If ACS checkpoints or taint tracking checks fail.
        """
        passed, reason = self.evaluate_payload(
            action_name=action_name,
            tool_name=tool_name,
            kwargs=kwargs,
            state=state,
            next_state=next_state,
            is_sensitive=is_sensitive,
            sensitive_args=sensitive_args,
        )
        if not passed:
            raise PolicyViolationError(f"A2A execution blocked: {reason}")

        # Execute in MCPKernel Sandbox
        result = self.sandbox.execute(
            tool_func,
            kwargs,
            is_sensitive=is_sensitive,
            sensitive_args=sensitive_args,
        )

        # 3. Post-execution ACS Output Sanitization Checkpoint
        action_data = {
            "action_name": action_name,
            "tool_name": tool_name,
            "kwargs": kwargs,
            "output": result,
        }
        post_passed, post_reason = self.checkpoints.validate_post_execution(action_data)
        if not post_passed:
            raise PolicyViolationError(f"A2A output evaluation failed: {post_reason}")

        return result

    async def execute_hook_async(
        self,
        tool_func: Callable[..., Any],
        action_name: str,
        tool_name: str,
        kwargs: Dict[str, Any],
        state: str = "idle",
        next_state: Optional[str] = None,
        is_sensitive: bool = False,
        sensitive_args: Optional[Set[str]] = None,
    ) -> Any:
        """Executes an asynchronous tool call dynamically through A2A safety checkpoints.

        Args:
            tool_func: The async tool function to execute.
            action_name: Action type (e.g. 'execute').
            tool_name: Name of tool being called.
            kwargs: Inputs passed to tool call.
            state: Current state.
            next_state: Target state transition.
            is_sensitive: Whether tool call is sensitive.
            sensitive_args: Optional set of sensitive argument names.

        Returns:
            The awaited tool execution result.

        Raises:
            PolicyViolationError: If ACS checkpoints or taint tracking checks fail.
        """
        passed, reason = self.evaluate_payload(
            action_name=action_name,
            tool_name=tool_name,
            kwargs=kwargs,
            state=state,
            next_state=next_state,
            is_sensitive=is_sensitive,
            sensitive_args=sensitive_args,
        )
        if not passed:
            raise PolicyViolationError(f"A2A execution blocked: {reason}")

        # Execute in MCPKernel Sandbox asynchronously
        if inspect.iscoroutinefunction(tool_func):
            result = await self.sandbox.execute_async(
                tool_func,
                kwargs,
                is_sensitive=is_sensitive,
                sensitive_args=sensitive_args,
            )
        else:
            result = self.sandbox.execute(
                tool_func,
                kwargs,
                is_sensitive=is_sensitive,
                sensitive_args=sensitive_args,
            )

        # Post-execution ACS Output Sanitization Checkpoint
        action_data = {
            "action_name": action_name,
            "tool_name": tool_name,
            "kwargs": kwargs,
            "output": result,
        }
        post_passed, post_reason = self.checkpoints.validate_post_execution(action_data)
        if not post_passed:
            raise PolicyViolationError(f"A2A output evaluation failed: {post_reason}")

        return result


# Class alias for backward compatibility / flexibility
A2AMCPKernelGuard = A2AMCPKernelGuardV4
