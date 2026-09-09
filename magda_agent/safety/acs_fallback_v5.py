import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

from magda_agent.safety.acs_controls_v5 import ACSControlsV5
from magda_agent.safety.acs_state_transition_v5 import ACSStateTransitionV5
from magda_agent.safety.guardrails import SecurityViolationError

DEFAULT_NEUTRAL_RESPONSE = "I cannot provide this output as it failed safety validation and state checks."


class ACSControlFallbackV5:
    """
    ACS Control Fallback Strategy V5.
    Provides graceful fallback to a neutral response when a tool's output validation
    or state check is denied.
    """

    def __init__(
        self,
        acs_controls: Optional[ACSControlsV5] = None,
        state_transition: Optional[ACSStateTransitionV5] = None,
        neutral_response: Union[str, Dict[str, Any]] = DEFAULT_NEUTRAL_RESPONSE,
    ) -> None:
        """
        Initializes ACSControlFallbackV5.

        Args:
            acs_controls: Optional ACSControlsV5 instance for running output sanitization checks.
            state_transition: Optional ACSStateTransitionV5 instance for state checks.
            neutral_response: Custom neutral response to return on validation denial.
        """
        self.logger = logging.getLogger(__name__)
        self.acs_controls = acs_controls or ACSControlsV5()
        self.state_transition = state_transition or ACSStateTransitionV5()
        self.neutral_response = neutral_response

    def validate_state_and_output(
        self, action_data: Dict[str, Any], output: Any
    ) -> Tuple[bool, str]:
        """
        Validates both the state transition and output sanitization for the given action data and output.

        Args:
            action_data: Dictionary containing action metadata (e.g., tool_name, current_state, next_state).
            output: The output returned by tool execution.

        Returns:
            Tuple[bool, str]: (passed, reason)
        """
        # 1. State check validation
        state_passed, state_reason = self.state_transition.checkpoint_4_state_transition(
            action_data
        )
        if not state_passed:
            return False, f"State Check Denied: {state_reason}"

        # 2. Output sanitization validation
        output_passed, output_reason = self.acs_controls.checkpoint_5_output_sanitization(
            output
        )
        if not output_passed:
            return False, f"Output Validation Denied: {output_reason}"

        return True, "Output validation and state check passed."

    def get_neutral_response(self, reason: Optional[str] = None) -> Any:
        """
        Returns the configured neutral response.

        Args:
            reason: Optional explanation string for why fallback was triggered.

        Returns:
            Any: The neutral response payload or string.
        """
        if isinstance(self.neutral_response, dict):
            resp = dict(self.neutral_response)
            if reason and "reason" not in resp:
                resp["reason"] = reason
            return resp
        return self.neutral_response

    async def execute_with_fallback_async(
        self, tool_func: Callable[..., Any], action_data: Dict[str, Any]
    ) -> Tuple[bool, Any]:
        """
        Asynchronously executes the tool function with ACS state check and output validation.
        If validation is denied or raises a SecurityViolationError, falls back to a neutral response.

        Args:
            tool_func: The callable tool function to execute.
            action_data: Metadata for the action (e.g. kwargs, current_state, next_state).

        Returns:
            Tuple[bool, Any]: (is_success, output_or_neutral_fallback)
        """
        kwargs = action_data.get("kwargs", {})
        try:
            if asyncio.iscoroutinefunction(tool_func):
                raw_output = await tool_func(**kwargs)
            else:
                raw_output = await asyncio.to_thread(tool_func, **kwargs)

            passed, reason = self.validate_state_and_output(action_data, raw_output)
            if not passed:
                self.logger.warning(
                    f"ACS Control Fallback V5: Output validation or state check denied. {reason}. Falling back to neutral response."
                )
                return False, self.get_neutral_response(reason)

            return True, raw_output

        except SecurityViolationError as sve:
            self.logger.warning(
                f"ACS Control Fallback V5: Security violation caught: {sve}. Falling back to neutral response."
            )
            return False, self.get_neutral_response(str(sve))
        except Exception as e:
            self.logger.error(
                f"ACS Control Fallback V5: Unexpected error during execution: {e}."
            )
            raise

    def execute_with_fallback(
        self, tool_func: Callable[..., Any], action_data: Dict[str, Any]
    ) -> Tuple[bool, Any]:
        """
        Synchronously executes the tool function with ACS state check and output validation.
        If validation is denied or raises a SecurityViolationError, falls back to a neutral response.

        Args:
            tool_func: The callable tool function to execute.
            action_data: Metadata for the action.

        Returns:
            Tuple[bool, Any]: (is_success, output_or_neutral_fallback)
        """
        kwargs = action_data.get("kwargs", {})
        try:
            if asyncio.iscoroutinefunction(tool_func):
                raw_output = asyncio.run(tool_func(**kwargs))
            else:
                raw_output = tool_func(**kwargs)

            passed, reason = self.validate_state_and_output(action_data, raw_output)
            if not passed:
                self.logger.warning(
                    f"ACS Control Fallback V5: Output validation or state check denied. {reason}. Falling back to neutral response."
                )
                return False, self.get_neutral_response(reason)

            return True, raw_output

        except SecurityViolationError as sve:
            self.logger.warning(
                f"ACS Control Fallback V5: Security violation caught: {sve}. Falling back to neutral response."
            )
            return False, self.get_neutral_response(str(sve))


# Alias for compatibility
ACSFallbackV5 = ACSControlFallbackV5
