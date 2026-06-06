import logging
from enum import Enum
from typing import Any, Tuple, Optional
from magda_agent.safety.policy import PolicyLayer

class FallbackStrategy(Enum):
    STOP_EXECUTION = "stop_execution"
    REQUEST_REVIEW = "request_review"
    NONE = "none"

class RealtimeGuardrail:
    """
    Realtime guardrails that trigger a safe fallback action immediately
    when a policy violation is detected mid-execution.
    """

    def __init__(self, policy_layer: PolicyLayer, default_strategy: FallbackStrategy = FallbackStrategy.STOP_EXECUTION):
        self.policy_layer = policy_layer
        self.default_strategy = default_strategy

    def check_action(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str, FallbackStrategy]:
        """
        Checks an action against the policy and returns if it's allowed,
        an explanation, and the fallback strategy to apply if denied.
        """
        allow, explanation = self.policy_layer.evaluate(tool_name, **kwargs)

        if allow:
            return True, explanation, FallbackStrategy.NONE

        # Determine fallback strategy based on tool or context
        # For now, use the default strategy
        strategy = self.default_strategy

        logging.warning(f"RealtimeGuardrail: Violation detected for '{tool_name}'. Strategy: {strategy.value}. Reason: {explanation}")

        return False, explanation, strategy
