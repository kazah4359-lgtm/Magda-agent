import logging
from typing import Dict, Any, Tuple, Optional
from magda_agent.safety.policy import PolicyLayer

class ACSRuntimeGuardV5:
    """
    ACS Checkpoint 3 (Runtime Safety) policy framework.
    Evaluates tool calls dynamically using the underlying PolicyLayer.
    """

    def __init__(self, policy_layer: Optional[PolicyLayer] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.policy_layer = policy_layer or PolicyLayer()

    def checkpoint_3_tool_policy(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates the tool call using the policy layer.

        Args:
            workflow_data: The context data including 'tool' and 'kwargs'.

        Returns:
            Tuple[bool, str]: (Passed, Reason)
        """
        tool = workflow_data.get("tool")
        if not tool:
            return False, "Tool policy failed: missing 'tool' field."

        if tool == "forbidden_tool":
            return False, f"Tool policy failed: tool '{tool}' is explicitly forbidden."

        kwargs = workflow_data.get("kwargs", {})
        if not isinstance(kwargs, dict):
            return False, "Tool policy failed: 'kwargs' must be a dictionary."

        allow, explanation = self.policy_layer.evaluate(tool, **kwargs)
        if not allow:
            return False, f"Tool policy failed: {explanation}"

        return True, "Tool policy passed."

    def evaluate(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Main entry point for evaluating runtime safety.
        Delegates to checkpoint_3_tool_policy.
        """
        return self.checkpoint_3_tool_policy(workflow_data)
