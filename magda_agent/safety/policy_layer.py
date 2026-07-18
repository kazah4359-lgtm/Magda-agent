"""Agent Guard Policy Extension for granular runtime governance."""

import logging
from typing import Any, Callable, Dict, List, Tuple

from magda_agent.safety.policy import PolicyLayer

# A granular rule evaluates a tool execution.
# Returns (True, "") if allowed, or (False, "Reason...") if denied.
GranularRule = Callable[[str, Dict[str, Any]], Tuple[bool, str]]


class AgentGuardPolicyLayer(PolicyLayer):
    """
    Extended PolicyLayer inspired by Agent Guard runtime governance.
    Supports granular runtime checks in addition to base policies.
    """

    def __init__(self) -> None:
        """Initialize the extended policy layer with empty rulesets."""
        super().__init__()
        self._global_rules: List[GranularRule] = []
        self._tool_rules: Dict[str, List[GranularRule]] = {}

    def add_global_rule(self, rule: GranularRule) -> None:
        """
        Add a global rule that applies to all tool executions.

        Args:
            rule: A callable rule evaluating the tool name and arguments.
        """
        self._global_rules.append(rule)

    def add_tool_rule(self, tool_name: str, rule: GranularRule) -> None:
        """
        Add a rule specific to a canonical tool name.

        Args:
            tool_name: The canonical tool name.
            rule: A callable rule evaluating the tool name and arguments.
        """
        canonical = self._canonical_tool_name(tool_name)
        if canonical not in self._tool_rules:
            self._tool_rules[canonical] = []
        self._tool_rules[canonical].append(rule)

    def evaluate(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """
        Evaluate an action by first running granular rules, then falling back to base policy.

        Args:
            tool_name: The tool or action being executed.
            kwargs: Arguments provided to the tool.

        Returns:
            A tuple containing a boolean (allow/deny) and an explanation string.
        """
        # 1. Run global rules
        for rule in self._global_rules:
            allow, explanation = rule(tool_name, kwargs)
            if not allow:
                self._log_decision(tool_name, kwargs, explanation)
                return allow, explanation

        # 2. Run tool-specific rules
        canonical_tool = self._canonical_tool_name(tool_name)
        for rule in self._tool_rules.get(canonical_tool, []):
            allow, explanation = rule(tool_name, kwargs)
            if not allow:
                self._log_decision(tool_name, kwargs, explanation)
                return allow, explanation

        # 3. Fallback to base PolicyLayer checks
        return super().evaluate(tool_name, **kwargs)

    def _log_decision(
        self, tool_name: str, kwargs: Dict[str, Any], explanation: str
    ) -> None:
        """
        Log a denied action via the AuditLogger.

        Args:
            tool_name: Name of the tool.
            kwargs: Tool arguments.
            explanation: Reason for the decision.
        """
        self.audit_logger.log_call(
            tool_name=tool_name,
            kwargs=kwargs,
            why=kwargs.get("why", "Granular rule intervention"),
            result={"allowed": False, "explanation": explanation},
            duration=0.0,
        )

        logging.warning(f"AgentGuardPolicyLayer: DENY - {tool_name} with args {kwargs}. Reason: {explanation}")
