import logging
import asyncio
from typing import Any, Callable, Dict, Tuple, List, Optional
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.guardrails import FallbackStrategy, SecurityViolationError

class DynamicRule:
    """
    Interface for a dynamic context rule.
    """
    def evaluate(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """
        Evaluates the tool call.
        Returns (True, "Allowed...") or (False, "Denied because...").
        """
        return True, "Default allow."

class DynamicPolicyManager:
    """
    Manages and evaluates dynamic context rules.
    """
    def __init__(self) -> None:
        self.rules: List[DynamicRule] = []

    def add_rule(self, rule: DynamicRule) -> None:
        """
        Registers a dynamic rule.
        """
        self.rules.append(rule)

    def evaluate(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """
        Evaluates all rules. If any rule denies, returns False and its explanation.
        """
        for rule in self.rules:
            allow, explanation = rule.evaluate(tool_name, **kwargs)
            if not allow:
                return False, explanation
        return True, "All dynamic rules allowed."

class AgentGuardRuntimePolicyV5:
    """
    A robust runtime policy layer that intercepts tool executions based on
    dynamic context rules, falling back to the static policy layer if allowed.
    """
    def __init__(self, policy_layer: PolicyLayer, default_strategy: FallbackStrategy = FallbackStrategy.STOP_EXECUTION):
        self.policy_layer = policy_layer
        self.dynamic_manager = DynamicPolicyManager()
        self.default_strategy = default_strategy

    def add_dynamic_rule(self, rule: DynamicRule) -> None:
        """
        Adds a dynamic rule to the policy manager.
        """
        self.dynamic_manager.add_rule(rule)

    def check_action(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str, FallbackStrategy]:
        """
        Checks dynamic rules first, then static policy.
        """
        dyn_allow, dyn_expl = self.dynamic_manager.evaluate(tool_name, **kwargs)
        if not dyn_allow:
            logging.warning(f"AgentGuardRuntimePolicyV5: Dynamic rule blocked '{tool_name}'. Reason: {dyn_expl}")
            return False, dyn_expl, self.default_strategy

        stat_allow, stat_expl = self.policy_layer.evaluate(tool_name, **kwargs)
        if not stat_allow:
            logging.warning(f"AgentGuardRuntimePolicyV5: Static rule blocked '{tool_name}'. Reason: {stat_expl}")
            return False, stat_expl, self.default_strategy

        return True, "Allowed", FallbackStrategy.NONE

    def execute_with_guardrails(self, tool_func: Callable, tool_name: str, **kwargs: Any) -> Any:
        """
        Executes the given tool function if allowed by dynamic and static policies.
        Supports both sync and async.
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
                except asyncio.CancelledError:
                    logging.warning(f"Action '{tool_name}' interrupted (CancelledError).")
                    raise
                except Exception as e:
                    logging.error(f"Error executing tool '{tool_name}': {e}")
                    return f"Action '{tool_name}' failed during execution: {str(e)}"
            return async_exec()
        else:
            try:
                return tool_func(**kwargs)
            except KeyboardInterrupt:
                logging.warning(f"Action '{tool_name}' interrupted (KeyboardInterrupt).")
                raise
            except Exception as e:
                logging.error(f"Error executing tool '{tool_name}': {e}")
                return f"Action '{tool_name}' failed during execution: {str(e)}"
