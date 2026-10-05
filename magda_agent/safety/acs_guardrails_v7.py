import asyncio
import inspect
import logging
import re
from typing import Dict, Any, Tuple, Optional, Callable, List, Union

from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.safety.acs_persistence import ACSPersistence
from magda_agent.safety.acs_guard_runtime_v7 import ACSGuardRuntimeV7, SecurityViolationError
from magda_agent.safety.taint import is_tainted


class ACSGuardrailViolationError(SecurityViolationError):
    """Exception raised when an action or execution is blocked by ACS Guardrails V7 policy rules."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ACSGuardrailsV7:
    """
    ACS (Agent Control Specification) Compliance Policy Guardrails V7.
    Provides a centralized policy validation layer prior to tool execution,
    supporting both synchronous and asynchronous tool invocations with dynamic rule registration.
    """

    def __init__(
        self,
        policy_layer: Optional[PolicyLayer] = None,
        audit_trail: Optional[AuditTrail] = None,
        persistence: Optional[ACSPersistence] = None,
        custom_policy_rules: Optional[List[Callable]] = None
    ) -> None:
        """
        Initializes ACS Guardrails V7.

        Args:
            policy_layer: PolicyLayer instance for standard policy evaluations.
            audit_trail: AuditTrail instance for audit logging.
            persistence: ACSPersistence instance for checkpoint state persistence.
            custom_policy_rules: List of custom sync/async policy rule functions.
        """
        self.logger = logging.getLogger(__name__)
        self.policy_layer = policy_layer or PolicyLayer()
        self.audit_trail = audit_trail or AuditTrail()
        self.persistence = persistence or ACSPersistence()
        self.acs_guard = ACSGuardRuntimeV7(
            policy_layer=self.policy_layer,
            audit_trail=self.audit_trail,
            persistence=self.persistence
        )
        self.custom_policy_rules: List[Callable] = custom_policy_rules or []

    def add_policy_rule(self, rule_fn: Callable) -> None:
        """
        Registers a custom policy rule callback (sync or async).
        Rule function signature: rule_fn(tool_name, kwargs, context) -> Tuple[bool, str]
        """
        if rule_fn not in self.custom_policy_rules:
            self.custom_policy_rules.append(rule_fn)

    def register_policy_rule(self, rule_fn: Callable) -> None:
        """Alias for add_policy_rule."""
        self.add_policy_rule(rule_fn)

    def validate_policy(
        self,
        tool_name: str,
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates the tool execution request against centralized policy rules before execution.

        Args:
            tool_name: Name of the tool to be executed.
            kwargs: Arguments dictionary passed to the tool.
            context: Additional workflow context.

        Returns:
            Tuple[bool, str, Dict[str, Any]]: (is_allowed, reason, details)
        """
        kwargs = kwargs or {}
        context = context or {}

        if not tool_name or not isinstance(tool_name, str):
            details = {"error": "Invalid tool name", "resolution": "Provide a non-empty string tool_name."}
            return False, "Policy validation failed: tool name must be a non-empty string.", details

        if is_tainted(tool_name):
            details = {"error": "Tainted tool name", "resolution": "Sanitize tool name."}
            return False, "Policy validation failed: tool name is tainted.", details

        if is_tainted(kwargs):
            details = {"error": "Tainted kwargs", "resolution": "Sanitize tool arguments."}
            return False, "Policy validation failed: tool arguments (kwargs) are tainted.", details

        workflow_data = {
            "action": context.get("action", "execute"),
            "tool": tool_name,
            "kwargs": kwargs,
            "current_state": context.get("current_state", "idle"),
            "next_state": context.get("next_state", "executing")
        }

        # 1. Evaluate ACS Guard checkpoints 1 to 3 (Input Validation, Intent Authorization, Tool Policy Layer)
        passed_1, reason_1 = self.acs_guard.checkpoint_1_input_validation(workflow_data)
        if not passed_1:
            return False, f"Policy validation failed: {reason_1}", {"checkpoint": 1, "reason": reason_1}

        passed_2, reason_2 = self.acs_guard.checkpoint_2_intent_authorization(workflow_data)
        if not passed_2:
            return False, f"Policy validation failed: {reason_2}", {"checkpoint": 2, "reason": reason_2}

        passed_3, reason_3 = self.acs_guard.checkpoint_3_tool_policy(workflow_data)
        if not passed_3:
            return False, f"Policy validation failed: {reason_3}", {"checkpoint": 3, "reason": reason_3}

        # 2. Evaluate custom sync policy rules
        for rule in self.custom_policy_rules:
            if not asyncio.iscoroutinefunction(rule):
                try:
                    res = rule(tool_name, kwargs, context)
                    if isinstance(res, tuple):
                        rule_passed, rule_reason = res[0], res[1]
                    else:
                        rule_passed, rule_reason = bool(res), "Custom rule denied execution"
                    if not rule_passed:
                        return False, f"Custom policy rule failed: {rule_reason}", {"rule": getattr(rule, '__name__', str(rule)), "reason": rule_reason}
                except Exception as e:
                    return False, f"Custom policy rule evaluation error: {e}", {"error": str(e)}

        return True, "Policy validation passed.", {"tool": tool_name}

    async def validate_policy_async(
        self,
        tool_name: str,
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Asynchronous variant of validate_policy, evaluating both sync and async custom policy rules.
        """
        kwargs = kwargs or {}
        context = context or {}

        # First run base sync checks
        passed, reason, details = self.validate_policy(tool_name, kwargs, context)
        if not passed:
            return False, reason, details

        # Evaluate custom async policy rules
        for rule in self.custom_policy_rules:
            if asyncio.iscoroutinefunction(rule) or inspect.isawaitable(rule):
                try:
                    res = await rule(tool_name, kwargs, context)
                    if isinstance(res, tuple):
                        rule_passed, rule_reason = res[0], res[1]
                    else:
                        rule_passed, rule_reason = bool(res), "Custom async rule denied execution"
                    if not rule_passed:
                        return False, f"Custom async policy rule failed: {rule_reason}", {"rule": getattr(rule, '__name__', str(rule)), "reason": rule_reason}
                except Exception as e:
                    return False, f"Custom async policy rule evaluation error: {e}", {"error": str(e)}

        return True, "Async policy validation passed.", {"tool": tool_name}

    def execute_with_guardrails(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        tool_func: Callable[..., Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Interprets and executes a tool with centralized pre-execution policy guardrails
        and post-execution output sanitization.

        Args:
            tool_name: Name of the tool.
            kwargs: Arguments dictionary for the tool.
            tool_func: Callable tool function to execute.
            context: Additional context data.

        Returns:
            Any: Result of the tool execution.

        Raises:
            ACSGuardrailViolationError: If pre-execution policy checks or post-execution sanitization fail.
        """
        context = context or {}
        passed, reason, details = self.validate_policy(tool_name, kwargs, context)
        if not passed:
            self.logger.warning(f"ACS Guardrails pre-execution policy blocked '{tool_name}': {reason}")
            self.audit_trail.log_call(
                tool_name=tool_name,
                kwargs=kwargs,
                why=f"Pre-execution policy violation: {reason}",
                result="blocked",
                duration=0.0
            )
            self.persistence.log_checkpoint(
                checkpoint_id=3,
                status="failed",
                reason=reason,
                workflow_context={"tool": tool_name, "kwargs": kwargs}
            )
            raise ACSGuardrailViolationError(f"Pre-execution policy violation: {reason}", details=details)

        try:
            output = tool_func(**kwargs)
        except Exception as e:
            self.logger.error(f"Tool execution '{tool_name}' failed with error: {e}")
            raise

        # Post-execution output sanitization
        workflow_data = {"tool": tool_name, "output": output}
        passed_out, reason_out = self.acs_guard.checkpoint_5_output_sanitization(workflow_data)
        if not passed_out:
            self.logger.warning(f"ACS Guardrails post-execution output sanitization blocked '{tool_name}': {reason_out}")
            self.audit_trail.log_call(
                tool_name=tool_name,
                kwargs=kwargs,
                why=f"Post-execution sanitization violation: {reason_out}",
                result="blocked",
                duration=0.0
            )
            self.persistence.log_checkpoint(
                checkpoint_id=5,
                status="failed",
                reason=reason_out,
                workflow_context=workflow_data
            )
            raise ACSGuardrailViolationError(
                f"Post-execution output sanitization violation: {reason_out}",
                details={"checkpoint": 5, "reason": reason_out}
            )

        self.audit_trail.log_call(
            tool_name=tool_name,
            kwargs=kwargs,
            why="Tool execution allowed and output sanitized by ACS Guardrails V7.",
            result="allowed",
            duration=0.0
        )
        return output

    async def execute_with_guardrails_async(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        tool_func: Callable[..., Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Asynchronously executes a tool wrapped with pre-execution policy guardrails
        and post-execution output sanitization. Supports AsyncMock and coroutines.

        Args:
            tool_name: Name of the tool.
            kwargs: Arguments dictionary for the tool.
            tool_func: Async or sync callable tool function to execute.
            context: Additional context data.

        Returns:
            Any: Result of the tool execution.

        Raises:
            ACSGuardrailViolationError: If pre-execution policy checks or post-execution sanitization fail.
        """
        context = context or {}
        passed, reason, details = await self.validate_policy_async(tool_name, kwargs, context)
        if not passed:
            self.logger.warning(f"ACS Guardrails async pre-execution policy blocked '{tool_name}': {reason}")
            self.audit_trail.log_call(
                tool_name=tool_name,
                kwargs=kwargs,
                why=f"Pre-execution async policy violation: {reason}",
                result="blocked",
                duration=0.0
            )
            self.persistence.log_checkpoint(
                checkpoint_id=3,
                status="failed",
                reason=reason,
                workflow_context={"tool": tool_name, "kwargs": kwargs}
            )
            raise ACSGuardrailViolationError(f"Pre-execution policy violation: {reason}", details=details)

        try:
            res = tool_func(**kwargs)
            if inspect.isawaitable(res):
                output = await res
            else:
                output = res
        except Exception as e:
            self.logger.error(f"Async tool execution '{tool_name}' failed with error: {e}")
            raise

        # Post-execution output sanitization
        workflow_data = {"tool": tool_name, "output": output}
        passed_out, reason_out = self.acs_guard.checkpoint_5_output_sanitization(workflow_data)
        if not passed_out:
            self.logger.warning(f"ACS Guardrails async post-execution output sanitization blocked '{tool_name}': {reason_out}")
            self.audit_trail.log_call(
                tool_name=tool_name,
                kwargs=kwargs,
                why=f"Post-execution sanitization violation: {reason_out}",
                result="blocked",
                duration=0.0
            )
            self.persistence.log_checkpoint(
                checkpoint_id=5,
                status="failed",
                reason=reason_out,
                workflow_context=workflow_data
            )
            raise ACSGuardrailViolationError(
                f"Post-execution output sanitization violation: {reason_out}",
                details={"checkpoint": 5, "reason": reason_out}
            )

        self.audit_trail.log_call(
            tool_name=tool_name,
            kwargs=kwargs,
            why="Async tool execution allowed and output sanitized by ACS Guardrails V7.",
            result="allowed",
            duration=0.0
        )
        return output


ACSGuardrails = ACSGuardrailsV7
ACSCompliancePolicyGuardrailsV7 = ACSGuardrailsV7
ACSGuardrailsPolicyV7 = ACSGuardrailsV7
