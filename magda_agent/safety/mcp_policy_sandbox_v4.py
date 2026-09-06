"""MCP Tool Runtime Execution Policy Sandbox V4.

Provides a tight runtime policy evaluator sandbox to intercept action tools
and prevent unverified side-effects.
"""

import asyncio
import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from magda_agent.safety.guardrails import SecurityViolationError

logger = logging.getLogger(__name__)


class MCPPolicySandboxViolationError(SecurityViolationError):
    """Raised when an MCP action tool call is blocked by the runtime policy sandbox."""

    pass


class SandboxPolicyRule:
    """Base class / Interface for sandbox policy evaluation rules."""

    def evaluate(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """Evaluates whether the tool execution should be allowed.

        Args:
            tool_name: The name of the tool to evaluate.
            **kwargs: Arguments supplied for the tool execution.

        Returns:
            A tuple of (is_allowed: bool, reason: str).
        """
        return True, "Default allow."


class SideEffectPolicyRule(SandboxPolicyRule):
    """Rule that intercepts state-mutating action tools and enforces verification."""

    DEFAULT_SIDE_EFFECT_PREFIXES = (
        "write_",
        "delete_",
        "update_",
        "execute_",
        "mutate_",
        "post_",
        "put_",
        "remove_",
        "create_",
    )

    def __init__(
        self,
        prefixes: Optional[Tuple[str, ...]] = None,
        require_verification: bool = True,
    ) -> None:
        """Initialize SideEffectPolicyRule.

        Args:
            prefixes: Prefix tuple indicating action tools with side-effects.
            require_verification: If True, requires verified flag in execution kwargs or attribute.
        """
        self.prefixes = prefixes or self.DEFAULT_SIDE_EFFECT_PREFIXES
        self.require_verification = require_verification

    def evaluate(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """Evaluates whether an action tool with side-effects has been verified."""
        is_side_effect_tool = any(tool_name.lower().startswith(p) for p in self.prefixes) or kwargs.get(
            "is_side_effect", False
        )

        if is_side_effect_tool and self.require_verification:
            is_verified = kwargs.get("is_verified", False) or kwargs.get("is_approved", False)
            if not is_verified:
                return (
                    False,
                    f"Unverified side-effect blocked: Action tool '{tool_name}' requires explicit verification.",
                )

        return True, "Side-effect verification check passed."


class BlockedToolsPolicyRule(SandboxPolicyRule):
    """Rule that explicitly blocks specified tools or tool patterns."""

    def __init__(self, blocked_tools: Optional[List[str]] = None) -> None:
        """Initialize BlockedToolsPolicyRule.

        Args:
            blocked_tools: List of tool names that are explicitly blocked.
        """
        self.blocked_tools = set(blocked_tools or [])

    def evaluate(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """Evaluates whether tool_name is in the blocked set."""
        if tool_name in self.blocked_tools:
            return False, f"Tool '{tool_name}' is explicitly blocked by policy."
        return True, "Tool is not blocked."


class MCPPolicySandboxV4:
    """Runtime policy evaluator sandbox for MCP action tools."""

    def __init__(
        self,
        rules: Optional[List[SandboxPolicyRule]] = None,
        strict_side_effects: bool = True,
        blocked_tools: Optional[List[str]] = None,
        audit_trail: Optional[Any] = None,
    ) -> None:
        """Initialize MCPPolicySandboxV4.

        Args:
            rules: Custom list of SandboxPolicyRule instances.
            strict_side_effects: If True, adds SideEffectPolicyRule by default.
            blocked_tools: Optional list of tool names to block.
            audit_trail: Optional audit logger instance to log policy evaluations.
        """
        self.rules: List[SandboxPolicyRule] = list(rules or [])
        self.audit_trail = audit_trail

        if strict_side_effects and not any(isinstance(r, SideEffectPolicyRule) for r in self.rules):
            self.rules.append(SideEffectPolicyRule())

        if blocked_tools:
            self.rules.append(BlockedToolsPolicyRule(blocked_tools))

    def add_rule(self, rule: SandboxPolicyRule) -> None:
        """Registers a new sandbox policy rule.

        Args:
            rule: SandboxPolicyRule instance to add.
        """
        self.rules.append(rule)

    def check_policy(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """Evaluates all registered sandbox policy rules against a tool call.

        Args:
            tool_name: The name of the tool to evaluate.
            **kwargs: Arguments supplied for tool execution.

        Returns:
            Tuple of (allowed: bool, reason: str).
        """
        for rule in self.rules:
            allowed, reason = rule.evaluate(tool_name, **kwargs)
            if not allowed:
                return False, reason
        return True, "Allowed by sandbox policy."

    def execute(self, tool_func: Callable[..., Any], tool_name: str, **kwargs: Any) -> Any:
        """Executes a synchronous tool within the policy sandbox.

        Args:
            tool_func: The target synchronous tool function.
            tool_name: The name of the tool.
            **kwargs: Keyword arguments for the tool call.

        Returns:
            The return value of tool_func.

        Raises:
            MCPPolicySandboxViolationError: If policy evaluation fails.
            ValueError: If an async coroutine function is passed.
        """
        if inspect.iscoroutinefunction(tool_func):
            raise ValueError("execute() does not support coroutines; use execute_async() instead.")

        allowed, reason = self.check_policy(tool_name, **kwargs)
        self._log_audit(tool_name, allowed, reason, kwargs)

        if not allowed:
            logger.warning(f"MCPPolicySandboxV4 BLOCKED '{tool_name}': {reason}")
            raise MCPPolicySandboxViolationError(f"Action '{tool_name}' blocked by sandbox policy: {reason}")

        try:
            return tool_func(**kwargs)
        except Exception as e:
            logger.error(f"MCPPolicySandboxV4 tool execution error in '{tool_name}': {e}")
            raise

    async def execute_async(self, tool_func: Callable[..., Any], tool_name: str, **kwargs: Any) -> Any:
        """Executes an asynchronous tool within the policy sandbox.

        Args:
            tool_func: The target asynchronous tool coroutine function.
            tool_name: The name of the tool.
            **kwargs: Keyword arguments for the tool call.

        Returns:
            The return value of awaiting tool_func.

        Raises:
            MCPPolicySandboxViolationError: If policy evaluation fails.
            ValueError: If a synchronous function is passed.
        """
        if not inspect.iscoroutinefunction(tool_func):
            raise ValueError("execute_async() requires a coroutine function; use execute() instead.")

        allowed, reason = self.check_policy(tool_name, **kwargs)
        self._log_audit(tool_name, allowed, reason, kwargs)

        if not allowed:
            logger.warning(f"MCPPolicySandboxV4 BLOCKED async '{tool_name}': {reason}")
            raise MCPPolicySandboxViolationError(f"Action '{tool_name}' blocked by sandbox policy: {reason}")

        try:
            return await tool_func(**kwargs)
        except Exception as e:
            logger.error(f"MCPPolicySandboxV4 async tool execution error in '{tool_name}': {e}")
            raise

    def wrap_tool(self, tool_name: Optional[str] = None) -> Callable[..., Any]:
        """Decorator to wrap a tool function with sandbox policy evaluation.

        Args:
            tool_name: Optional custom tool name (defaults to function __name__).

        Returns:
            Decorated function enforcing sandbox policy.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            name = tool_name or func.__name__

            if inspect.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    return await self.execute_async(func, name, **bound.arguments)

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    return self.execute(func, name, **bound.arguments)

                return sync_wrapper

        return decorator

    def _log_audit(self, tool_name: str, allowed: bool, reason: str, kwargs: Dict[str, Any]) -> None:
        """Helper to log evaluation details to audit trail if provided."""
        if not self.audit_trail:
            return

        entry = {
            "tool_name": tool_name,
            "status": "allowed" if allowed else "blocked",
            "reason": reason,
            "args": kwargs,
        }

        if hasattr(self.audit_trail, "log_event"):
            self.audit_trail.log_event("mcp_policy_sandbox", entry)
        elif hasattr(self.audit_trail, "append"):
            self.audit_trail.append(entry)
