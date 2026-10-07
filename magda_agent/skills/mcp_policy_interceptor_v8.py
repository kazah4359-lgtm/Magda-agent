"""
MCP Action Tool Policy Interceptor V8.

Provides a centralized runtime policy interceptor tailored for dynamic MCP action tools
to validate payloads, inspect taint indicators, check blocked tools/hosts/patterns,
and enforce safety rules before execution.
"""

import asyncio
import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

try:
    from magda_agent.safety.guardrails import SecurityViolationError
except ImportError:
    class SecurityViolationError(Exception):
        """Fallback SecurityViolationError if guardrails module is unavailable."""
        pass

try:
    from magda_agent.safety.taint import is_tainted as check_is_tainted, TaintedData
except ImportError:
    def check_is_tainted(obj: Any) -> bool:
        return False
    class TaintedData:
        pass


logger = logging.getLogger(__name__)


class MCPPolicyViolationErrorV8(SecurityViolationError):
    """Raised when an MCP action tool call payload violates policy or contains tainted data."""

    pass


class MCPActionToolPolicyInterceptorV8:
    """
    Centralized runtime policy interceptor for dynamic MCP tools.
    Evaluates payloads for taint, blocked tools, forbidden hosts, and dangerous patterns.
    """

    DEFAULT_BLOCKED_HOSTS: Set[str] = {
        "malicious.com",
        "untrusted.org",
        "phishing.net",
        "evil-domain.com",
        "badhost.local",
    }

    DEFAULT_UNSAFE_PATTERNS: Set[str] = {
        "rm -rf",
        "drop table",
        ":(){ :|:& };:",
        "curl -s http",
        "wget -q http",
        "eval(",
        "exec(",
        "<script>",
    }

    DEFAULT_SENSITIVE_ARGUMENTS: Set[str] = {
        "command",
        "script",
        "payload",
        "file_path",
        "url",
        "code",
        "query",
    }

    def __init__(
        self,
        blocked_tools: Optional[List[str]] = None,
        blocked_hosts: Optional[List[str]] = None,
        blocked_patterns: Optional[List[str]] = None,
        sensitive_arguments: Optional[List[str]] = None,
        strict_taint_check: bool = True,
        audit_trail: Optional[Any] = None,
    ) -> None:
        """
        Initializes MCPActionToolPolicyInterceptorV8.

        Args:
            blocked_tools: List of tool names that are explicitly forbidden.
            blocked_hosts: List of domain names/hosts forbidden in payload URLs.
            blocked_patterns: List of string patterns forbidden in execution arguments.
            sensitive_arguments: Argument names that trigger taint enforcement.
            strict_taint_check: If True, rejects any payload flagged as tainted.
            audit_trail: Optional audit logger instance for tracking policy checks.
        """
        self.blocked_tools: Set[str] = set(blocked_tools or [])
        self.blocked_hosts: Set[str] = set(
            host.lower() for host in (blocked_hosts or self.DEFAULT_BLOCKED_HOSTS)
        )
        self.blocked_patterns: Set[str] = set(
            p.lower() for p in (blocked_patterns or self.DEFAULT_UNSAFE_PATTERNS)
        )
        self.sensitive_arguments: Set[str] = set(
            sensitive_arguments or self.DEFAULT_SENSITIVE_ARGUMENTS
        )
        self.strict_taint_check: bool = strict_taint_check
        self.audit_trail: Optional[Any] = audit_trail

    def add_blocked_tool(self, tool_name: str) -> None:
        """Add a tool name to the explicit blocklist."""
        self.blocked_tools.add(tool_name)

    def add_blocked_host(self, host: str) -> None:
        """Add a domain/host to the blocked target list."""
        self.blocked_hosts.add(host.lower())

    def add_blocked_pattern(self, pattern: str) -> None:
        """Add a string pattern to the dangerous command/pattern set."""
        self.blocked_patterns.add(pattern.lower())

    def _extract_hosts_from_value(self, value: Any) -> List[str]:
        """Recursively extracts hostnames from string/dict/list values in payload."""
        hosts: List[str] = []
        if isinstance(value, str):
            if "://" in value:
                try:
                    parsed = urlparse(value)
                    if parsed.netloc:
                        hosts.append(parsed.netloc.split(":")[0].lower())
                except Exception:
                    pass
            val_lower = value.lower()
            for host in self.blocked_hosts:
                if host in val_lower:
                    hosts.append(host)
        elif isinstance(value, dict):
            for v in value.values():
                hosts.extend(self._extract_hosts_from_value(v))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                hosts.extend(self._extract_hosts_from_value(item))
        return hosts

    def _find_unsafe_pattern(self, value: Any) -> Optional[str]:
        """Recursively inspects payload values for forbidden command patterns."""
        if isinstance(value, str):
            val_lower = value.lower()
            for pattern in self.blocked_patterns:
                if pattern in val_lower:
                    return pattern
        elif isinstance(value, dict):
            for v in value.values():
                pattern = self._find_unsafe_pattern(v)
                if pattern:
                    return pattern
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                pattern = self._find_unsafe_pattern(item)
                if pattern:
                    return pattern
        return None

    def _inspect_taint(self, value: Any, parent_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Recursively inspects a value for taint indicators or sensitive argument violations.

        Returns:
            Tuple of (is_tainted: bool, reason: str).
        """
        if check_is_tainted(value) or isinstance(value, TaintedData) or getattr(value, "is_tainted", False):
            return True, f"Tainted object detected in payload (key: '{parent_key or 'root'}')."

        if isinstance(value, dict):
            if value.get("__tainted__") or value.get("is_tainted"):
                return True, f"Explicit taint flag found in dictionary (key: '{parent_key or 'root'}')."
            for k, v in value.items():
                is_t, reason = self._inspect_taint(v, parent_key=str(k))
                if is_t:
                    return True, reason
                # Also check if sensitive argument key holds a value marked as tainted
                if str(k).lower() in self.sensitive_arguments:
                    if check_is_tainted(v) or getattr(v, "is_tainted", False) or (isinstance(v, str) and "tainted" in str(v).lower()):
                        return True, f"Sensitive argument '{k}' contains tainted content."

        elif isinstance(value, (list, tuple, set)):
            for idx, item in enumerate(value):
                is_t, reason = self._inspect_taint(item, parent_key=f"{parent_key}[{idx}]" if parent_key else f"[{idx}]")
                if is_t:
                    return True, reason

        elif isinstance(value, str):
            if "tainted_input" in value.lower() or "tainted_payload" in value.lower():
                return True, f"Tainted keyword found in string argument (key: '{parent_key or 'root'}')."

        return False, ""

    def intercept_execution(self, tool_name: str, payload: Dict[str, Any]) -> bool:
        """
        Interprets and validates an MCP action tool execution request.

        Args:
            tool_name: The name of the MCP tool to execute.
            payload: Arguments/kwargs passed to the tool.

        Returns:
            True if execution is permitted.

        Raises:
            MCPPolicyViolationErrorV8: If execution violates policy or payload is tainted.
        """
        # 1. Check explicit tool blocklist
        if tool_name in self.blocked_tools:
            reason = f"Tool '{tool_name}' is explicitly forbidden by blocklist policy."
            self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
            raise MCPPolicyViolationErrorV8(reason)

        # 2. Check for tainted payload content
        if self.strict_taint_check:
            is_tainted_flag, taint_reason = self._inspect_taint(payload)
            if is_tainted_flag:
                reason = f"Execution blocked for '{tool_name}': {taint_reason}"
                self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
                raise MCPPolicyViolationErrorV8(reason)

        # 3. Check for blocked external hosts/targets
        extracted_hosts = self._extract_hosts_from_value(payload)
        for host in extracted_hosts:
            if host in self.blocked_hosts:
                reason = f"Execution blocked for '{tool_name}': Payload targets forbidden host '{host}'."
                self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
                raise MCPPolicyViolationErrorV8(reason)

        # 4. Check for dangerous command/execution patterns
        unsafe_pattern = self._find_unsafe_pattern(payload)
        if unsafe_pattern:
            reason = f"Execution blocked for '{tool_name}': Dangerous pattern '{unsafe_pattern}' detected in payload."
            self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
            raise MCPPolicyViolationErrorV8(reason)

        self._log_audit(tool_name, allowed=True, reason="Validated by MCP Policy Interceptor V8.", payload=payload)
        return True

    def execute(self, tool_func: Callable[..., Any], tool_name: str, **kwargs: Any) -> Any:
        """
        Executes a synchronous tool within the policy interceptor wrapper.

        Args:
            tool_func: The synchronous target function.
            tool_name: Tool identifier name.
            **kwargs: Arguments to pass to tool_func.

        Returns:
            The return value of tool_func(**kwargs).
        """
        if inspect.iscoroutinefunction(tool_func):
            raise ValueError("execute() cannot execute async coroutine functions; use execute_async() instead.")

        self.intercept_execution(tool_name, kwargs)
        return tool_func(**kwargs)

    async def execute_async(self, tool_func: Callable[..., Any], tool_name: str, **kwargs: Any) -> Any:
        """
        Executes an asynchronous tool within the policy interceptor wrapper.

        Args:
            tool_func: The asynchronous coroutine target function.
            tool_name: Tool identifier name.
            **kwargs: Arguments to pass to tool_func.

        Returns:
            The awaited return value of tool_func(**kwargs).
        """
        if not inspect.iscoroutinefunction(tool_func):
            raise ValueError("execute_async() requires a coroutine function; use execute() instead.")

        self.intercept_execution(tool_name, kwargs)
        return await tool_func(**kwargs)

    def wrap_tool(self, tool_name: Optional[str] = None) -> Callable[..., Any]:
        """
        Decorator that wraps a tool function with MCP policy interceptor checks.

        Args:
            tool_name: Custom tool name override (defaults to func.__name__).

        Returns:
            Decorated sync or async tool function.
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

    def _log_audit(self, tool_name: str, allowed: bool, reason: str, payload: Dict[str, Any]) -> None:
        """Helper to record policy evaluation event in audit_trail if configured."""
        if not self.audit_trail:
            return

        entry = {
            "tool_name": tool_name,
            "status": "allowed" if allowed else "blocked",
            "reason": reason,
            "payload": payload,
        }

        if hasattr(self.audit_trail, "log_event"):
            self.audit_trail.log_event("mcp_policy_interceptor_v8", entry)
        elif hasattr(self.audit_trail, "log_call"):
            self.audit_trail.log_call(tool_name, payload, reason, "allowed" if allowed else "blocked", 0.0)
        elif hasattr(self.audit_trail, "append"):
            self.audit_trail.append(entry)


# Alias for backward compatibility and concise import
MCPPolicyInterceptorV8 = MCPActionToolPolicyInterceptorV8
