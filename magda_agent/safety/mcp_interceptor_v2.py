"""
MCP Dynamic Tool Registry Auth Interceptor V2.

Provides a runtime interceptor for the MCP dynamic tool registry to verify
auth bindings, roles, and block unsafe external calls or side effects.
"""

import asyncio
import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from magda_agent.safety.guardrails import SecurityViolationError

logger = logging.getLogger(__name__)


class MCPAuthInterceptorError(SecurityViolationError):
    """Raised when an MCP action tool call is blocked by the auth interceptor."""

    pass


class MCPToolRegistryAuthInterceptorV2:
    """
    Runtime interceptor for MCP dynamic tool registry that evaluates schema
    registrations and intercepts tool execution calls to block unauthorized or
    unsafe external calls.
    """

    DEFAULT_BLOCKED_HOSTS: Set[str] = {
        "malicious.com",
        "untrusted.org",
        "phishing.net",
        "evil-domain.com",
    }

    DEFAULT_UNSAFE_COMMAND_PATTERNS: Set[str] = {
        "rm -rf",
        "drop table",
        ":(){ :|:& };:",
        "curl -s http",
        "wget -q http",
        "eval(",
        "exec(",
    }

    def __init__(
        self,
        required_tokens: Optional[Dict[str, str]] = None,
        required_roles: Optional[Dict[str, str]] = None,
        blocked_tools: Optional[List[str]] = None,
        blocked_hosts: Optional[List[str]] = None,
        sensitive_prefixes: Optional[List[str]] = None,
        audit_trail: Optional[Any] = None,
    ) -> None:
        """
        Initializes the MCPToolRegistryAuthInterceptorV2.

        Args:
            required_tokens: Dict mapping tool_name (or prefix) to expected auth token binding.
            required_roles: Dict mapping tool_name (or prefix) to minimum required user role.
            blocked_tools: List of tool names that are explicitly forbidden.
            blocked_hosts: List of domain names / hosts forbidden in external call payloads.
            sensitive_prefixes: List of tool name prefixes requiring authentication.
            audit_trail: Optional logger or audit trail instance for tracking calls.
        """
        self.required_tokens: Dict[str, str] = dict(required_tokens or {})
        self.required_roles: Dict[str, str] = dict(required_roles or {})
        self.blocked_tools: Set[str] = set(blocked_tools or [])
        self.blocked_hosts: Set[str] = set(blocked_hosts or self.DEFAULT_BLOCKED_HOSTS)
        if sensitive_prefixes is not None:
            self.sensitive_prefixes: List[str] = list(sensitive_prefixes)
        else:
            self.sensitive_prefixes = ["write_", "delete_", "execute_", "external_", "remote_", "http_"]
        self.audit_trail = audit_trail

    def add_required_token(self, tool_name_or_prefix: str, token: str) -> None:
        """Register a required token binding for a tool or prefix."""
        self.required_tokens[tool_name_or_prefix] = token

    def add_required_role(self, tool_name_or_prefix: str, role: str) -> None:
        """Register a required role for a tool or prefix."""
        self.required_roles[tool_name_or_prefix] = role

    def add_blocked_tool(self, tool_name: str) -> None:
        """Add a tool name to the explicit block list."""
        self.blocked_tools.add(tool_name)

    def add_blocked_host(self, host: str) -> None:
        """Add a domain / host to the blocked external target list."""
        self.blocked_hosts.add(host.lower())

    def validate_schema_interceptor(self, tool_schema: Dict[str, Any]) -> None:
        """
        Interceptor callback for MCPRegistry.add_interceptor().

        Validates schema for blocked tools or disallowed external endpoints.

        Args:
            tool_schema: The MCP tool schema dictionary.

        Raises:
            MCPAuthInterceptorError: If the tool schema violates safety policies.
        """
        if not isinstance(tool_schema, dict):
            return

        name = tool_schema.get("name", "")
        if name in self.blocked_tools:
            raise MCPAuthInterceptorError(f"Registration blocked: Tool '{name}' is in the blocked tools list.")

        # Inspect schema metadata or description for blocked hosts
        desc = str(tool_schema.get("description", "")).lower()
        endpoint = str(tool_schema.get("endpoint", "")).lower()
        for host in self.blocked_hosts:
            if host in desc or host in endpoint:
                raise MCPAuthInterceptorError(
                    f"Registration blocked: Tool '{name}' references forbidden external host '{host}'."
                )

    def is_sensitive(self, tool_name: str) -> bool:
        """Determines if a tool requires auth evaluation based on name prefix."""
        return any(tool_name.lower().startswith(p) for p in self.sensitive_prefixes)

    def _extract_urls_and_hosts(self, value: Any) -> List[str]:
        """Recursively extracts domain hosts from dicts/lists/strings in payload."""
        hosts: List[str] = []
        if isinstance(value, str):
            if "://" in value:
                try:
                    parsed = urlparse(value)
                    if parsed.netloc:
                        hosts.append(parsed.netloc.split(":")[0].lower())
                except Exception:
                    pass
            for host in self.blocked_hosts:
                if host in value.lower():
                    hosts.append(host)
        elif isinstance(value, dict):
            for k, v in value.items():
                hosts.extend(self._extract_urls_and_hosts(v))
        elif isinstance(value, (list, tuple)):
            for item in value:
                hosts.extend(self._extract_urls_and_hosts(item))
        return hosts

    def _check_unsafe_patterns(self, value: Any) -> Optional[str]:
        """Recursively checks for dangerous command patterns in execution arguments."""
        if isinstance(value, str):
            val_lower = value.lower()
            for pattern in self.DEFAULT_UNSAFE_COMMAND_PATTERNS:
                if pattern in val_lower:
                    return pattern
        elif isinstance(value, dict):
            for k, v in value.items():
                pattern = self._check_unsafe_patterns(v)
                if pattern:
                    return pattern
        elif isinstance(value, (list, tuple)):
            for item in value:
                pattern = self._check_unsafe_patterns(item)
                if pattern:
                    return pattern
        return None

    def intercept_execution(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        auth_token: Optional[str] = None,
        user_role: Optional[str] = None,
    ) -> bool:
        """
        Evaluates whether an MCP action tool execution call is permitted.

        Args:
            tool_name: Name of the target MCP tool.
            payload: Tool execution arguments.
            auth_token: Optional authentication token provided in call context.
            user_role: Optional user role provided in call context.

        Returns:
            True if execution is allowed.

        Raises:
            MCPAuthInterceptorError: If execution violates security/auth rules.
        """
        # 1. Check explicit blocklist
        if tool_name in self.blocked_tools:
            reason = f"Execution blocked: Tool '{tool_name}' is explicitly forbidden."
            self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
            raise MCPAuthInterceptorError(reason)

        # 2. Check payload for blocked external calls / forbidden hosts
        extracted_hosts = self._extract_urls_and_hosts(payload)
        for host in extracted_hosts:
            if host in self.blocked_hosts:
                reason = f"Execution blocked: Unsafe external call target '{host}' detected in payload."
                self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
                raise MCPAuthInterceptorError(reason)

        # 3. Check payload for unsafe execution commands
        unsafe_pattern = self._check_unsafe_patterns(payload)
        if unsafe_pattern:
            reason = f"Execution blocked: Unsafe command pattern '{unsafe_pattern}' detected in payload."
            self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
            raise MCPAuthInterceptorError(reason)

        # 4. Check token requirements
        required_token = self.required_tokens.get(tool_name)
        if not required_token:
            # Check prefix match if exact match not found
            for prefix, token in self.required_tokens.items():
                if tool_name.startswith(prefix):
                    required_token = token
                    break

        if required_token or self.is_sensitive(tool_name):
            expected_token = required_token
            if expected_token:
                if not auth_token:
                    reason = f"Execution blocked: Tool '{tool_name}' requires an auth token, but none was provided."
                    self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
                    raise MCPAuthInterceptorError(reason)

                if auth_token != expected_token and not auth_token.startswith(f"{expected_token}:"):
                    reason = f"Execution blocked: Invalid auth token provided for tool '{tool_name}'."
                    self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
                    raise MCPAuthInterceptorError(reason)

        # 5. Check role requirements
        required_role = self.required_roles.get(tool_name)
        if not required_role:
            for prefix, role in self.required_roles.items():
                if tool_name.startswith(prefix):
                    required_role = role
                    break

        if required_role:
            if not user_role:
                reason = f"Execution blocked: Tool '{tool_name}' requires user role '{required_role}', but none provided."
                self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
                raise MCPAuthInterceptorError(reason)

            # Basic role check: admin allows all; otherwise exact match
            if user_role.lower() != "admin" and user_role.lower() != required_role.lower():
                reason = f"Execution blocked: User role '{user_role}' insufficient for required role '{required_role}' on tool '{tool_name}'."
                self._log_audit(tool_name, allowed=False, reason=reason, payload=payload)
                raise MCPAuthInterceptorError(reason)

        self._log_audit(tool_name, allowed=True, reason="Allowed by interceptor.", payload=payload)
        return True

    def wrap_tool(
        self,
        tool_name: Optional[str] = None,
        required_token: Optional[str] = None,
        required_role: Optional[str] = None,
    ) -> Callable[..., Any]:
        """
        Decorator that wraps a tool function with auth interceptor checks.

        Args:
            tool_name: Custom tool name (defaults to func.__name__).
            required_token: Optional explicit required token for this wrapped tool.
            required_role: Optional explicit required role for this wrapped tool.

        Returns:
            Decorated sync or async callable.
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            name = tool_name or func.__name__
            if required_token:
                self.add_required_token(name, required_token)
            if required_role:
                self.add_required_role(name, required_role)

            sig = inspect.signature(func)
            accepts_auth_token = "auth_token" in sig.parameters
            accepts_user_role = "user_role" in sig.parameters

            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    func_kwargs = dict(kwargs)
                    auth_token = func_kwargs.pop("auth_token", None) if not accepts_auth_token else func_kwargs.get("auth_token")
                    user_role = func_kwargs.pop("user_role", None) if not accepts_user_role else func_kwargs.get("user_role")

                    bound = sig.bind(*args, **func_kwargs)
                    bound.apply_defaults()
                    payload = dict(bound.arguments)

                    self.intercept_execution(name, payload, auth_token=auth_token, user_role=user_role)
                    return await func(*args, **func_kwargs)

                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    func_kwargs = dict(kwargs)
                    auth_token = func_kwargs.pop("auth_token", None) if not accepts_auth_token else func_kwargs.get("auth_token")
                    user_role = func_kwargs.pop("user_role", None) if not accepts_user_role else func_kwargs.get("user_role")

                    bound = sig.bind(*args, **func_kwargs)
                    bound.apply_defaults()
                    payload = dict(bound.arguments)

                    self.intercept_execution(name, payload, auth_token=auth_token, user_role=user_role)
                    return func(*args, **func_kwargs)

                return sync_wrapper

        return decorator

    def _log_audit(self, tool_name: str, allowed: bool, reason: str, payload: Dict[str, Any]) -> None:
        """Helper to record audit details if audit_trail instance is provided."""
        if not self.audit_trail:
            return

        entry = {
            "tool_name": tool_name,
            "status": "allowed" if allowed else "blocked",
            "reason": reason,
            "payload": payload,
        }

        if hasattr(self.audit_trail, "log_event"):
            self.audit_trail.log_event("mcp_auth_interceptor_v2", entry)
        elif hasattr(self.audit_trail, "log_call"):
            self.audit_trail.log_call(tool_name, payload, reason, "allowed" if allowed else "blocked", 0.0)
        elif hasattr(self.audit_trail, "append"):
            self.audit_trail.append(entry)


# Convenience alias for backwards compatibility
MCPAuthInterceptorV2 = MCPToolRegistryAuthInterceptorV2
