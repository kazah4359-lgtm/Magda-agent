"""
MCP Dynamic Verification Sandbox Hook V2.

Provides a dynamic execution hook for intercepting, evaluating, and sandboxing
external MCP action tools at runtime prior to and after execution.
"""

import asyncio
import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from magda_agent.safety.guardrails import SecurityViolationError

logger = logging.getLogger(__name__)


class MCPDynamicSandboxViolationError(SecurityViolationError):
    """Raised when an MCP tool execution is blocked by the dynamic sandbox hook."""

    pass


class MCPDynamicSandboxHookV2:
    """
    Dynamic execution hook and sandbox manager for external MCP action tools.

    Provides pre-execution dynamic verification hooks, payload inspection for
    blocked hosts/tools, isolated synchronous and asynchronous tool execution,
    post-execution result hooks, and audit trail logging.
    """

    DEFAULT_BLOCKED_HOSTS: Set[str] = {
        "malicious.com",
        "untrusted.org",
        "phishing.net",
        "evil-domain.com",
    }

    DEFAULT_EXTERNAL_PREFIXES: Tuple[str, ...] = (
        "external_",
        "remote_",
        "http_",
        "api_",
        "web_",
        "fetch_",
    )

    def __init__(
        self,
        pre_hooks: Optional[List[Callable[..., Any]]] = None,
        post_hooks: Optional[List[Callable[..., Any]]] = None,
        blocked_tools: Optional[List[str]] = None,
        blocked_hosts: Optional[List[str]] = None,
        external_prefixes: Optional[Tuple[str, ...]] = None,
        strict_verification: bool = True,
        audit_trail: Optional[Any] = None,
    ) -> None:
        """
        Initialize MCPDynamicSandboxHookV2.

        Args:
            pre_hooks: List of callable pre-execution verification hooks.
            post_hooks: List of callable post-execution hooks.
            blocked_tools: Optional list of tool names explicitly blocked.
            blocked_hosts: Optional list of forbidden target domain hosts.
            external_prefixes: Tuple of tool name prefixes considered external action tools.
            strict_verification: If True, external action tools require explicit verification flag.
            audit_trail: Optional audit trail instance for event recording.
        """
        self.pre_hooks: List[Callable[..., Any]] = list(pre_hooks or [])
        self.post_hooks: List[Callable[..., Any]] = list(post_hooks or [])
        self.blocked_tools: Set[str] = set(blocked_tools or [])
        self.blocked_hosts: Set[str] = set(blocked_hosts or self.DEFAULT_BLOCKED_HOSTS)
        self.external_prefixes: Tuple[str, ...] = external_prefixes or self.DEFAULT_EXTERNAL_PREFIXES
        self.strict_verification: bool = strict_verification
        self.audit_trail = audit_trail

    def add_pre_hook(self, hook: Callable[..., Any]) -> None:
        """Register a dynamic pre-execution verification hook callback."""
        if hook not in self.pre_hooks:
            self.pre_hooks.append(hook)

    def remove_pre_hook(self, hook: Callable[..., Any]) -> None:
        """Remove a registered pre-execution hook callback."""
        if hook in self.pre_hooks:
            self.pre_hooks.remove(hook)

    def add_post_hook(self, hook: Callable[..., Any]) -> None:
        """Register a dynamic post-execution hook callback."""
        if hook not in self.post_hooks:
            self.post_hooks.append(hook)

    def remove_post_hook(self, hook: Callable[..., Any]) -> None:
        """Remove a registered post-execution hook callback."""
        if hook in self.post_hooks:
            self.post_hooks.remove(hook)

    def add_blocked_tool(self, tool_name: str) -> None:
        """Add a tool name to the blocked set."""
        self.blocked_tools.add(tool_name)

    def add_blocked_host(self, host: str) -> None:
        """Add a domain/host to the blocked targets set."""
        self.blocked_hosts.add(host.lower())

    def is_external_tool(self, tool_name: str, kwargs: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if a tool is an external action tool based on name prefix or kwargs."""
        if any(tool_name.lower().startswith(prefix) for prefix in self.external_prefixes):
            return True
        if kwargs and kwargs.get("is_external", False):
            return True
        return False

    def _extract_urls_and_hosts(self, value: Any) -> List[str]:
        """Recursively scan payload values for domain hosts or blocked target strings."""
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

    def verify_execution(self, tool_name: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Synchronously evaluates whether tool execution is permitted in sandbox.

        Args:
            tool_name: Name of target tool.
            payload: Keyword arguments supplied for tool execution.

        Returns:
            Tuple of (is_allowed: bool, reason: str).
        """
        # 1. Check explicit blocklist
        if tool_name in self.blocked_tools:
            return False, f"Tool '{tool_name}' is explicitly blocked by dynamic sandbox policy."

        # 2. Check for forbidden domain targets in payload
        extracted_hosts = self._extract_urls_and_hosts(payload)
        for host in extracted_hosts:
            if host in self.blocked_hosts:
                return False, f"Unsafe external target '{host}' detected in call payload for tool '{tool_name}'."

        # 3. Check strict verification requirement for external tools
        if self.is_external_tool(tool_name, payload) and self.strict_verification:
            is_verified = payload.get("is_verified", False) or payload.get("verified", False) or payload.get("is_approved", False)
            if not is_verified:
                return False, f"External tool '{tool_name}' requires explicit verification before execution."

        # 4. Evaluate registered synchronous pre-execution hooks
        for hook in self.pre_hooks:
            if inspect.iscoroutinefunction(hook):
                continue
            try:
                res = hook(tool_name, payload)
                if isinstance(res, tuple):
                    allowed, reason = res
                    if not allowed:
                        return False, f"Pre-execution hook check failed: {reason}"
                elif res is False:
                    return False, f"Pre-execution hook '{getattr(hook, '__name__', str(hook))}' denied execution."
            except Exception as e:
                logger.error(f"Error in pre-execution hook '{getattr(hook, '__name__', str(hook))}': {e}")
                return False, f"Pre-execution hook error: {str(e)}"

        return True, "Dynamic sandbox policy check passed."

    async def verify_execution_async(self, tool_name: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Asynchronously evaluates whether tool execution is permitted in sandbox.

        Args:
            tool_name: Name of target tool.
            payload: Keyword arguments supplied for tool execution.

        Returns:
            Tuple of (is_allowed: bool, reason: str).
        """
        # Run base sync checks first
        allowed, reason = self.verify_execution(tool_name, payload)
        if not allowed:
            return False, reason

        # Evaluate registered async pre-execution hooks
        for hook in self.pre_hooks:
            if not inspect.iscoroutinefunction(hook):
                continue
            try:
                res = await hook(tool_name, payload)
                if isinstance(res, tuple):
                    allowed, reason = res
                    if not allowed:
                        return False, f"Async pre-execution hook check failed: {reason}"
                elif res is False:
                    return False, f"Async pre-execution hook '{getattr(hook, '__name__', str(hook))}' denied execution."
            except Exception as e:
                logger.error(f"Error in async pre-execution hook '{getattr(hook, '__name__', str(hook))}': {e}")
                return False, f"Async pre-execution hook error: {str(e)}"

        return True, "Dynamic sandbox policy check passed."

    def execute(self, tool_func: Callable[..., Any], tool_name: str, **kwargs: Any) -> Any:
        """
        Executes a synchronous tool within the dynamic verification sandbox hook lifecycle.

        Args:
            tool_func: The target synchronous tool callable.
            tool_name: Name of the tool.
            **kwargs: Arguments to pass to the tool.

        Returns:
            Return value of tool_func.

        Raises:
            MCPDynamicSandboxViolationError: If sandbox verification fails.
            ValueError: If a coroutine function is passed.
        """
        if inspect.iscoroutinefunction(tool_func):
            raise ValueError("execute() does not support coroutines; use execute_async() instead.")

        allowed, reason = self.verify_execution(tool_name, kwargs)
        self._log_audit(tool_name, allowed, reason, kwargs)

        if not allowed:
            logger.warning(f"MCPDynamicSandboxHookV2 BLOCKED '{tool_name}': {reason}")
            raise MCPDynamicSandboxViolationError(reason)

        try:
            result = tool_func(**kwargs)
            self._run_post_hooks(tool_name, result, kwargs)
            return result
        except Exception as e:
            logger.error(f"MCPDynamicSandboxHookV2 execution error in '{tool_name}': {e}")
            raise

    async def execute_async(self, tool_func: Callable[..., Any], tool_name: str, **kwargs: Any) -> Any:
        """
        Executes an asynchronous tool within the dynamic verification sandbox hook lifecycle.

        Args:
            tool_func: The target asynchronous tool callable.
            tool_name: Name of the tool.
            **kwargs: Arguments to pass to the tool.

        Returns:
            Return value of awaiting tool_func.

        Raises:
            MCPDynamicSandboxViolationError: If sandbox verification fails.
            ValueError: If a synchronous function is passed.
        """
        if not inspect.iscoroutinefunction(tool_func):
            raise ValueError("execute_async() requires a coroutine function; use execute() instead.")

        allowed, reason = await self.verify_execution_async(tool_name, kwargs)
        self._log_audit(tool_name, allowed, reason, kwargs)

        if not allowed:
            logger.warning(f"MCPDynamicSandboxHookV2 BLOCKED async '{tool_name}': {reason}")
            raise MCPDynamicSandboxViolationError(reason)

        try:
            result = await tool_func(**kwargs)
            await self._run_post_hooks_async(tool_name, result, kwargs)
            return result
        except Exception as e:
            logger.error(f"MCPDynamicSandboxHookV2 async execution error in '{tool_name}': {e}")
            raise

    def wrap_tool(self, tool_name: Optional[str] = None) -> Callable[..., Any]:
        """
        Decorator to wrap a sync or async tool function with the dynamic verification sandbox hook.

        Args:
            tool_name: Optional custom tool name (defaults to func.__name__).

        Returns:
            Decorated callable enforcing sandboxed execution lifecycle.
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

    def _run_post_hooks(self, tool_name: str, result: Any, kwargs: Dict[str, Any]) -> None:
        """Helper to invoke synchronous post-execution hooks."""
        for hook in self.post_hooks:
            if inspect.iscoroutinefunction(hook):
                continue
            try:
                hook(tool_name, result, kwargs)
            except Exception as e:
                logger.error(f"Error in post-execution hook '{getattr(hook, '__name__', str(hook))}': {e}")

    async def _run_post_hooks_async(self, tool_name: str, result: Any, kwargs: Dict[str, Any]) -> None:
        """Helper to invoke synchronous and asynchronous post-execution hooks."""
        for hook in self.post_hooks:
            try:
                if inspect.iscoroutinefunction(hook):
                    await hook(tool_name, result, kwargs)
                else:
                    hook(tool_name, result, kwargs)
            except Exception as e:
                logger.error(f"Error in post-execution hook '{getattr(hook, '__name__', str(hook))}': {e}")

    def _log_audit(self, tool_name: str, allowed: bool, reason: str, kwargs: Dict[str, Any]) -> None:
        """Helper to record audit log events if audit_trail instance is present."""
        if not self.audit_trail:
            return

        entry = {
            "tool_name": tool_name,
            "status": "allowed" if allowed else "blocked",
            "reason": reason,
            "args": kwargs,
        }

        if hasattr(self.audit_trail, "log_event"):
            self.audit_trail.log_event("mcp_sandbox_hook_v2", entry)
        elif hasattr(self.audit_trail, "append"):
            self.audit_trail.append(entry)


# Alias for backward compatibility
MCPSandboxHookV2 = MCPDynamicSandboxHookV2
