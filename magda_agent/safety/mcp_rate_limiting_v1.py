"""MCP Action Tool Rate Limiting V1.

Provides a strict token-bucket based rate limiting guardrail specifically tailored
for MCP action tools to prevent runaway external side-effects.
"""

import asyncio
import inspect
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

from magda_agent.safety.guardrails import SecurityViolationError

logger = logging.getLogger(__name__)


class MCPRateLimitViolationError(SecurityViolationError):
    """Raised when an MCP action tool call exceeds its rate limit."""

    pass


class TokenBucket:
    """Token bucket implementation for rate limiting."""

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        time_func: Callable[[], float] = time.time,
    ) -> None:
        """Initialize TokenBucket.

        Args:
            capacity: Maximum number of tokens the bucket can hold.
            refill_rate: Number of tokens added per second.
            time_func: Function to retrieve the current timestamp in seconds.
        """
        self.capacity: float = float(capacity)
        self.refill_rate: float = float(refill_rate)
        self.tokens: float = float(capacity)
        self.time_func: Callable[[], float] = time_func
        self.last_updated: float = self.time_func()

    def _refill(self, now: Optional[float] = None) -> None:
        """Refill tokens based on elapsed time."""
        current_time = now if now is not None else self.time_func()
        elapsed = max(0.0, current_time - self.last_updated)
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_updated = current_time

    def can_consume(self, tokens: float = 1.0, now: Optional[float] = None) -> bool:
        """Check if requested tokens can be consumed without mutating state."""
        current_time = now if now is not None else self.time_func()
        elapsed = max(0.0, current_time - self.last_updated)
        available = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        return available >= tokens

    def consume(self, tokens: float = 1.0, now: Optional[float] = None) -> Tuple[bool, float]:
        """Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume.
            now: Optional current timestamp.

        Returns:
            Tuple of (success: bool, wait_time_seconds: float).
        """
        current_time = now if now is not None else self.time_func()
        self._refill(current_time)

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0.0

        needed = tokens - self.tokens
        wait_time = needed / self.refill_rate if self.refill_rate > 0 else float("inf")
        return False, wait_time


class MCPActionToolRateLimiterV1:
    """Token-bucket based rate limiter for MCP action tools."""

    def __init__(
        self,
        default_capacity: float = 5.0,
        default_refill_rate: float = 1.0,
        tool_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        block_mode: bool = True,
        time_func: Callable[[], float] = time.time,
        audit_trail: Optional[Any] = None,
    ) -> None:
        """Initialize MCPActionToolRateLimiterV1.

        Args:
            default_capacity: Default maximum bucket capacity (max tokens).
            default_refill_rate: Default refill rate (tokens per second).
            tool_limits: Dict mapping tool_name -> (capacity, refill_rate).
            block_mode: If True, raises MCPRateLimitViolationError on rate limit breach.
                        If False, delays/sleeps until tokens are available.
            time_func: Callable returning current timestamp in seconds.
            audit_trail: Optional audit trail instance for logging rate limiting events.
        """
        self.default_capacity = float(default_capacity)
        self.default_refill_rate = float(default_refill_rate)
        self.tool_limits = tool_limits or {}
        self.block_mode = block_mode
        self.time_func = time_func
        self.audit_trail = audit_trail

        self._buckets: Dict[str, TokenBucket] = {}

    def _get_bucket(self, tool_name: str) -> TokenBucket:
        """Retrieve or initialize the TokenBucket for a specific tool."""
        if tool_name not in self._buckets:
            cap, rate = self.tool_limits.get(
                tool_name, (self.default_capacity, self.default_refill_rate)
            )
            self._buckets[tool_name] = TokenBucket(
                capacity=cap, refill_rate=rate, time_func=self.time_func
            )
        return self._buckets[tool_name]

    def set_tool_limit(self, tool_name: str, capacity: float, refill_rate: float) -> None:
        """Set or update rate limit for a specific tool."""
        self.tool_limits[tool_name] = (capacity, refill_rate)
        self._buckets[tool_name] = TokenBucket(
            capacity=capacity, refill_rate=refill_rate, time_func=self.time_func
        )

    def reset(self, tool_name: Optional[str] = None) -> None:
        """Reset rate limit bucket state for a specific tool or all tools."""
        if tool_name:
            if tool_name in self._buckets:
                cap, rate = self.tool_limits.get(
                    tool_name, (self.default_capacity, self.default_refill_rate)
                )
                self._buckets[tool_name] = TokenBucket(
                    capacity=cap, refill_rate=rate, time_func=self.time_func
                )
        else:
            self._buckets.clear()

    def check_rate_limit(self, tool_name: str, tokens: float = 1.0) -> Tuple[bool, float]:
        """Check rate limit without consuming tokens or sleeping.

        Returns:
            Tuple of (allowed: bool, wait_time_seconds: float).
        """
        bucket = self._get_bucket(tool_name)
        now = self.time_func()
        if bucket.can_consume(tokens, now=now):
            return True, 0.0
        needed = tokens - min(bucket.capacity, bucket.tokens + max(0.0, now - bucket.last_updated) * bucket.refill_rate)
        wait_time = needed / bucket.refill_rate if bucket.refill_rate > 0 else float("inf")
        return False, wait_time

    def acquire(self, tool_name: str, tokens: float = 1.0) -> None:
        """Synchronously acquire tokens or raise MCPRateLimitViolationError if blocked."""
        bucket = self._get_bucket(tool_name)
        now = self.time_func()
        success, wait_time = bucket.consume(tokens, now=now)

        if not success:
            reason = (
                f"Rate limit exceeded for MCP tool '{tool_name}'. "
                f"Requires {wait_time:.2f}s to recover tokens."
            )
            self._log_audit(tool_name, allowed=False, reason=reason, tokens=tokens)
            logger.warning(f"MCPActionToolRateLimiterV1 BLOCKED '{tool_name}': {reason}")
            raise MCPRateLimitViolationError(reason)

        self._log_audit(tool_name, allowed=True, reason="Rate limit check passed", tokens=tokens)

    async def acquire_async(self, tool_name: str, tokens: float = 1.0) -> None:
        """Asynchronously acquire tokens, sleeping if block_mode is False or raising if True."""
        while True:
            bucket = self._get_bucket(tool_name)
            now = self.time_func()
            success, wait_time = bucket.consume(tokens, now=now)

            if success:
                self._log_audit(tool_name, allowed=True, reason="Rate limit check passed", tokens=tokens)
                return

            reason = (
                f"Rate limit exceeded for MCP tool '{tool_name}'. "
                f"Requires {wait_time:.2f}s to recover tokens."
            )
            self._log_audit(tool_name, allowed=False, reason=reason, tokens=tokens)

            if self.block_mode:
                logger.warning(f"MCPActionToolRateLimiterV1 BLOCKED async '{tool_name}': {reason}")
                raise MCPRateLimitViolationError(reason)

            logger.info(f"MCPActionToolRateLimiterV1 delaying '{tool_name}' for {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

    def execute(self, tool_func: Callable[..., Any], tool_name: str, **kwargs: Any) -> Any:
        """Executes a synchronous tool function guarded by rate limiting."""
        if inspect.iscoroutinefunction(tool_func):
            raise ValueError("execute() does not support coroutines; use execute_async() instead.")

        self.acquire(tool_name)
        return tool_func(**kwargs)

    async def execute_async(self, tool_func: Callable[..., Any], tool_name: str, **kwargs: Any) -> Any:
        """Executes an asynchronous tool coroutine guarded by rate limiting."""
        if not inspect.iscoroutinefunction(tool_func):
            raise ValueError("execute_async() requires a coroutine function; use execute() instead.")

        await self.acquire_async(tool_name)
        return await tool_func(**kwargs)

    def wrap_tool(self, tool_name: Optional[str] = None) -> Callable[..., Any]:
        """Decorator to wrap a sync or async tool function with rate limiting."""
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

    def _log_audit(self, tool_name: str, allowed: bool, reason: str, tokens: float) -> None:
        """Log audit details to audit_trail if provided."""
        if not self.audit_trail:
            return

        entry = {
            "tool_name": tool_name,
            "status": "allowed" if allowed else "blocked",
            "reason": reason,
            "tokens": tokens,
        }

        if hasattr(self.audit_trail, "log_event"):
            self.audit_trail.log_event("mcp_rate_limiting", entry)
        elif hasattr(self.audit_trail, "append"):
            self.audit_trail.append(entry)


# Backward compatibility alias
MCPRateLimiterV1 = MCPActionToolRateLimiterV1
