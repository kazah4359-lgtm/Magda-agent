"""Tests for MCP Action Tool Rate Limiting V1."""

import asyncio
import pytest
from typing import Any, List, Dict

from magda_agent.safety.mcp_rate_limiting_v1 import (
    MCPActionToolRateLimiterV1,
    MCPRateLimiterV1,
    MCPRateLimitViolationError,
    TokenBucket,
)
from magda_agent.safety.guardrails import SecurityViolationError


class MockTime:
    """Helper class to simulate deterministic time progression in tests."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.now = start_time

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class MockAuditTrail:
    """Mock audit trail for capturing rate limit logs."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        self.events.append({"event_type": event_type, **data})


def test_token_bucket_refill_and_capacity() -> None:
    mock_time = MockTime(start_time=0.0)
    # Capacity 2, 1 token per second
    bucket = TokenBucket(capacity=2.0, refill_rate=1.0, time_func=mock_time.time)

    # Initial state should be full
    assert bucket.tokens == 2.0

    # Consume 2 tokens
    success, wait = bucket.consume(2.0)
    assert success is True
    assert bucket.tokens == 0.0

    # Consume 1 more token should fail
    success, wait = bucket.consume(1.0)
    assert success is False
    assert pytest.approx(wait, 0.01) == 1.0

    # Advance time by 0.5s -> should have 0.5 tokens
    mock_time.advance(0.5)
    assert bucket.can_consume(1.0) is False

    # Advance time by another 0.5s (total 1.0s) -> should have 1 token
    mock_time.advance(0.5)
    assert bucket.can_consume(1.0) is True
    success, wait = bucket.consume(1.0)
    assert success is True
    assert bucket.tokens == 0.0

    # Advance 5 seconds -> tokens capped at max capacity 2.0
    mock_time.advance(5.0)
    assert bucket.can_consume(2.0) is True
    bucket.consume(1.0)
    assert bucket.tokens == 1.0


def test_mcp_rate_limiter_blocking_excessive_requests() -> None:
    mock_time = MockTime()
    audit = MockAuditTrail()
    limiter = MCPActionToolRateLimiterV1(
        default_capacity=2.0,
        default_refill_rate=0.5,
        time_func=mock_time.time,
        audit_trail=audit,
    )

    # First two requests allowed
    limiter.acquire("write_file")
    limiter.acquire("write_file")

    # Third request should raise MCPRateLimitViolationError
    with pytest.raises(MCPRateLimitViolationError) as exc_info:
        limiter.acquire("write_file")

    assert "Rate limit exceeded for MCP tool 'write_file'" in str(exc_info.value)
    assert issubclass(MCPRateLimitViolationError, SecurityViolationError)

    # Check audit trail logs
    assert len(audit.events) == 3
    assert audit.events[0]["status"] == "allowed"
    assert audit.events[1]["status"] == "allowed"
    assert audit.events[2]["status"] == "blocked"


def test_mcp_rate_limiter_custom_tool_limits() -> None:
    mock_time = MockTime()
    limiter = MCPActionToolRateLimiterV1(
        default_capacity=1.0,
        default_refill_rate=1.0,
        tool_limits={"high_freq_tool": (10.0, 5.0)},
        time_func=mock_time.time,
    )

    # Standard tool blocked after 1 call
    limiter.acquire("standard_tool")
    with pytest.raises(MCPRateLimitViolationError):
        limiter.acquire("standard_tool")

    # Custom tool allows 10 calls
    for _ in range(10):
        limiter.acquire("high_freq_tool")

    with pytest.raises(MCPRateLimitViolationError):
        limiter.acquire("high_freq_tool")


def test_sync_execute_and_wrap_tool() -> None:
    mock_time = MockTime()
    limiter = MCPActionToolRateLimiterV1(
        default_capacity=1.0,
        default_refill_rate=1.0,
        time_func=mock_time.time,
    )

    def sample_sync_func(a: int, b: int) -> int:
        return a + b

    # Execute directly
    res = limiter.execute(sample_sync_func, "add_tool", a=2, b=3)
    assert res == 5

    # Second call without time advancement fails
    with pytest.raises(MCPRateLimitViolationError):
        limiter.execute(sample_sync_func, "add_tool", a=2, b=3)

    # Advance time to replenish token
    mock_time.advance(1.0)

    # Test wrap_tool decorator
    wrapped = limiter.wrap_tool(tool_name="add_tool")(sample_sync_func)
    assert wrapped(3, 4) == 7

    # Exceeded again
    with pytest.raises(MCPRateLimitViolationError):
        wrapped(1, 1)


@pytest.mark.asyncio
async def test_async_execute_and_wrap_tool() -> None:
    mock_time = MockTime()
    limiter = MCPActionToolRateLimiterV1(
        default_capacity=1.0,
        default_refill_rate=1.0,
        block_mode=True,
        time_func=mock_time.time,
    )

    async def sample_async_func(text: str) -> str:
        return text.upper()

    res = await limiter.execute_async(sample_async_func, "upper_tool", text="hello")
    assert res == "HELLO"

    with pytest.raises(MCPRateLimitViolationError):
        await limiter.execute_async(sample_async_func, "upper_tool", text="world")

    mock_time.advance(1.0)

    wrapped = limiter.wrap_tool("upper_tool")(sample_async_func)
    res_wrapped = await wrapped(text="test")
    assert res_wrapped == "TEST"


def test_reset_and_check_rate_limit() -> None:
    mock_time = MockTime()
    limiter = MCPRateLimiterV1(
        default_capacity=2.0,
        default_refill_rate=1.0,
        time_func=mock_time.time,
    )

    allowed, wait = limiter.check_rate_limit("tool_a")
    assert allowed is True
    assert wait == 0.0

    limiter.acquire("tool_a")
    limiter.acquire("tool_a")

    allowed, wait = limiter.check_rate_limit("tool_a")
    assert allowed is False
    assert wait > 0.0

    # Reset single tool
    limiter.reset("tool_a")
    allowed, wait = limiter.check_rate_limit("tool_a")
    assert allowed is True

    # Reset all
    limiter.acquire("tool_a")
    limiter.reset()
    allowed, wait = limiter.check_rate_limit("tool_a")
    assert allowed is True


def test_execute_type_validation_errors() -> None:
    limiter = MCPActionToolRateLimiterV1()

    def sync_fn() -> str:
        return "sync"

    async def async_fn() -> str:
        return "async"

    # Passing coroutine to execute should fail
    with pytest.raises(ValueError, match="execute\\(\\) does not support coroutines"):
        limiter.execute(async_fn, "tool")

    # Passing sync function to execute_async should fail
    with pytest.raises(ValueError, match="execute_async\\(\\) requires a coroutine function"):
        asyncio.run(limiter.execute_async(sync_fn, "tool"))
