import asyncio
import pytest
from typing import Any, Dict, List
from magda_agent.channels.base import ChannelAdapter
from magda_agent.channels.hub import ChannelHub
from magda_agent.channels.rate_limiter import ChannelRateLimiter, RateLimitExceededError
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class MockClock:
    """Mock clock for testing rate limiter without real-time delay."""

    def __init__(self, start_time: float = 1000.0) -> None:
        """Initialize mock clock with a starting time."""
        self.current_time: float = start_time

    def time(self) -> float:
        """Return the current mock time."""
        return self.current_time

    def advance(self, duration: float) -> None:
        """Advance the mock time."""
        self.current_time += duration

    async def sleep(self, duration: float) -> None:
        """Simulate sleep by advancing mock time."""
        self.advance(duration)


class DummyChannel(ChannelAdapter):
    """A dummy channel adapter to verify hub sending."""

    def __init__(self, channel_id: str, gateway: GatewayRouter) -> None:
        """Initialize the dummy channel."""
        super().__init__(channel_id, gateway)
        self.sent_messages: List[Dict[str, Any]] = []

    async def receive(self, raw_data: Any) -> Any:
        """Mock receive method."""
        return None

    async def send(self, recipient_id: str, text: str, metadata: Dict[str, Any] = None) -> str:
        """Record and return sent message confirmation."""
        self.sent_messages.append({"recipient": recipient_id, "text": text, "metadata": metadata})
        return f"Sent to {recipient_id}: {text}"


@pytest.fixture
def gateway() -> GatewayRouter:
    """Fixture providing a GatewayRouter."""
    return GatewayRouter()


@pytest.mark.asyncio
async def test_rate_limiter_block_strategy() -> None:
    """Verify that the block strategy raises RateLimitExceededError immediately on limit breach."""
    clock = MockClock()
    # 2 requests per 5 seconds, strategy block
    limiter = ChannelRateLimiter(
        limits={"telegram": {"max_requests": 2, "period": 5.0, "strategy": "block"}},
        time_func=clock.time,
        sleep_func=clock.sleep
    )

    # First request
    await limiter.acquire("telegram")
    # Second request
    await limiter.acquire("telegram")

    # Third request should fail immediately
    with pytest.raises(RateLimitExceededError) as exc_info:
        await limiter.acquire("telegram")
    assert "Rate limit exceeded on channel 'telegram'" in str(exc_info.value)

    # If we advance the clock by 5.1 seconds, the sliding window clears and we can send again
    clock.advance(5.1)
    await limiter.acquire("telegram")


@pytest.mark.asyncio
async def test_rate_limiter_delay_strategy() -> None:
    """Verify that the delay strategy correctly delays requests until a slot is free."""
    clock = MockClock()
    # 2 requests per 10 seconds, strategy delay
    limiter = ChannelRateLimiter(
        limits={"discord": {"max_requests": 2, "period": 10.0, "strategy": "delay"}},
        time_func=clock.time,
        sleep_func=clock.sleep
    )

    # First request at t = 1000.0
    await limiter.acquire("discord")
    assert clock.time() == 1000.0

    # Second request at t = 1000.0
    await limiter.acquire("discord")
    assert clock.time() == 1000.0

    # Third request should be delayed until first request expires (t = 1010.0)
    await limiter.acquire("discord")
    assert clock.time() == 1010.0


@pytest.mark.asyncio
async def test_rate_limiter_unconfigured_defaults() -> None:
    """Verify rate limiter falls back to default_limit for unconfigured channels."""
    clock = MockClock()
    limiter = ChannelRateLimiter(
        default_limit={"max_requests": 1, "period": 2.0, "strategy": "block"},
        time_func=clock.time,
        sleep_func=clock.sleep
    )

    # Unconfigured channel uses default_limit
    await limiter.acquire("slack")

    with pytest.raises(RateLimitExceededError):
        await limiter.acquire("slack")

    clock.advance(2.1)
    # Now it works again
    await limiter.acquire("slack")


@pytest.mark.asyncio
async def test_channel_hub_integration(gateway: GatewayRouter) -> None:
    """Verify rate limiting integrated with ChannelHub send_to_channel."""
    clock = MockClock()
    limiter = ChannelRateLimiter(
        limits={"whatsapp": {"max_requests": 1, "period": 5.0, "strategy": "block"}},
        time_func=clock.time,
        sleep_func=clock.sleep
    )

    hub = ChannelHub(rate_limiter=limiter)
    adapter = DummyChannel("whatsapp", gateway)
    hub.register_adapter(adapter)

    # First message goes through
    res1 = await hub.send_to_channel("whatsapp", "user1", "Hello 1")
    assert res1 == "Sent to user1: Hello 1"

    # Second message fails with RateLimitExceededError
    with pytest.raises(RateLimitExceededError):
        await hub.send_to_channel("whatsapp", "user1", "Hello 2")

    # Dynamic setter test
    hub.set_rate_limiter(None)
    # Now it works because rate limiter is removed
    res2 = await hub.send_to_channel("whatsapp", "user1", "Hello 2")
    assert res2 == "Sent to user1: Hello 2"


@pytest.mark.asyncio
async def test_concurrent_delay_serialization() -> None:
    """Verify concurrent requests are serialized and delayed correctly under lock."""
    clock = MockClock()
    limiter = ChannelRateLimiter(
        limits={"telegram": {"max_requests": 1, "period": 1.0, "strategy": "delay"}},
        time_func=clock.time,
        sleep_func=clock.sleep
    )

    # Trigger 3 concurrent acquires
    await asyncio.gather(
        limiter.acquire("telegram"),
        limiter.acquire("telegram"),
        limiter.acquire("telegram")
    )

    # First acquire completes at 1000.0.
    # Second is delayed by 1.0s to 1001.0.
    # Third is delayed by another 1.0s to 1002.0.
    assert clock.time() == 1002.0
