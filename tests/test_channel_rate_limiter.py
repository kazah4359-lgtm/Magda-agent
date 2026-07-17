import pytest
import asyncio
from typing import Any, Dict, Optional, Tuple
from magda_agent.channels.base import ChannelAdapter
from magda_agent.channels.hub import ChannelHub
from magda_agent.channels.rate_limiter import ChannelRateLimiter, RateLimitExceeded
from magda_agent.gateway.router import GatewayRouter

class MockAdapterForRateLimit(ChannelAdapter):
    """Simple mock adapter for rate limit testing."""
    def __init__(self, channel_id: str, gateway: GatewayRouter) -> None:
        super().__init__(channel_id, gateway)
        self.sent_messages = []

    async def receive(self, raw_data: Any) -> Any:
        pass

    async def send(self, recipient_id: str, text: str, metadata: Dict[str, Any] = None) -> str:
        self.sent_messages.append((recipient_id, text))
        return f"Sent: {text}"

@pytest.fixture
def gateway() -> GatewayRouter:
    return GatewayRouter()

@pytest.mark.asyncio
async def test_block_mode(gateway: GatewayRouter) -> None:
    """Verify rate limiter in block mode raises RateLimitExceeded when exceeded."""
    virtual_time = 100.0

    def mock_time_func() -> float:
        return virtual_time

    # Rate limit: 2 messages per 10 seconds. Block mode.
    limiter = ChannelRateLimiter(
        default_rate_limit=2,
        default_window=10.0,
        block_mode=True,
        time_func=mock_time_func
    )

    hub = ChannelHub(rate_limiter=limiter)
    adapter = MockAdapterForRateLimit("telegram", gateway)
    hub.register_adapter(adapter)

    # First send - consumes 1 token (1 token remaining)
    res = await hub.send_to_channel("telegram", "user1", "msg1")
    assert res == "Sent: msg1"

    # Second send - consumes 1 token (0 tokens remaining)
    res = await hub.send_to_channel("telegram", "user1", "msg2")
    assert res == "Sent: msg2"

    # Third send - should raise RateLimitExceeded
    with pytest.raises(RateLimitExceeded) as excinfo:
        await hub.send_to_channel("telegram", "user1", "msg3")
    assert "Rate limit exceeded" in str(excinfo.value)

    # Advance virtual time by 5 seconds -> refills 5 * (2/10) = 1 token
    virtual_time += 5.0
    res = await hub.send_to_channel("telegram", "user1", "msg3")
    assert res == "Sent: msg3"

    # Now 0 tokens again, next send raises RateLimitExceeded
    with pytest.raises(RateLimitExceeded):
        await hub.send_to_channel("telegram", "user1", "msg4")


@pytest.mark.asyncio
async def test_delay_mode(gateway: GatewayRouter) -> None:
    """Verify rate limiter in delay mode waits the correct duration."""
    virtual_time = 100.0
    sleeps = []

    def mock_time_func() -> float:
        return virtual_time

    async def mock_sleep_func(seconds: float) -> None:
        nonlocal virtual_time
        sleeps.append(seconds)
        virtual_time += seconds

    # Rate limit: 1 message per 5 seconds. Delay mode.
    limiter = ChannelRateLimiter(
        default_rate_limit=1,
        default_window=5.0,
        block_mode=False,
        time_func=mock_time_func,
        sleep_func=mock_sleep_func
    )

    hub = ChannelHub(rate_limiter=limiter)
    adapter = MockAdapterForRateLimit("telegram", gateway)
    hub.register_adapter(adapter)

    # First send - consumes 1 token immediately (0 remaining)
    res = await hub.send_to_channel("telegram", "user1", "msg1")
    assert res == "Sent: msg1"
    assert len(sleeps) == 0

    # Second send - needs 1 token. Wait time = 1 / refill_rate = 1 / (1/5) = 5.0s.
    res = await hub.send_to_channel("telegram", "user1", "msg2")
    assert res == "Sent: msg2"
    assert len(sleeps) == 1
    assert sleeps[0] == 5.0
    assert virtual_time == 105.0


@pytest.mark.asyncio
async def test_per_channel_limits(gateway: GatewayRouter) -> None:
    """Verify that channel-specific limits override default limits."""
    virtual_time = 100.0

    def mock_time_func() -> float:
        return virtual_time

    # Default: 1 message per 10s.
    # Discord specific override: 3 messages per 10s.
    limiter = ChannelRateLimiter(
        default_rate_limit=1,
        default_window=10.0,
        block_mode=True,
        channel_limits={
            "discord": (3, 10.0)
        },
        time_func=mock_time_func
    )

    hub = ChannelHub(rate_limiter=limiter)
    adapter1 = MockAdapterForRateLimit("telegram", gateway)
    adapter2 = MockAdapterForRateLimit("discord", gateway)
    hub.register_adapter(adapter1)
    hub.register_adapter(adapter2)

    # Telegram uses default: 1 msg/10s
    await hub.send_to_channel("telegram", "user1", "t1")
    with pytest.raises(RateLimitExceeded):
        await hub.send_to_channel("telegram", "user1", "t2")

    # Discord uses override: 3 msg/10s
    await hub.send_to_channel("discord", "user1", "d1")
    await hub.send_to_channel("discord", "user1", "d2")
    await hub.send_to_channel("discord", "user1", "d3")
    with pytest.raises(RateLimitExceeded):
        await hub.send_to_channel("discord", "user1", "d4")
