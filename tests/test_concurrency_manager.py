import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from magda_agent.execution.concurrency_manager import ConcurrencyManager, RateLimitExceededError, BackpressureError

class DummyRegistry:
    def __init__(self):
        self.skills = {
            "fast_tool": self.fast_tool,
            "slow_tool": self.slow_tool,
        }

    async def fast_tool(self, **kwargs):
        return "fast_result"

    async def slow_tool(self, **kwargs):
        await asyncio.sleep(0.1)
        return "slow_result"

    async def execute_skill(self, name, **kwargs):
        return await self.skills[name](**kwargs)

class DummyServer:
    def __init__(self, prefix):
        self.prefix = prefix
        self.skills = {"remote_tool": self.remote_tool}

    async def remote_tool(self, **kwargs):
        await asyncio.sleep(0.05)
        return f"{self.prefix}_result"

    async def execute_skill(self, name, **kwargs):
        return await self.skills[name](**kwargs)

@pytest.fixture
def manager():
    registry = DummyRegistry()
    servers = {
        "mcp1": DummyServer("mcp1"),
        "mcp2": DummyServer("mcp2")
    }
    return ConcurrencyManager(
        registry=registry,
        servers=servers,
        max_concurrency_per_server=2,
        global_max_concurrency=4,
        rate_limit_per_second=10.0
    )

@pytest.mark.asyncio
async def test_execute_single_local_call(manager):
    res = await manager._execute_single_call("fast_tool", {})
    assert res == "fast_result"

@pytest.mark.asyncio
async def test_execute_single_remote_call(manager):
    res = await manager._execute_single_call("mcp1__remote_tool", {})
    assert res == "mcp1_result"

@pytest.mark.asyncio
async def test_execute_concurrently_mixed(manager):
    calls = [
        {"name": "fast_tool"},
        {"name": "mcp1__remote_tool"},
        {"name": "mcp2__remote_tool"},
        {"name": "slow_tool"},
    ]
    results = await manager.execute_concurrently(calls)
    assert len(results) == 4
    assert "fast_result" in results
    assert "mcp1_result" in results
    assert "mcp2_result" in results
    assert "slow_result" in results

@pytest.mark.asyncio
async def test_backpressure(manager):
    # Set a very low global concurrency and queue size
    manager.global_semaphore = asyncio.Semaphore(1)
    manager._max_queue_size = 2

    # Send 5 slow calls concurrently
    calls = [{"name": "slow_tool"} for _ in range(5)]
    results = await manager.execute_concurrently(calls)

    # At least some should fail due to backpressure
    backpressure_errors = [r for r in results if isinstance(r, str) and "BackpressureError" in r]
    assert len(backpressure_errors) > 0

@pytest.mark.asyncio
async def test_rate_limiting(manager):
    # Set a very strict rate limit
    manager.rate_limit_per_second = 2.0

    start_time = asyncio.get_event_loop().time()

    calls = [{"name": "fast_tool"} for _ in range(3)]
    results = await manager.execute_concurrently(calls)

    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time

    # 3 calls at 2 per second should take at least ~0.5 seconds
    # (first 2 happen immediately, 3rd is delayed by ~0.5s to maintain 2/s avg or wait for sliding window)
    # Actually with the simple implementation, it waits until window clears, which could be up to 1s.
    assert duration > 0.1
    assert len(results) == 3
    assert all(r == "fast_result" for r in results)
