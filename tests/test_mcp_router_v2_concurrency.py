import asyncio
import time
import pytest
from magda_agent.skills.mcp_router_v2_concurrency import MCPConcurrentRouterV2

def slow_sync_handler(tool_name: str, arguments: dict) -> str:
    """A mock handler that takes 0.1 seconds to execute synchronously."""
    time.sleep(0.1)
    return f"Executed {tool_name} with {arguments}"

async def slow_async_handler(tool_name: str, arguments: dict) -> str:
    """A mock handler that takes 0.1 seconds to execute asynchronously."""
    await asyncio.sleep(0.1)
    return f"Async executed {tool_name} with {arguments}"

def error_handler(tool_name: str, arguments: dict) -> str:
    """A mock handler that always raises an exception."""
    raise RuntimeError("Intentional error for testing")

@pytest.mark.asyncio
async def test_concurrent_execution_timing_and_results():
    """Test that concurrent execution runs multiple tools in parallel, saving time."""
    router = MCPConcurrentRouterV2()
    router.register_server("weather", slow_sync_handler)
    router.register_server("asyncweather", slow_async_handler)

    tool_calls = [
        {"name": "weather_get_forecast", "kwargs": {"location": "London"}},
        {"name": "weather_get_temperature", "kwargs": {"location": "London"}},
        {"name": "asyncweather_get_wind", "kwargs": {"location": "London"}},
    ]

    start_time = time.monotonic()
    results = await router.execute_concurrently(tool_calls)
    end_time = time.monotonic()

    duration = end_time - start_time

    # If executed sequentially, it would take ~0.3 seconds.
    # Concurrently, it should take slightly more than 0.1 seconds.
    assert duration < 0.25, f"Execution took too long: {duration}s"

    assert len(results) == 3
    assert "Executed get_forecast" in results[0]
    assert "Executed get_temperature" in results[1]
    assert "Async executed get_wind" in results[2]

@pytest.mark.asyncio
async def test_error_handling():
    """Test that errors during execution are caught and returned as strings."""
    router = MCPConcurrentRouterV2()
    router.register_server("failing", error_handler)

    tool_calls = [
        {"name": "failing_tool", "kwargs": {}},
        {"name": "missingprefix", "kwargs": {}},
        {"name": "unknown_tool", "kwargs": {}},
        {"kwargs": {}}  # missing name
    ]

    results = await router.execute_concurrently(tool_calls)

    assert len(results) == 4
    assert "Error executing tool 'failing_tool'" in results[0]
    assert "Intentional error for testing" in results[0]

    assert "must be prefixed with a server name" in results[1]
    assert "No registered MCP server handler found" in results[2]
    assert "Error: Tool name is missing." in results[3]
