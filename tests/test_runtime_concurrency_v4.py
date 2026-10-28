import asyncio
import time
from typing import Any, Dict, List
import pytest

from magda_agent.skills.runtime_concurrency_v4 import (
    RuntimeFunctionConcurrencyV4,
    OpenAIRuntimeConcurrencyV4,
)
from magda_agent.skills.registry import SkillRegistry


class MockGuardrail:
    """Mock guardrail object for testing validate_call."""

    def __init__(self, blocked_func_name: str = "blocked_func") -> None:
        self.blocked_func_name = blocked_func_name

    def validate_call(self, name: str, kwargs: Dict[str, Any]) -> bool:
        if name == self.blocked_func_name:
            return False
        return True


@pytest.mark.asyncio
async def test_concurrent_execution_async_and_sync() -> None:
    """Tests concurrent execution of mixed async and sync functions."""
    executor = RuntimeFunctionConcurrencyV4(max_concurrency=5)

    async def async_slow_task(x: int) -> int:
        await asyncio.sleep(0.1)
        return x * 2

    def sync_slow_task(y: int) -> int:
        time.sleep(0.1)
        return y + 10

    executor.register_function("async_task", async_slow_task)
    executor.register_function("sync_task", sync_slow_task)

    calls = [
        {"id": "c1", "name": "async_task", "kwargs": {"x": 5}},
        {"id": "c2", "name": "sync_task", "kwargs": {"y": 20}},
        {"id": "c3", "name": "async_task", "kwargs": {"x": 10}},
    ]

    start_time = time.time()
    results = await executor.execute_concurrently(calls)
    duration = time.time() - start_time

    assert len(results) == 3
    # Concurrent execution of three ~0.1s tasks should take significantly less than 0.3s
    assert duration < 0.25

    res_map = {r["id"]: r for r in results}
    assert res_map["c1"]["status"] == "success"
    assert res_map["c1"]["output"] == 10

    assert res_map["c2"]["status"] == "success"
    assert res_map["c2"]["output"] == 30

    assert res_map["c3"]["status"] == "success"
    assert res_map["c3"]["output"] == 20


@pytest.mark.asyncio
async def test_openai_tool_call_format_parsing() -> None:
    """Tests parsing OpenAI tool call schemas with JSON string arguments."""
    executor = OpenAIRuntimeConcurrencyV4()

    async def fetch_weather(location: str, unit: str = "celsius") -> str:
        return f"Weather in {location}: 22 degrees {unit}"

    executor.register_function("fetch_weather", fetch_weather)

    openai_call = {
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "fetch_weather",
            "arguments": '{"location": "Tokyo", "unit": "celsius"}',
        },
    }

    results = await executor.execute_concurrently([openai_call])
    assert len(results) == 1
    res = results[0]

    assert res["id"] == "call_abc123"
    assert res["name"] == "fetch_weather"
    assert res["status"] == "success"
    assert res["output"] == "Weather in Tokyo: 22 degrees celsius"


@pytest.mark.asyncio
async def test_guardrail_blocking_and_error_isolation() -> None:
    """Tests guardrail check blocking a tool while independent tools succeed."""
    guardrail = MockGuardrail(blocked_func_name="dangerous_tool")
    executor = RuntimeFunctionConcurrencyV4(guardrail=guardrail)

    async def safe_tool(val: str) -> str:
        return f"safe_{val}"

    async def dangerous_tool(val: str) -> str:
        return f"dangerous_{val}"

    executor.register_function("safe_tool", safe_tool)
    executor.register_function("dangerous_tool", dangerous_tool)

    calls = [
        {"id": "1", "name": "safe_tool", "kwargs": {"val": "a"}},
        {"id": "2", "name": "dangerous_tool", "kwargs": {"val": "b"}},
        {"id": "3", "name": "safe_tool", "kwargs": {"val": "c"}},
    ]

    results = await executor.execute_concurrently(calls)
    assert len(results) == 3

    res_map = {r["id"]: r for r in results}
    assert res_map["1"]["status"] == "success"
    assert res_map["1"]["output"] == "safe_a"

    assert res_map["2"]["status"] == "error"
    assert "guardrail" in res_map["2"]["error"].lower()

    assert res_map["3"]["status"] == "success"
    assert res_map["3"]["output"] == "safe_c"


@pytest.mark.asyncio
async def test_error_isolation_on_function_exception() -> None:
    """Tests that a function throwing an unhandled exception does not fail other concurrent tasks."""
    executor = RuntimeFunctionConcurrencyV4()

    async def failing_tool() -> None:
        raise ValueError("Simulated unexpected tool failure")

    async def working_tool() -> str:
        return "ok"

    executor.register_function("failing_tool", failing_tool)
    executor.register_function("working_tool", working_tool)

    calls = [
        {"id": "fail_1", "name": "failing_tool", "kwargs": {}},
        {"id": "work_1", "name": "working_tool", "kwargs": {}},
    ]

    results = await executor.execute_concurrently(calls)
    res_map = {r["id"]: r for r in results}

    assert res_map["fail_1"]["status"] == "error"
    assert "Simulated unexpected tool failure" in res_map["fail_1"]["error"]

    assert res_map["work_1"]["status"] == "success"
    assert res_map["work_1"]["output"] == "ok"


@pytest.mark.asyncio
async def test_skill_registry_integration() -> None:
    """Tests resolving tools directly from a SkillRegistry instance."""
    registry = SkillRegistry()

    def multiply(a: int, b: int) -> int:
        return a * b

    registry.register_skill(name="multiply", func=multiply, description="Multiplies numbers")

    executor = RuntimeFunctionConcurrencyV4(registry=registry)

    calls = [
        {"id": "m1", "name": "multiply", "kwargs": {"a": 6, "b": 7}},
    ]

    results = await executor.execute_concurrently(calls)
    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0]["output"] == 42


@pytest.mark.asyncio
async def test_callable_guardrail_hook() -> None:
    """Tests a function/lambda as guardrail hook."""
    def custom_guardrail(name: str, kwargs: Dict[str, Any]) -> bool:
        if kwargs.get("forbidden"):
            return False
        return True

    executor = RuntimeFunctionConcurrencyV4(guardrail=custom_guardrail)

    def echo(text: str, forbidden: bool = False) -> str:
        return text

    executor.register_function("echo", echo)

    calls = [
        {"id": "g1", "name": "echo", "kwargs": {"text": "allowed", "forbidden": False}},
        {"id": "g2", "name": "echo", "kwargs": {"text": "denied", "forbidden": True}},
    ]

    results = await executor.execute_concurrently(calls)
    res_map = {r["id"]: r for r in results}

    assert res_map["g1"]["status"] == "success"
    assert res_map["g1"]["output"] == "allowed"

    assert res_map["g2"]["status"] == "error"
    assert "blocked" in res_map["g2"]["error"].lower()
