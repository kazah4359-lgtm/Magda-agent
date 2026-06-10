import pytest
import asyncio
import time
from typing import Any
from magda_agent.skills.concurrency import ConcurrentSkillExecutor

class MockRegistry:
    """Mock registry for testing concurrency."""
    def __init__(self) -> None:
        """Initializes the mock registry."""
        self.skills = {}

    def register_skill(self, name: str, func: Any) -> None:
        """Registers a skill."""
        self.skills[name] = func

    def execute_skill(self, name: str, **kwargs: Any) -> Any:
        """Executes a skill if it exists."""
        if name not in self.skills:
            return f"Error: Skill '{name}' not found."
        return self.skills[name](**kwargs)

def slow_sync_tool(duration: float = 0.1) -> str:
    """A slow synchronous tool to test concurrency."""
    time.sleep(duration)
    return f"done {duration}"

@pytest.mark.asyncio
async def test_concurrent_execution_speeds_up_processing() -> None:
    """Tests that concurrent execution is faster than sequential execution."""
    registry = MockRegistry()
    registry.register_skill("slow", slow_sync_tool)
    executor = ConcurrentSkillExecutor(registry)

    start_time = time.time()
    calls = [{"name": "slow", "kwargs": {"duration": 0.2}} for _ in range(3)]
    results = await executor.execute_concurrently(calls)
    end_time = time.time()

    assert results == ["done 0.2", "done 0.2", "done 0.2"]
    assert end_time - start_time < 0.4  # Should be ~0.2s, not 0.6s

@pytest.mark.asyncio
async def test_missing_skill() -> None:
    """Tests that missing skills are handled gracefully."""
    registry = MockRegistry()
    executor = ConcurrentSkillExecutor(registry)
    calls = [{"name": "missing", "kwargs": {}}]
    results = await executor.execute_concurrently(calls)
    assert results == ["Error: Skill 'missing' not found."]
