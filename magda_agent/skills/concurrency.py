import asyncio
from typing import Any, Dict, List, Coroutine, Callable
import inspect

class ConcurrentSkillExecutor:
    """Executes multiple skills concurrently."""
    def __init__(self, registry: Any) -> None:
        self.registry = registry

    async def execute_concurrently(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """
        Executes a list of tool calls concurrently.
        Each element in tool_calls must have 'name' and 'kwargs'.
        """
        tasks = []
        for call in tool_calls:
            name = call.get("name")
            kwargs = call.get("kwargs", {})

            if name not in self.registry.skills:
                tasks.append(asyncio.to_thread(lambda n=name: f"Error: Skill '{n}' not found."))
                continue

            skill_func = self.registry.skills[name]

            # Using the registry's execute_skill method to pass through all guards
            # The registry's execute_skill method handles execution regardless of
            # whether the underlying skill is async or sync.

            async def wrap_execution(n=name, k=kwargs):
                # Run the synchronous part in a thread to unblock the event loop
                res = await asyncio.to_thread(self.registry.execute_skill, n, **k)
                if inspect.isawaitable(res):
                    return await res
                return res

            tasks.append(wrap_execution())

        return await asyncio.gather(*tasks)
