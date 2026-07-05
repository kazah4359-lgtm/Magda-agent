import asyncio
import inspect
from typing import Any, Dict, List, Optional


class ConcurrentToolExecutorV1:
    """Executes multiple skills concurrently, with dynamic handling of async skills."""

    def __init__(self, registry: Any, mcp_registry: Optional[Any] = None) -> None:
        """
        Initialize the ConcurrentToolExecutorV1.

        Args:
            registry (Any): The main skill registry containing callable skills.
            mcp_registry (Optional[Any]): The MCP tool registry to look up metadata.
        """
        self.registry = registry
        self.mcp_registry = mcp_registry

    async def execute_concurrently(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """
        Executes a list of tool calls concurrently.
        Each element in tool_calls must have 'name' and 'kwargs'.

        Args:
            tool_calls (List[Dict[str, Any]]): The list of tool calls to execute.

        Returns:
            List[Any]: The list of results from the tool calls.
        """
        tasks = []
        for call in tool_calls:
            name = call.get("name")
            kwargs = call.get("kwargs", {})

            if not name or not hasattr(self.registry, "skills") or name not in self.registry.skills:
                tasks.append(asyncio.to_thread(lambda n=name: f"Error: Skill '{n}' not found."))
                continue

            skill_func = self.registry.skills[name]

            # Determine if the skill is async
            is_async = False

            # Check metadata from MCP registry if available
            if self.mcp_registry and hasattr(self.mcp_registry, "get_tool"):
                tool_metadata = self.mcp_registry.get_tool(name)
                if tool_metadata and isinstance(tool_metadata, dict):
                    is_async = tool_metadata.get("__is_async__", False)

            # Fallback to standard inspect
            if not is_async and inspect.iscoroutinefunction(skill_func):
                is_async = True

            async def wrap_execution(n: str = name, k: Dict[str, Any] = kwargs, is_async_skill: bool = is_async) -> Any:
                if is_async_skill:
                    # Execute async natively
                    res = self.registry.execute_skill(n, **k)
                    if inspect.isawaitable(res):
                        return await res
                    return res
                else:
                    # Run the synchronous part in a thread to unblock the event loop
                    res = await asyncio.to_thread(self.registry.execute_skill, n, **k)
                    if inspect.isawaitable(res):
                        return await res
                    return res

            tasks.append(wrap_execution())

        return await asyncio.gather(*tasks)
