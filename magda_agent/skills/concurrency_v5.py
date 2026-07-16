import asyncio
import inspect
import json
from typing import Any, Dict, List, Optional, Tuple


class ConcurrentToolRouterV5:
    """
    Handles routing and concurrent execution of server-prefixed tool calls,
    optimizing performance via asyncio.gather and ensuring safety and error isolation.
    """

    def __init__(
        self,
        registry: Any,
        servers: Optional[Dict[str, Any]] = None,
        max_concurrency_per_server: int = 5,
        separators: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the ConcurrentToolRouterV5.

        Args:
            registry (Any): The default local skill registry.
            servers (Optional[Dict[str, Any]]): Dict mapping server prefixes to their respective MCPServer/Registry instances.
            max_concurrency_per_server (int): Max parallel tasks allowed per individual server prefix.
            separators (Optional[List[str]]): List of separators to check when routing prefix-prefixed methods.
        """
        self.registry = registry
        self.servers = servers or {}
        self.separators = separators or ["__", "-", "_"]

        # Concurrency Isolation: individual semaphores per prefix to prevent any single server prefix
        # from starving others or overwhelming system resources.
        self.semaphores: Dict[str, asyncio.Semaphore] = {
            prefix: asyncio.Semaphore(max_concurrency_per_server)
            for prefix in self.servers
        }
        self.default_semaphore = asyncio.Semaphore(max_concurrency_per_server)

    def _resolve_tool(self, name: str) -> Tuple[Optional[str], Optional[Any], str]:
        """
        Resolves a tool name to its respective server prefix, mapped server instance,
        and its stripped, unprefixed name.

        Args:
            name (str): The full tool/skill name.

        Returns:
            Tuple[Optional[str], Optional[Any], str]: (prefix, target_server, unprefixed_name)
        """
        if not name:
            return None, None, ""

        # Sort prefixes by length descending to match longest prefixes first
        sorted_prefixes = sorted(self.servers.keys(), key=len, reverse=True)
        for prefix in sorted_prefixes:
            if not prefix:
                continue
            for sep in self.separators:
                full_prefix = f"{prefix}{sep}"
                if name.startswith(full_prefix):
                    unprefixed = name[len(full_prefix):]
                    return prefix, self.servers[prefix], unprefixed

        return None, None, name

    def _unwrap_result(self, result: Any) -> Any:
        """
        Unwraps and formats standard JSON-RPC or MCP response formats into clean outputs.

        Args:
            result (Any): The raw execution result object.

        Returns:
            Any: Cleaned unwrapped result text/data.
        """
        if isinstance(result, dict):
            # Case 1: Full JSON-RPC response format
            if "result" in result:
                inner = result["result"]
                if isinstance(inner, dict) and "content" in inner:
                    content = inner["content"]
                    if isinstance(content, list) and len(content) > 0:
                        return content[0].get("text", result)
                return inner
            # Case 2: Direct MCP tool response format
            if "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", result)
        return result

    async def _execute_single_call(self, name: Optional[str], kwargs: Dict[str, Any]) -> Any:
        """
        Executes a single tool call with safety and exception isolation.

        Args:
            name (Optional[str]): The tool/skill name.
            kwargs (Dict[str, Any]): The arguments dictionary.

        Returns:
            Any: The execution result or an isolated error string.
        """
        if not name:
            return "Error: Empty tool name provided."

        prefix, server, unprefixed_name = self._resolve_tool(name)
        sem = self.semaphores.get(prefix, self.default_semaphore) if prefix else self.default_semaphore

        async with sem:
            try:
                if server is not None:
                    # Case A: Execute on mapped server instance
                    if hasattr(server, "exporter") and hasattr(server.exporter, "handle_rpc_request"):
                        rpc_req = {
                            "jsonrpc": "2.0",
                            "id": "router-req",
                            "method": unprefixed_name,
                            "params": kwargs
                        }
                        res = await server.exporter.handle_rpc_request(rpc_req)
                        if isinstance(res, dict) and "error" in res:
                            err_msg = res["error"].get("message", "Unknown RPC error")
                            return f"Error: {err_msg}"
                        return self._unwrap_result(res)

                    elif hasattr(server, "execute_skill") and hasattr(server, "skills"):
                        if unprefixed_name not in server.skills:
                            return f"Error: Skill '{unprefixed_name}' not found on server '{prefix}'."

                        # Check if execute_skill itself is a coroutine function
                        if inspect.iscoroutinefunction(server.execute_skill):
                            res = await server.execute_skill(unprefixed_name, **kwargs)
                        else:
                            skill_func = server.skills[unprefixed_name]
                            if inspect.iscoroutinefunction(skill_func):
                                res = server.execute_skill(unprefixed_name, **kwargs)
                            else:
                                res = await asyncio.to_thread(server.execute_skill, unprefixed_name, **kwargs)

                        if inspect.isawaitable(res):
                            res = await res
                        return self._unwrap_result(res)

                    elif hasattr(server, "handle_request"):
                        rpc_req = {
                            "jsonrpc": "2.0",
                            "id": "router-req",
                            "method": name,
                            "params": kwargs
                        }
                        res_str = await server.handle_request(json.dumps(rpc_req))
                        res = json.loads(res_str)
                        if isinstance(res, dict) and "error" in res:
                            err_msg = res["error"].get("message", "Unknown RPC error")
                            return f"Error: {err_msg}"
                        return self._unwrap_result(res)

                    elif callable(server):
                        if inspect.iscoroutinefunction(server):
                            res = await server(unprefixed_name, **kwargs)
                        else:
                            res = await asyncio.to_thread(server, unprefixed_name, **kwargs)
                        return self._unwrap_result(res)

                    else:
                        return f"Error: Server '{prefix}' interface not supported."

                else:
                    # Case B: Execute locally against main default registry
                    if not self.registry or not hasattr(self.registry, "skills") or name not in self.registry.skills:
                        return f"Error: Skill '{name}' not found."

                    skill_func = self.registry.skills[name]
                    is_async = inspect.iscoroutinefunction(skill_func)

                    if is_async:
                        res = self.registry.execute_skill(name, **kwargs)
                        if inspect.isawaitable(res):
                            res = await res
                    else:
                        res = await asyncio.to_thread(self.registry.execute_skill, name, **kwargs)
                        if inspect.isawaitable(res):
                            res = await res

                    return self._unwrap_result(res)

            except Exception as e:
                # Safety/Exception Isolation: Wrap and isolate error without breaking other tasks in the gather
                return f"Error: Tool execution failed: {str(e)}"

    async def execute_concurrently(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """
        Executes multiple tool calls concurrently using asyncio.gather.

        Args:
            tool_calls (List[Dict[str, Any]]): A list of dictionaries representing tool calls,
                                              where each dict contains 'name' and optionally 'kwargs'.

        Returns:
            List[Any]: A list of results corresponding to the completed tasks.
        """
        tasks = []
        for call in tool_calls:
            name = call.get("name")
            kwargs = call.get("kwargs", {})
            tasks.append(self._execute_single_call(name, kwargs))

        return await asyncio.gather(*tasks)
