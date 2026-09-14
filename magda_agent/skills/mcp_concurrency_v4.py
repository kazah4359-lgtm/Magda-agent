"""
MCP Action Tool Concurrency Manager v4.

This module provides concurrent, isolated execution of Model Context Protocol (MCP) action tools,
supporting server prefix routing, rate/concurrency limiting, and offloading synchronous tool functions to worker threads.
"""

import asyncio
import inspect
import json
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union


class MCPConcurrencyManagerV4:
    """
    Safely executes MCP action tools concurrently.

    Features:
    - Parallel tool execution using asyncio.gather with per-server and global semaphores.
    - Automatic offloading of synchronous tool functions to worker threads via asyncio.to_thread.
    - Namespace routing for server-prefixed tool names with configurable separators ('__', '-', '_').
    - Error and exception isolation: failure in one tool call does not abort other concurrent tasks.
    - Support for both direct tool call dictionaries and JSON-RPC 2.0 single/batch payloads.
    - Thread-safe active task counting and status inspection.
    """

    def __init__(
        self,
        registry: Optional[Any] = None,
        servers: Optional[Dict[str, Any]] = None,
        max_concurrency: int = 10,
        max_concurrency_per_server: int = 5,
        separators: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the MCPConcurrencyManagerV4.

        Args:
            registry (Optional[Any]): The default local SkillRegistry instance.
            servers (Optional[Dict[str, Any]]): Dict mapping server prefixes to their respective server/registry instances.
            max_concurrency (int): Global maximum number of concurrent executions allowed.
            max_concurrency_per_server (int): Maximum concurrent executions allowed per individual server prefix.
            separators (Optional[List[str]]): List of prefix separators to check for namespacing.
        """
        self.registry = registry
        self.servers: Dict[str, Any] = servers.copy() if servers else {}
        self.max_concurrency = max_concurrency
        self.max_concurrency_per_server = max_concurrency_per_server
        self.separators = separators or ["__", "-", "_"]

        self.global_semaphore = asyncio.Semaphore(max_concurrency)
        self.server_semaphores: Dict[str, asyncio.Semaphore] = {
            prefix: asyncio.Semaphore(max_concurrency_per_server)
            for prefix in self.servers
        }

        self._lock = threading.Lock()
        self._active_tasks_count = 0

    @property
    def active_tasks_count(self) -> int:
        """Returns the current number of active concurrent tool execution tasks."""
        with self._lock:
            return self._active_tasks_count

    def register_server(self, prefix: str, server: Any) -> None:
        """
        Registers an MCP server instance with a specific prefix.

        Args:
            prefix (str): The prefix/namespace associated with the server.
            server (Any): The server or SkillRegistry instance.
        """
        self.servers[prefix] = server
        if prefix not in self.server_semaphores:
            self.server_semaphores[prefix] = asyncio.Semaphore(self.max_concurrency_per_server)

    def _resolve_tool(self, name: str) -> Tuple[Optional[str], Optional[Any], str]:
        """
        Resolves a full tool name to its matching server prefix, target server instance, and unprefixed name.

        Args:
            name (str): Full tool name.

        Returns:
            Tuple[Optional[str], Optional[Any], str]: (prefix, target_server, unprefixed_name)
        """
        if not name:
            return None, None, ""

        sorted_prefixes = sorted(self.servers.keys(), key=len, reverse=True)
        for prefix in sorted_prefixes:
            if not prefix:
                continue
            for sep in self.separators:
                full_prefix = f"{prefix}{sep}"
                if name.startswith(full_prefix):
                    unprefixed = name[len(full_prefix):]
                    return prefix, self.servers[prefix], unprefixed

        if "" in self.servers:
            return "", self.servers[""], name

        return None, None, name

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists available tools across all registered servers with their prefixed tool names.

        Returns:
            List[Dict[str, Any]]: List of tool definition dictionaries.
        """
        all_tools: List[Dict[str, Any]] = []

        # Local default registry tools
        if self.registry:
            if hasattr(self.registry, "list_skills"):
                skills = self.registry.list_skills()
                for s in skills:
                    if isinstance(s, dict):
                        all_tools.append(s)
                    else:
                        all_tools.append({"name": str(s), "description": ""})
            elif hasattr(self.registry, "skills") and isinstance(self.registry.skills, dict):
                descriptions = getattr(self.registry, "descriptions", {})
                for name, fn in self.registry.skills.items():
                    desc = descriptions.get(name, getattr(fn, "__doc__", "") or "")
                    all_tools.append({"name": name, "description": desc})

        # Registered server tools
        for prefix, server in self.servers.items():
            tools = []
            if hasattr(server, "list_tools"):
                tools = server.list_tools()
            elif hasattr(server, "list_skills"):
                tools = server.list_skills()
            elif hasattr(server, "skills") and isinstance(server.skills, dict):
                descriptions = getattr(server, "descriptions", {})
                tools = [{"name": name, "description": descriptions.get(name, getattr(fn, "__doc__", "") or "")} for name, fn in server.skills.items()]

            for tool in tools:
                tool_dict = dict(tool) if isinstance(tool, dict) else {"name": str(tool)}
                tool_name = tool_dict.get("name", "")

                if not prefix:
                    all_tools.append(tool_dict)
                    continue

                sep = self.separators[0] if self.separators else "__"
                tool_dict["name"] = f"{prefix}{sep}{tool_name}"
                all_tools.append(tool_dict)

        return all_tools

    async def _execute_single_tool_call(self, name: str, kwargs: Dict[str, Any]) -> Any:
        """
        Executes a single tool call with safety guarantees, concurrency limits, and error isolation.

        Args:
            name (str): Tool/skill name.
            kwargs (Dict[str, Any]): Arguments dictionary.

        Returns:
            Any: The execution result or an isolated error message string.
        """
        if not name or not isinstance(name, str):
            return "Error: Empty or invalid tool name provided."

        prefix, server, unprefixed_name = self._resolve_tool(name)
        server_sem = self.server_semaphores.get(prefix) if prefix else None

        async def _run_tool() -> Any:
            with self._lock:
                self._active_tasks_count += 1
            try:
                if server is not None:
                    # Case 1: Exporter handle_rpc_request
                    if hasattr(server, "exporter") and hasattr(server.exporter, "handle_rpc_request"):
                        rpc_req = {
                            "jsonrpc": "2.0",
                            "id": str(uuid.uuid4()),
                            "method": unprefixed_name,
                            "params": kwargs,
                        }
                        res = await server.exporter.handle_rpc_request(rpc_req)
                        if isinstance(res, dict) and "error" in res:
                            return f"Error: {res['error'].get('message', 'Unknown RPC error')}"
                        if isinstance(res, dict) and "result" in res:
                            inner = res["result"]
                            if isinstance(inner, dict) and "content" in inner:
                                content = inner["content"]
                                if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                                    return content[0]["text"]
                            return inner
                        return res

                    # Case 2: MCPServer or object with handle_request
                    elif hasattr(server, "handle_request"):
                        rpc_req = {
                            "jsonrpc": "2.0",
                            "id": str(uuid.uuid4()),
                            "method": name,
                            "params": kwargs,
                        }
                        res_str = await server.handle_request(json.dumps(rpc_req))
                        res = json.loads(res_str)
                        if isinstance(res, dict) and "error" in res:
                            return f"Error: {res['error'].get('message', 'Unknown RPC error')}"
                        if isinstance(res, dict) and "result" in res:
                            inner = res["result"]
                            if isinstance(inner, dict) and "content" in inner:
                                content = inner["content"]
                                if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                                    return content[0]["text"]
                            return inner
                        return res

                    # Case 3: SkillRegistry or server with execute_skill and skills
                    elif hasattr(server, "execute_skill") and hasattr(server, "skills"):
                        if unprefixed_name not in server.skills:
                            return f"Error: Skill '{unprefixed_name}' not found on server '{prefix}'."
                        skill_fn = server.skills[unprefixed_name]
                        if inspect.iscoroutinefunction(skill_fn):
                            res = await server.execute_skill(unprefixed_name, **kwargs)
                        else:
                            res = await asyncio.to_thread(server.execute_skill, unprefixed_name, **kwargs)
                        if inspect.isawaitable(res):
                            res = await res
                        return res

                    # Case 4: Callable server endpoint
                    elif callable(server):
                        if inspect.iscoroutinefunction(server):
                            return await server(unprefixed_name, **kwargs)
                        else:
                            return await asyncio.to_thread(server, unprefixed_name, **kwargs)

                    else:
                        return f"Error: Unsupported interface for server prefix '{prefix}'."

                else:
                    # Case 5: Default local SkillRegistry
                    if not self.registry or not hasattr(self.registry, "skills") or name not in self.registry.skills:
                        return f"Error: Skill '{name}' not found."

                    skill_fn = self.registry.skills[name]
                    if inspect.iscoroutinefunction(skill_fn):
                        res = self.registry.execute_skill(name, **kwargs)
                        if inspect.isawaitable(res):
                            res = await res
                        return res
                    else:
                        res = await asyncio.to_thread(self.registry.execute_skill, name, **kwargs)
                        if inspect.isawaitable(res):
                            res = await res
                        return res

            except Exception as e:
                return f"Error: Tool execution failed: {str(e)}"
            finally:
                with self._lock:
                    self._active_tasks_count -= 1

        async with self.global_semaphore:
            if server_sem:
                async with server_sem:
                    return await _run_tool()
            else:
                return await _run_tool()

    async def execute_concurrently(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """
        Executes a batch of tool calls concurrently.

        Args:
            tool_calls (List[Dict[str, Any]]): List of tool call dicts containing 'name' and optionally 'kwargs'.

        Returns:
            List[Any]: List of execution results corresponding to the input tool calls order.
        """
        tasks = [
            self._execute_single_tool_call(call.get("name", ""), call.get("kwargs", {}))
            for call in tool_calls
        ]
        return await asyncio.gather(*tasks)

    async def handle_request(self, payload: str) -> str:
        """
        Processes a single or batch JSON-RPC request string concurrently.

        Args:
            payload (str): JSON string representing a single JSON-RPC request or an array of requests.

        Returns:
            str: JSON string representing JSON-RPC response or array of responses.
        """
        try:
            req_data = json.loads(payload)
        except json.JSONDecodeError:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            })

        if isinstance(req_data, list):
            if not req_data:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"},
                })

            async def _process_batch_item(req: Any) -> Dict[str, Any]:
                if not isinstance(req, dict):
                    return {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32600, "message": "Invalid Request"},
                    }
                req_id = req.get("id")
                method = req.get("method")
                params = req.get("params", {})
                if not method or not isinstance(method, str):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": "Method not found"},
                    }
                res = await self._execute_single_tool_call(method, params)
                if isinstance(res, str) and res.startswith("Error"):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32000, "message": res},
                    }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": str(res)}],
                        "isError": False,
                    },
                }

            results = await asyncio.gather(*[_process_batch_item(item) for item in req_data])
            return json.dumps(results)
        elif isinstance(req_data, dict):
            req_id = req_data.get("id")
            method = req_data.get("method")
            params = req_data.get("params", {})
            if not method or not isinstance(method, str):
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "Method not found"},
                })

            res = await self._execute_single_tool_call(method, params)
            if isinstance(res, str) and res.startswith("Error"):
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": res},
                })

            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(res)}],
                    "isError": False,
                },
            })
        else:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            })


# Class Alias
MCPActionToolConcurrencyV4 = MCPConcurrencyManagerV4
