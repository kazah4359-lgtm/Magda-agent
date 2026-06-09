import httpx
import json
import uuid
from typing import Any, Dict
import asyncio
import logging

class MCPClient:
    """
    A client to execute remote MCP server-prefixed tools.
    """
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.registered_tools: Dict[str, Any] = {}
        self.registered_servers: Dict[str, Any] = {}

    def register_mcp_server(self, server_name: str, connection_info: Any) -> None:
        """Register a remote MCP server for prefixed tool routing."""
        self.registered_servers[server_name] = connection_info
        logging.info(f"Registered remote MCP server: {server_name}")

    def register_remote_tool(self, tool_name: str, connection_info: Any) -> None:
        """Register a remote MCP tool."""
        self.registered_tools[tool_name] = connection_info
        logging.info(f"Registered remote MCP tool: {tool_name}")

    def has_tool(self, name: str) -> bool:
        """Check if a remote tool is registered directly or via server prefix."""
        if name in self.registered_tools:
            return True

        if "__" in name:
            server_name = name.split("__", 1)[0]
            if server_name in self.registered_servers:
                return True
        if "." in name:
            server_name = name.split(".", 1)[0]
            if server_name in self.registered_servers:
                return True

        return False

    async def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a remote MCP tool using JSON-RPC over HTTP.
        """
        connection_info = None
        method_name = name

        if name in self.registered_tools:
            connection_info = self.registered_tools[name]
        else:
            server_name = None
            if "__" in name:
                server_name, extracted_method = name.split("__", 1)
            elif "." in name:
                server_name, extracted_method = name.split(".", 1)

            if server_name and server_name in self.registered_servers:
                connection_info = self.registered_servers[server_name]
                method_name = extracted_method

        if not connection_info:
            return f"Error: Remote MCP skill '{name}' not found."

        url = connection_info.get("url")
        if not url:
            return f"Error: Remote MCP skill '{name}' has no URL configured."

        payload = {
            "jsonrpc": "2.0",
            "method": method_name,
            "params": kwargs,
            "id": str(uuid.uuid4())
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    return f"Error from remote MCP server: {data['error']}"
                return data.get("result", "No result returned.")
        except Exception as e:
            logging.error(f"Failed to execute remote MCP tool {name} at {url}: {e}")
            return f"Error executing remote MCP tool {name}: {e}"
