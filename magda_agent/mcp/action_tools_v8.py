import json
import logging
import inspect
from typing import Dict, Any, List, Optional, Tuple

from magda_agent.skills.registry import SkillRegistry
from magda_agent.safety.policy import PolicyLayer
from magda_agent.integration.mcp_preflight import MCPPreflightValidator

logger = logging.getLogger(__name__)

class MCPActionToolManagerV8:
    """
    Manages state-mutating action tools exported via MCP.
    Protected by a PolicyLayer and integrated with SkillRegistry.
    """

    def __init__(self, registry: SkillRegistry, policy_layer: PolicyLayer) -> None:
        """
        Initialize the MCP Action Tool Manager.

        Args:
            registry: The SkillRegistry to execute skills.
            policy_layer: The PolicyLayer to gate executions.
        """
        self.registry = registry
        self.policy_layer = policy_layer
        self.action_tools: Dict[str, Dict[str, Any]] = {}
        self.validator = MCPPreflightValidator(registry=self.registry)

    def register_action_tool(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        """
        Registers an action tool metadata.

        Args:
            name: The name of the tool.
            description: A description of the tool.
            input_schema: JSON schema for tool inputs.
        """
        self.action_tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "is_action": True
        }
        logger.info(f"Registered MCP action tool: {name}")

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists all registered action tools in MCP format.
        """
        return list(self.action_tools.values())

    async def handle_mcp_request(self, payload: str) -> str:
        """
        Processes a JSON-RPC 2.0 payload.

        Args:
            payload: JSON string of the request.

        Returns:
            JSON string of the response.
        """
        try:
            request = json.loads(payload)
        except json.JSONDecodeError:
            return self._error_response(None, -32700, "Parse error")

        req_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            return self._error_response(req_id, -32600, "Invalid Request")

        method = request.get("method")
        params = request.get("params", {})

        if method == "list_tools":
            return self._result_response(req_id, {"tools": self.list_tools()})

        if method == "call_tool":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if not tool_name:
                return self._error_response(req_id, -32602, "Invalid params: 'name' is required")

            # 1. Pre-flight Validation
            virtual_request = {
                "jsonrpc": "2.0",
                "method": tool_name,
                "params": arguments
            }
            is_valid, err_code, err_msg = self.validator.validate_request_dict(virtual_request)
            if not is_valid:
                logger.warning(f"MCP Action Tool '{tool_name}' failed pre-flight: {err_msg}")
                return self._error_response(req_id, err_code, err_msg)

            if tool_name not in self.action_tools:
                return self._error_response(req_id, -32601, f"Tool '{tool_name}' not found")

            return await self._execute_tool(req_id, tool_name, arguments)

        return self._error_response(req_id, -32601, f"Method '{method}' not found")

    async def _execute_tool(self, req_id: Any, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Evaluates policy and executes the tool.
        """
        # 1. Policy Gating
        allowed, explanation = self.policy_layer.evaluate(tool_name, **arguments)
        if not allowed:
            logger.warning(f"MCP Action Tool '{tool_name}' denied by policy: {explanation}")
            return self._error_response(req_id, -32000, f"Policy violation: {explanation}")

        # 2. Execution via SkillRegistry
        try:
            result = self.registry.execute_skill(tool_name, **arguments)

            if inspect.isawaitable(result):
                result = await result

            # Wrap result in MCP content format
            mcp_result = {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False
            }

            # Check if SkillRegistry returned an error string
            if isinstance(result, str) and result.startswith("Error:"):
                mcp_result["isError"] = True

            return self._result_response(req_id, mcp_result)

        except Exception as e:
            logger.error(f"Error executing MCP action tool '{tool_name}': {e}")
            return self._error_response(req_id, -32000, f"Execution error: {str(e)}")

    def _error_response(self, req_id: Any, code: int, message: str) -> str:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message}
        })

    def _result_response(self, req_id: Any, result: Any) -> str:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result
        })
