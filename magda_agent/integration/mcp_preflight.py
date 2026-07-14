"""MCP Action Tools Pre-flight Validation.

Provides a pre-flight validation mechanism for all MCP action tools before invoking JSON-RPC.
"""

import json
import re
import logging
from typing import Any, Dict, List, Tuple, Optional
from magda_agent.skills.registry import SkillRegistry
from magda_agent.safety.taint import is_tainted
from magda_agent.integration.mcp_server import MCPServer

# Set up logging
logger = logging.getLogger(__name__)

# Common hazardous injection and path traversal pattern compiled regexes
_HAZARDOUS_SHELL_PATTERNS = [
    re.compile(r"rm\s+-rf", re.IGNORECASE),
    re.compile(r";\s*bash", re.IGNORECASE),
    re.compile(r";\s*sh", re.IGNORECASE),
    re.compile(r"\bcurl\b", re.IGNORECASE),
    re.compile(r"\bwget\b", re.IGNORECASE),
]

_PATH_TRAVERSAL_PATTERNS = [
    re.compile(r"\.\./\.\."),
    re.compile(r"/etc/passwd"),
    re.compile(r"/etc/hosts"),
    re.compile(r"\.env\b", re.IGNORECASE),
    re.compile(r"id_rsa", re.IGNORECASE),
]

_SQL_INJECTION_PATTERNS = [
    re.compile(r"'\s*or\s*'\d+'\s*=\s*'\d+", re.IGNORECASE),
    re.compile(r"'\s*or\s*\d+\s*=\s*\d+", re.IGNORECASE),
    re.compile(r"union\s+select", re.IGNORECASE),
]


class MCPPreflightValidator:
    """
    Validator to perform pre-flight checks on MCP JSON-RPC requests
    before executing the underlying tool or invoking the client.
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        forbidden_tools: Optional[List[str]] = None
    ) -> None:
        """
        Initializes the MCPPreflightValidator.

        Args:
            registry (Optional[SkillRegistry]): The skill registry to validate against.
            forbidden_tools (Optional[List[str]]): List of explicitly blacklisted tool names.
        """
        self.registry = registry
        self.forbidden_tools = forbidden_tools or ["forbidden_tool", "unsafe_execute", "nuke_system"]

    def _check_string_for_hazards(self, val: str) -> Tuple[bool, str]:
        """
        Inspects a single string for hazardous patterns (shell injection, path traversal, SQLi).

        Args:
            val (str): The string parameter to check.

        Returns:
            Tuple[bool, str]: (is_safe, error_reason)
        """
        for pattern in _HAZARDOUS_SHELL_PATTERNS:
            if pattern.search(val):
                return False, f"Hazardous shell pattern detected: {pattern.pattern}"

        for pattern in _PATH_TRAVERSAL_PATTERNS:
            if pattern.search(val):
                return False, f"Hazardous path traversal pattern detected: {pattern.pattern}"

        for pattern in _SQL_INJECTION_PATTERNS:
            if pattern.search(val):
                return False, f"Hazardous SQL injection pattern detected: {pattern.pattern}"

        return True, ""

    def _validate_value_recursively(self, val: Any) -> Tuple[bool, str]:
        """
        Recursively checks parameter values for hazardous strings or taint.

        Args:
            val (Any): Any value to check.

        Returns:
            Tuple[bool, str]: (is_safe, error_reason)
        """
        if is_tainted(val):
            return False, "Tainted data detected in parameters."

        if isinstance(val, str):
            is_safe, reason = self._check_string_for_hazards(val)
            if not is_safe:
                return False, reason

        elif isinstance(val, list):
            for item in val:
                is_safe, reason = self._validate_value_recursively(item)
                if not is_safe:
                    return False, reason

        elif isinstance(val, dict):
            for k, v in val.items():
                # Check key
                is_safe, reason = self._validate_value_recursively(k)
                if not is_safe:
                    return False, reason
                # Check value
                is_safe, reason = self._validate_value_recursively(v)
                if not is_safe:
                    return False, reason

        return True, ""

    def validate_request_dict(self, request: Dict[str, Any]) -> Tuple[bool, int, str]:
        """
        Validates a parsed JSON-RPC request dictionary.

        Args:
            request (Dict[str, Any]): The JSON-RPC request dictionary.

        Returns:
            Tuple[bool, int, str]: (is_valid, error_code, error_message)
        """
        if not isinstance(request, dict):
            return False, -32600, "Invalid Request: request must be an object."

        # Validate JSON-RPC version
        if request.get("jsonrpc") != "2.0":
            return False, -32600, "Invalid Request: jsonrpc version must be '2.0'."

        method = request.get("method")
        if not method or not isinstance(method, str):
            return False, -32600, "Invalid Request: 'method' must be a non-empty string."

        # Safety Check: Tainted tool names
        if is_tainted(method):
            return False, -32000, "Pre-flight Blocked: tool name is tainted."

        # Safety Check: Explicitly forbidden tools
        if method in self.forbidden_tools:
            return False, -32000, f"Pre-flight Blocked: Tool '{method}' is blacklisted."

        params = request.get("params", {})
        if not isinstance(params, dict):
            return False, -32602, "Invalid params: 'params' must be an object."

        # Safety Check: Recursively inspect parameters for taint and hazardous patterns
        is_safe, reason = self._validate_value_recursively(params)
        if not is_safe:
            return False, -32000, f"Pre-flight Blocked: {reason}"

        # Skill Registry Schema and existence verification (if registry is available)
        if self.registry:
            # Strip server prefix if the registry itself uses local names
            method_to_check = method
            # If the registry doesn't have the prefixed name, try has_skill
            if not self.registry.has_skill(method_to_check):
                return False, -32601, f"Method not found: '{method}' is not registered."

        return True, 0, ""

    def validate_payload(self, payload: str) -> Tuple[bool, int, str]:
        """
        Validates a raw JSON-RPC payload string.

        Args:
            payload (str): The raw JSON-RPC payload string.

        Returns:
            Tuple[bool, int, str]: (is_valid, error_code, error_message)
        """
        try:
            request = json.loads(payload)
        except json.JSONDecodeError:
            return False, -32700, "Parse error: payload is not valid JSON."

        return self.validate_request_dict(request)


class PreflightMCPServerWrapper:
    """
    Wrapper for MCPServer to intercept JSON-RPC payload executions
    and perform pre-flight validation before passing the request to the underlying server.
    """

    def __init__(self, server: MCPServer, validator: Optional[MCPPreflightValidator] = None) -> None:
        """
        Initializes the PreflightMCPServerWrapper.

        Args:
            server (MCPServer): The MCPServer instance to wrap.
            validator (Optional[MCPPreflightValidator]): The validator to use.
        """
        self.server = server
        self.validator = validator or MCPPreflightValidator(registry=getattr(server.exporter, "registry", None))

    async def handle_request(self, payload: str) -> str:
        """
        Intercepts and processes a JSON-RPC payload string with pre-flight checks.

        Args:
            payload (str): A JSON string representing the RPC request.

        Returns:
            str: A JSON string representing the RPC response.
        """
        is_valid, err_code, err_msg = self.validator.validate_payload(payload)

        # If pre-flight validation fails, directly return the JSON-RPC error
        if not is_valid:
            logger.warning(f"MCP Pre-flight intercept: code={err_code}, message={err_msg}")
            try:
                request = json.loads(payload)
                req_id = request.get("id")
            except Exception:
                req_id = None

            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": err_code,
                    "message": err_msg
                }
            })

        # Delegate to the wrapped server on successful pre-flight check
        return await self.server.handle_request(payload)
