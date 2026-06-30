from typing import Dict, Any, List
import uuid

class AgentSkillsMCPConverter:
    """
    Converter to export legacy Magda skills (agentskills.io format) to standard MCP JSON-RPC.
    """

    @staticmethod
    def convert_to_mcp_tool(skill: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a single skill dictionary from agentskills.io format to MCP tool format.

        Args:
            skill: A dictionary representing an agentskills tool (contains 'parameters').

        Returns:
            A dictionary representing an MCP tool (contains 'inputSchema').
        """
        mcp_tool = {
            "name": skill.get("name", ""),
            "description": skill.get("description", ""),
        }

        # In agentskills, it's 'parameters'. In MCP, it's 'inputSchema'.
        if "parameters" in skill:
            mcp_tool["inputSchema"] = skill["parameters"]
        elif "inputSchema" in skill:
            # Fallback if it's already in MCP format
            mcp_tool["inputSchema"] = skill["inputSchema"]
        else:
            mcp_tool["inputSchema"] = {
                "type": "object",
                "properties": {},
                "required": []
            }

        return mcp_tool

    @staticmethod
    def convert_all(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts a list of skills from agentskills.io format to MCP tool format.

        Args:
            skills: A list of dictionaries representing agentskills tools.

        Returns:
            A list of dictionaries representing MCP tools.
        """
        return [AgentSkillsMCPConverter.convert_to_mcp_tool(skill) for skill in skills]

    @staticmethod
    def create_jsonrpc_request(method_name: str, params: Dict[str, Any], req_id: str = None) -> Dict[str, Any]:
        """
        Creates a JSON-RPC 2.0 request payload.

        Args:
            method_name: The name of the method to call.
            params: The parameters for the method call.
            req_id: An optional request ID. If None, a UUID will be generated.

        Returns:
            A JSON-RPC 2.0 request dictionary.
        """
        return {
            "jsonrpc": "2.0",
            "id": req_id or str(uuid.uuid4()),
            "method": method_name,
            "params": params
        }
