import inspect
import logging
from typing import Dict, Any, Callable, List, get_origin
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.mcp_registry import MCPRegistry


class MCPDynamicExporterV7:
    """
    Enhances skill registry to allow dynamic exporting and registration
    of action tools over MCP standard.
    """

    def __init__(self, skill_registry: SkillRegistry, mcp_registry: MCPRegistry) -> None:
        """
        Initializes the dynamic exporter with references to both registries.

        Args:
            skill_registry (SkillRegistry): The primary registry of internal skills.
            mcp_registry (MCPRegistry): The registry for MCP-compatible tools.
        """
        self.skill_registry = skill_registry
        self.mcp_registry = mcp_registry
        self.logger = logging.getLogger(__name__)

    def _get_json_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Extracts JSON schema parameters from a given function's signature.

        Args:
            func (Callable): The function to extract the schema from.

        Returns:
            Dict[str, Any]: The extracted JSON schema.
        """
        if hasattr(func, "__mcp_schema__"):
            return getattr(func, "__mcp_schema__")

        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name == 'self':
                continue

            param_type = "string"
            annotation = param.annotation
            origin = get_origin(annotation)

            actual_type = origin if origin is not None else annotation

            if actual_type is int:
                param_type = "integer"
            elif actual_type is float:
                param_type = "number"
            elif actual_type is bool:
                param_type = "boolean"
            elif actual_type in (list, List):
                param_type = "array"
            elif actual_type in (dict, Dict):
                param_type = "object"

            properties[name] = {"type": param_type}

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def export_skill_to_mcp(self, skill_name: str) -> bool:
        """
        Exports a specific skill from the SkillRegistry to the MCPRegistry.

        Args:
            skill_name (str): The name of the skill to export.

        Returns:
            bool: True if exported successfully, False otherwise.
        """
        if not self.skill_registry.has_skill(skill_name):
            self.logger.warning(f"Skill '{skill_name}' not found in SkillRegistry.")
            return False

        func = self.skill_registry.skills[skill_name]
        description = self.skill_registry.descriptions.get(skill_name, "")
        schema = self._get_json_schema(func)

        mcp_tool_schema = {
            "name": skill_name,
            "description": description,
            "inputSchema": schema
        }

        success = self.mcp_registry.load_tool(mcp_tool_schema)
        if success:
            self.logger.info(f"Successfully exported skill '{skill_name}' to MCPRegistry.")
        else:
            self.logger.error(f"Failed to export skill '{skill_name}' to MCPRegistry.")

        return success

    def export_all_skills(self) -> int:
        """
        Exports all registered skills from SkillRegistry to MCPRegistry dynamically.

        Returns:
            int: The number of skills successfully exported.
        """
        exported_count = 0
        for name in list(self.skill_registry.skills.keys()):
            if self.export_skill_to_mcp(name):
                exported_count += 1

        self.logger.info(f"Exported {exported_count} skills to MCPRegistry.")
        return exported_count
