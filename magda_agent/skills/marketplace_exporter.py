import json
import inspect
from typing import Dict, Any, List, Callable, get_origin
from magda_agent.skills.registry import SkillRegistry

class MarketplaceExporter:
    """
    Exports dynamically created skills from the SkillRegistry
    to the agentskills.io standard format.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        """
        Initializes the MarketplaceExporter with a given SkillRegistry.

        Args:
            registry (SkillRegistry): The registry containing the skills to export.
        """
        self.registry = registry

    def _get_json_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Extracts JSON schema parameters from the function signature.

        Args:
            func (Callable): The function to extract the schema from.

        Returns:
            Dict[str, Any]: The JSON schema representation of the function's parameters.
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

            # Basic type mapping
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

            properties[name] = {"type": param_type, "description": f"Parameter {name}"}

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def export_skills(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Exports all registered skills to a dictionary format compatible with agentskills.io.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing a 'skills' list with skill definitions.
        """
        skills_list = []
        for name, func in self.registry.skills.items():
            description = self.registry.descriptions.get(name, "No description provided.")
            schema = self._get_json_schema(func)

            skill_def = {
                "name": name,
                "description": description,
                "parameters": schema
            }
            skills_list.append(skill_def)

        return {"skills": skills_list}

    def export_skills_to_json(self) -> str:
        """
        Exports all registered skills to a JSON string formatted for agentskills.io.

        Returns:
            str: A JSON string containing the exported skills.
        """
        exported_data = self.export_skills()
        return json.dumps(exported_data, indent=2)
