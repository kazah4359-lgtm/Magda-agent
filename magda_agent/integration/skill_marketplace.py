from typing import Dict, Any, List, Callable
import inspect
from magda_agent.skills.registry import SkillRegistry

class AgentSkillsExporter:
    """
    Exports Magda skills as agentskills.io compatible JSON objects.
    """
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def _get_json_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Extracts JSON schema parameters from the function signature.
        """
        if hasattr(func, "__mcp_schema__"):
            return getattr(func, "__mcp_schema__")

        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name in ('self', 'kwargs', 'args'):
                continue

            param_type = "string"
            if param.annotation is not inspect.Parameter.empty:
                if param.annotation is int:
                    param_type = "integer"
                elif param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"
                elif param.annotation is list or param.annotation is List:
                    param_type = "array"
                elif param.annotation is dict or param.annotation is Dict:
                    param_type = "object"

            properties[name] = {"type": param_type}

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def export_skills(self) -> List[Dict[str, Any]]:
        """
        Exports all registered skills to an agentskills.io compatible format.
        """
        skills = []
        for name, func in self.registry.skills.items():
            description = self.registry.descriptions.get(name, "")
            schema = self._get_json_schema(func)

            skills.append({
                "name": name,
                "description": description,
                "parameters": schema
            })

        return skills
