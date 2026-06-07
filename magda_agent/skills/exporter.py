import json
import logging
from typing import Optional
import yaml
from magda_agent.memory.procedural import ProceduralMemory

class SkillExporter:
    """
    Utility to package and export successfully created skills from ProceduralMemory
    into a standard format (JSON/YAML) for a skill marketplace.
    """
    def __init__(self, procedural_memory: ProceduralMemory) -> None:
        self.procedural_memory = procedural_memory

    def export_skill(self, name: str, export_format: str = "json", user_id: Optional[int] = None) -> str:
        """
        Exports a skill from ProceduralMemory into JSON or YAML format.
        """
        try:
            results = self.procedural_memory.get_procedure_versions(name=name, user_id=user_id)
            if not results or not results.get("documents"):
                raise ValueError(f"Skill '{name}' not found in procedural memory.")

            # Assume we export the most recent/first match
            document = results["documents"][0]
            metadata = results["metadatas"][0] if results.get("metadatas") else {}

            if "Procedure: " not in document:
                raise ValueError("Skill document does not contain a procedure block.")

            code = document.split("Procedure: ")[-1].strip()

            export_data = {
                "metadata": metadata,
                "parameters": {},  # Future expansion
                "code": code
            }

            if export_format.lower() == "json":
                return json.dumps(export_data, indent=2)
            elif export_format.lower() == "yaml":
                return yaml.dump(export_data, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {export_format}")
        except Exception as e:
            logging.error(f"Failed to export skill '{name}': {e}")
            raise
