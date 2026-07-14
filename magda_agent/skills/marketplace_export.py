import ast
import json
import logging
import re
from typing import Dict, Any, List, Optional, Union, get_origin, get_args
from magda_agent.memory.procedural import ProceduralMemory

class HermesMarketplaceExporterV3:
    """
    Exports procedural skills learned via RL and Experience loops
    into the open agentskills.io format for marketplace publishing.
    """

    def __init__(self) -> None:
        pass

    def _extract_python_code(self, document: str) -> Optional[str]:
        """
        Extracts the Python procedure part from the stored document string.
        Uses a regular expression to find the procedure block robustly.
        Stored format typically: "Procedure Name: ...\nProcedure: def ..."
        """
        # Search for "Procedure: " followed by "def " at the start of a line or after the marker
        match = re.search(r"Procedure:\s*(def\s+.*)", document, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Basic validation that it looks like python code
            if "def " in code:
                return code

        # Fallback: if the whole document starts with def, maybe it was stored raw
        if document.strip().startswith("def "):
            return document.strip()

        return None

    def _get_type_name(self, node: Optional[ast.AST]) -> str:
        """
        Recursively resolves type names from AST nodes, handling typing module constructs.
        """
        if node is None:
            return "string"

        if isinstance(node, ast.Name):
            mapping = {
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "str": "string",
                "list": "array",
                "dict": "object",
                "Any": "string",
                "Dict": "object",
                "List": "array"
            }
            return mapping.get(node.id, "string")

        if isinstance(node, ast.Subscript):
            # Handle List[T], Dict[K, V], Optional[T], Union[T1, T2]
            if isinstance(node.value, ast.Name):
                origin = node.value.id
                if origin == "List":
                    return "array"
                if origin == "Dict":
                    return "object"
                if origin == "Optional":
                    return self._get_type_name(node.slice)
                if origin == "Union":
                    # For Union, we'll just pick the first non-None type or default to string
                    return "string"

        if isinstance(node, ast.Constant) and node.value is None:
            return "null"

        return "string"

    def _parse_code_to_schema(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Uses AST to parse Python code and extract skill definition.
        """
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    name = node.name
                    description = ast.get_docstring(node) or "No description provided."

                    properties = {}
                    required = []

                    # Logic for required: arguments without defaults
                    default_count = len(node.args.defaults)
                    arg_count = len(node.args.args)

                    for i, arg in enumerate(node.args.args):
                        arg_name = arg.arg
                        if arg_name == "self":
                            continue

                        param_type = self._get_type_name(arg.annotation)

                        prop_def = {
                            "type": param_type,
                            "description": f"Parameter {arg_name}"
                        }

                        # Handle array items if possible
                        if param_type == "array" and isinstance(arg.annotation, ast.Subscript):
                            inner_type = self._get_type_name(arg.annotation.slice)
                            prop_def["items"] = {"type": inner_type}

                        properties[arg_name] = prop_def

                        # If index is less than (arg_count - default_count), it has no default
                        if i < (arg_count - default_count):
                            required.append(arg_name)

                    return {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
            return None
        except Exception as e:
            logging.error(f"Failed to parse skill code via AST: {e}")
            return None

    def export_procedural_skills(self, memory: ProceduralMemory, user_id: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves skills from ProceduralMemory and formats them for agentskills.io.
        """
        skills_list = []
        try:
            where_clause = {}
            if user_id is not None:
                where_clause = {"user_id": user_id}

            # Retrieve all entries
            results = memory.collection.get(where=where_clause)

            if results and "documents" in results:
                for i, doc in enumerate(results["documents"]):
                    metadata = results["metadatas"][i] if "metadatas" in results else {}

                    code = self._extract_python_code(doc)

                    # If regex extraction failed, check metadata for explicit type
                    if not code and metadata.get("type") in ("hermes_experience_skill", "python_code"):
                        # Sometimes the whole document might just be the code
                        if doc.strip().startswith("def "):
                            code = doc.strip()

                    if code:
                        skill_def = self._parse_code_to_schema(code)
                        if skill_def:
                            skills_list.append(skill_def)

        except Exception as e:
            logging.error(f"Error exporting procedural skills: {e}")

        return {"skills": skills_list}

    def export_to_json(self, memory: ProceduralMemory, user_id: Optional[int] = None) -> str:
        """
        Returns the exported skills as a JSON string.
        """
        data = self.export_procedural_skills(memory, user_id)
        return json.dumps(data, indent=2)
