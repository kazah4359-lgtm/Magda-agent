import re
import json
import logging
from typing import Dict, Any, Optional

class ExperienceSkillCreator:
    """
    Creates Python skills dynamically from past execution experiences.
    Inspired by Hermes Agent trend.
    """
    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """
        Initializes the ExperienceSkillCreator.

        Args:
            llm_client: The LLM client used for code generation.
        """
        self.llm_client = llm_client
        self.created_skills: Dict[str, Dict[str, Any]] = {}

    async def generate_skill_from_experience(self, task_description: str, execution_trace: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyzes an execution trace and attempts to generate a Python skill.
        Also generates an agentskills.io compliant JSON schema for the skill.

        Args:
            task_description: A description of the task.
            execution_trace: A list of dicts representing the execution trace.

        Returns:
            A dictionary containing the generated code and schema, or None if failed.
        """
        if not self.llm_client:
            raise ValueError("llm_client is required for skill creation.")

        trace_text = ""
        for i, step in enumerate(execution_trace):
            action = step.get("action", "unknown action")
            outcome = step.get("outcome", "unknown outcome")
            trace_text += f"Step {i+1}: {action} -> {outcome}\n"

        prompt = (
            "Analyze the following execution trace of a task.\n"
            "Generate a Python function (a 'skill') that encapsulates this logic and is highly reusable.\n"
            "Return the Python code enclosed in ```python\n...\n```.\n"
            "Also return an agentskills.io compliant JSON schema enclosed in ```json\n...\n```.\n"
            "The JSON schema must have 'name', 'description', and 'parameters' keys.\n\n"
            f"Task: {task_description}\n\n"
            f"Execution Trace:\n{trace_text}"
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.llm_client.chat_completion(messages=messages)
        except Exception as e:
            logging.error(f"Failed to generate skill from experience: {e}")
            return None

        # Extract python code
        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if not code_match:
            logging.error("No valid Python code found in response.")
            return None
        code = code_match.group(1).strip()

        # Extract json schema
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if not json_match:
            logging.error("No valid JSON schema found in response.")
            return None

        try:
            schema = json.loads(json_match.group(1).strip())
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON schema: {e}")
            return None

        skill_name = schema.get("name", "generated_skill")

        skill_data = {
            "code": code,
            "schema": schema
        }

        self.created_skills[skill_name] = skill_data

        logging.info(f"Dynamically generated and stored new skill from experience: {skill_name}")
        return skill_data
