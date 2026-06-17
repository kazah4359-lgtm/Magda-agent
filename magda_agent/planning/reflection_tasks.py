import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ValidationError

class GeneratedTask(BaseModel):
    id: str
    status: str = "todo"
    area: str
    risk: str
    title: str
    description: str
    allowed_paths: List[str]
    acceptance: List[str]

class ReflectionTaskGenerator:
    """
    Translates agent reflections into structured tasks that can be appended
    to agent_tasks.json.
    """

    def __init__(self, llm_client: Any):
        self.llm = llm_client

    async def generate_tasks_from_reflection(self, reflection_text: str, lessons: List[str], anti_patterns: List[str], max_tasks: int = 3) -> List[Dict[str, Any]]:
        """
        Takes reflection outputs and generates new actionable tasks.
        """
        prompt = f"""
        You are an AI planner. Your goal is to analyze the agent's recent reflection and generate new structured tasks to improve the agent's codebase or configuration.

        Reflection Summary:
        {reflection_text}

        Lessons Learned:
        {json.dumps(lessons)}

        Anti-patterns Detected:
        {json.dumps(anti_patterns)}

        Generate up to {max_tasks} new, concrete, actionable tasks.
        They must be focused on code improvements, bug fixes, or new features directly related to the reflections.

        Output ONLY a JSON list of objects matching this exact structure:
        [
          {{
            "id": "unique-task-id-v1",
            "status": "todo",
            "area": "planning",  // or memory, safety, agents, etc.
            "risk": "low",       // low, medium, or high
            "title": "Short descriptive title",
            "description": "Detailed explanation of what needs to be done and why, referencing the reflection.",
            "allowed_paths": ["path/to/file.py", "tests/test_file.py", "agent_tasks.json"],
            "acceptance": ["Criteria 1", "Criteria 2"]
          }}
        ]
        """

        messages = [
            {"role": "system", "content": "You output strictly valid JSON conforming to the requested schema. No markdown wrapping."},
            {"role": "user", "content": prompt}
        ]

        try:
            response_text = await self.llm.chat_completion(messages)

            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            tasks_data = json.loads(cleaned_response.strip())

            if not isinstance(tasks_data, list):
                logging.error("Task generator output is not a list.")
                return []

            valid_tasks = []
            for task_dict in tasks_data:
                try:
                    task = GeneratedTask(**task_dict)
                    valid_tasks.append(task.model_dump())
                except ValidationError as ve:
                    logging.error(f"Task validation failed: {ve}")

            return valid_tasks

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode task generation JSON: {e}")
            return []
        except Exception as e:
            logging.error(f"Error during task generation: {e}")
            return []
