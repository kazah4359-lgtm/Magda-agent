import json
import logging
import re
from typing import Dict, Any, List, Optional

class SkillDiscoveryPipeline:
    """
    Pipeline to discover potential new skills from execution logs, user interactions,
    or other text sources. Inspired by Hermes Agent trend.
    """
    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """
        Initializes the SkillDiscoveryPipeline.

        Args:
            llm_client: The LLM client used for analysis and generation.
        """
        self.llm_client = llm_client

    async def discover_skills(self, text: str) -> List[Dict[str, str]]:
        """
        Analyzes the provided text to discover potential new skills.

        Args:
            text: The text to analyze (e.g., interaction logs, goal descriptions).

        Returns:
            A list of dictionaries, where each dictionary represents a discovered skill
            with 'name', 'description', and 'instructions'.
        """
        if not self.llm_client:
            logging.warning("No LLM client provided. Skill discovery cannot proceed.")
            return []

        prompt = (
            "Analyze the following text to identify potential new skills or actions "
            "that an autonomous agent should learn to perform better.\n"
            "Look for recurring patterns, manual repetitive tasks, or explicit user requests for new capabilities.\n"
            "Propose new skills that encapsulate these actions.\n"
            "For each proposed skill, provide:\n"
            "- 'name': A concise, snake_case name for the skill.\n"
            "- 'description': A short description of what the skill does.\n"
            "- 'instructions': Detailed natural language instructions on how the skill should be implemented or behave.\n\n"
            "Return the result as a JSON array of objects enclosed in ```json\n...\n```.\n\n"
            f"Text to analyze:\n{text}"
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.llm_client.chat_completion(messages=messages)
        except Exception as e:
            logging.error(f"Error calling LLM for skill discovery: {e}")
            return []

        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if not json_match:
            logging.error("No valid JSON array found in response for skill discovery.")
            return []

        try:
            discovered_skills = json.loads(json_match.group(1).strip())
            if not isinstance(discovered_skills, list):
                logging.error("JSON response is not a list.")
                return []

            valid_skills = []
            for skill in discovered_skills:
                if isinstance(skill, dict) and all(k in skill for k in ("name", "description", "instructions")):
                     valid_skills.append({
                         "name": str(skill["name"]),
                         "description": str(skill["description"]),
                         "instructions": str(skill["instructions"])
                     })
                else:
                    logging.warning(f"Skipping invalid skill definition: {skill}")

            return valid_skills

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from skill discovery response: {e}")
            return []
