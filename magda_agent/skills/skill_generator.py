import re
import logging
from typing import Dict, Any, Optional

class SkillGenerator:
    """
    Synthesizes executable python scripts dynamically for new agent skills based on repeated user queries.
    Inspired by Hermes Agent trend.
    """
    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """
        Initializes the SkillGenerator with an LLM client.
        """
        self.llm_client = llm_client
        self.synthesized_skills: Dict[str, str] = {}

    async def generate_skill_from_queries(self, queries: list[str]) -> Optional[str]:
        """
        Synthesizes a python skill based on user queries, conforming to sandbox policies.
        """
        if not self.llm_client:
            raise ValueError("llm_client is required for skill synthesis.")

        combined_queries = "\n- ".join(queries)
        prompt = (
            "You are an AI assistant. The user has repeatedly asked the following queries:\n"
            f"- {combined_queries}\n\n"
            "Create a single python function named 'synthesized_skill' that accepts **kwargs and returns a string.\n"
            "This function must not use the os or sys modules to ensure sandbox policy compliance.\n"
            "Return only the python code block."
        )
        messages = [{"role": "user", "content": prompt}]

        response = await self.llm_client.chat_completion(messages=messages)

        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response

        # Basic sandbox validation
        if "import os" in code or "import sys" in code:
            logging.error("Synthesized skill violates sandbox policy.")
            return None

        self.synthesized_skills["synthesized_skill"] = code
        return code
