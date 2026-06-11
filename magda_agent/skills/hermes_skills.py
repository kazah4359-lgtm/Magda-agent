import re
from typing import Dict, Any, Optional

class HermesSkillCreator:
    """
    Creates skills dynamically from agent experience using the LLM client.
    """
    def __init__(self, llm_client=None):
        self.created_skills: Dict[str, str] = {}
        # In actual usage this would be injected. If None, it will fail when trying to generate.
        self.llm_client = llm_client

    async def generate_skill(self, name: str, description: str, instructions: str) -> str:
        """
        Generates code for a new skill and registers it locally.

        Args:
            name: Name of the skill.
            description: Description of the skill.
            instructions: Instructions for the skill.

        Returns:
            The generated code.
        """
        if not self.llm_client:
            raise ValueError("llm_client is required for skill generation.")

        prompt = (
            f"You are a helpful assistant. Write a python function named '{name}'.\n"
            f"It must accept **kwargs and return a string.\n"
            f"Description: {description}\n"
            f"Instructions: {instructions}\n"
            "Return only the python code block."
        )
        messages = [{"role": "user", "content": prompt}]

        response = await self.llm_client.chat_completion(messages=messages)

        # Extract code from response
        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response # fallback

        self.created_skills[name] = code
        return code

    def load_skill_dynamically(self, registry: Any, name: str) -> bool:
        """
        Loads a generated skill into the provided SkillRegistry.

        Args:
            registry: The SkillRegistry instance.
            name: The name of the skill to load.

        Returns:
            True if successful, False otherwise.
        """
        if name not in self.created_skills:
            return False

        code = self.created_skills[name]

        # This is inherently risky, but we are asked to dynamically load python code.
        # It's an experimental feature for "skills created from experience".
        local_env = {}
        try:
            exec(code, {}, local_env)
            if name in local_env and callable(local_env[name]):
                # Assume registry has a register_skill method
                if hasattr(registry, "register_skill"):
                    registry.register_skill(name, local_env[name], "Dynamically generated skill")
                    return True
            return False
        except Exception:
            return False
