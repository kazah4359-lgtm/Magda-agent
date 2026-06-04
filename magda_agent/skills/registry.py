from typing import Dict, Callable, Any, Optional
import logging

class SkillRegistry:
    """
    Registry to manage and trigger available skills for the AGI agent.
    """
    def __init__(self):
        self.skills: Dict[str, Callable] = {}
        self.descriptions: Dict[str, str] = {}

    def register_skill(self, name: str, func: Callable, description: str):
        self.skills[name] = func
        self.descriptions[name] = description
        logging.info(f"Skill registered: {name}")

    def has_skill(self, name: str) -> bool:
        """
        Checks whether a skill with the given name is registered.

        Args:
            name (str): The name of the skill to check.

        Returns:
            bool: True if the skill exists, False otherwise.
        """
        return name in self.skills

    def execute_skill(self, name: str, **kwargs) -> Any:
        if name not in self.skills:
            return f"Error: Skill '{name}' not found."
        try:
            return self.skills[name](**kwargs)
        except Exception as e:
            logging.error(f"Error executing skill {name}: {e}")
            return f"Error executing skill {name}: {e}"

    def get_skills_summary(self) -> str:
        summary = "Available Skills:\n"
        for name, desc in self.descriptions.items():
            summary += f"- {name}: {desc}\n"
        return summary
