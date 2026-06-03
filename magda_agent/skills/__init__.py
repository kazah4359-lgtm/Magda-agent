import logging
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.system_execute_code import execute as code_executor
from magda_agent.skills.internet_search import search_internet

def initialize_skills() -> SkillRegistry:
    registry = SkillRegistry()

    # Register Programmer Skill
    registry.register_skill(
        name="programmer",
        func=code_executor,
        description="Executes Python code in a safe sandbox. Input: 'code' string."
    )

    # Register Search Skill
    registry.register_skill(
        name="internet_search",
        func=search_internet,
        description="Searches the internet for information. Input: 'query' string."
    )

    return registry
