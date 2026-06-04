import logging
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.system_execute_code import execute as code_executor
from magda_agent.skills.internet_search import search_internet
from magda_agent.skills.omnichannel import send_message as omnichannel_send

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

    # Register Omnichannel Skill
    registry.register_skill(
        name="omnichannel_send",
        func=omnichannel_send,
        description="Sends a message to a recipient on a specified platform (telegram, whatsapp, email). Input: 'platform', 'recipient', 'message' strings."
    )

    return registry
