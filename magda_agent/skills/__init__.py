import logging
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.system_execute_code import execute as code_executor
from magda_agent.skills.internet_search import search_internet
from magda_agent.skills.omnichannel import send_message as omnichannel_send
from magda_agent.skills.codex_worker import codex_worker

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

    # Register Codex Worker Skill
    registry.register_skill(
        name="codex_worker",
        func=codex_worker,
        description="Generates a Codex-ready task prompt from the project's task manifest. This is a low side-effect prompt-only capability. Input: optional 'task_id' string."
    )

    return registry
