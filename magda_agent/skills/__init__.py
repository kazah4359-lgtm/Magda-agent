import logging
from typing import Optional, TYPE_CHECKING
from magda_agent.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from magda_agent.safety.policy import PolicyLayer
from magda_agent.skills.system_execute_code import execute as code_executor
from magda_agent.skills.internet_search import search_internet
from magda_agent.skills.omnichannel import send_message as omnichannel_send
from magda_agent.skills.codex_worker import codex_worker
from magda_agent.skills.mcp_kernel_executor import execute as mcp_kernel_executor
from magda_agent.skills.web_navigation import web_navigate as web_navigation_skill

def initialize_skills(policy_layer: Optional["PolicyLayer"] = None) -> SkillRegistry:
    registry = SkillRegistry(policy_layer=policy_layer)

    # Register Programmer Skill
    registry.register_skill(
        name="programmer",
        func=code_executor,
        description="Executes Python code in a safe sandbox. Input: 'code' string."
    )

    # Register MCP Kernel Executor Skill
    registry.register_skill(
        name="mcp_kernel_execute",
        func=mcp_kernel_executor,
        description="Executes Python code in a strictly sandboxed MCP kernel environment with taint tracking. Input: 'code' string."
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


    # Register Web Navigation Skill
    registry.register_skill(
        name="web_navigation",
        func=web_navigation_skill,
        description="Navigates the web by loading URLs and interacting with DOM elements. Input: 'action' string ('load', 'click', 'type') and kwargs ('url', 'element_id', 'text')."
    )

    return registry

from magda_agent.skills.marketplace import fetch_and_register_skills
