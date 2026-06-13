import logging
from typing import Optional, TYPE_CHECKING
from magda_agent.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from magda_agent.safety.policy import PolicyLayer
from magda_agent.skills.system_execute_code import execute as code_executor
from magda_agent.skills.internet_search import search_internet
from magda_agent.skills.omnichannel import send_message as omnichannel_send
from magda_agent.skills.names import SkillNames
from magda_agent.skills.codex_worker import codex_worker
from magda_agent.skills.mcp_kernel_executor import execute as mcp_kernel_executor
from magda_agent.skills.web_navigation import web_navigate as web_navigation_skill

def initialize_skills(policy_layer: Optional["PolicyLayer"] = None) -> SkillRegistry:
    registry = SkillRegistry(policy_layer=policy_layer)

    # Register Programmer Skill
    registry.register_skill(
        name=SkillNames.PROGRAMMER,
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
        name=SkillNames.OMNICHANNEL_SEND,
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


    from magda_agent.skills.hermes_skills import HermesSkillCreator
    from magda_agent.skills.skill_generator import SkillGenerator
    def generate_skill_sync(skill_name: str, description: str, instructions: str) -> str:
        import asyncio
        from magda_agent.llm_client import LLMClient
        client = LLMClient()
        creator = HermesSkillCreator(llm_client=client)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import threading
                result = None
                def run_in_thread():
                    nonlocal result
                    result = asyncio.run(creator.generate_skill(skill_name, description, instructions))
                t = threading.Thread(target=run_in_thread)
                t.start()
                t.join()
                return result
        except RuntimeError:
            pass
        return asyncio.run(creator.generate_skill(skill_name, description, instructions))

    def generate_skill_from_queries_sync(queries: list[str]) -> Optional[str]:
        """
        Synchronously wraps the generate_skill_from_queries coroutine.
        """
        import asyncio
        from magda_agent.llm_client import LLMClient
        client = LLMClient()
        generator = SkillGenerator(llm_client=client)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import threading
                result = None
                def run_in_thread():
                    nonlocal result
                    result = asyncio.run(generator.generate_skill_from_queries(queries))
                t = threading.Thread(target=run_in_thread)
                t.start()
                t.join()
                return result
        except RuntimeError:
            pass
        return asyncio.run(generator.generate_skill_from_queries(queries))

    registry.register_skill(
        name="hermes_skill_creator",
        func=generate_skill_sync,
        description="Generate Python code for a new agent skill based on experience. Input: 'skill_name', 'description', 'instructions' strings."
    )

    registry.register_skill(
        name="hermes_skill_generator",
        func=generate_skill_from_queries_sync,
        description="Generate Python code for a new agent skill based on repeated user queries. Input: 'queries' list of strings."
    )

    return registry

from magda_agent.skills.marketplace import fetch_and_register_skills
