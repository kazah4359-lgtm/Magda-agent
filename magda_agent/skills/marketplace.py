import logging
from typing import Dict, Any, List
from magda_agent.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

def _create_dynamic_skill(skill_def: Dict[str, Any]):
    """
    Creates a dynamic skill function based on the agentskills.io definition.
    For this mock implementation, it simply logs the execution and returns a success message.
    In a real implementation, this might call an external API.
    """
    def dynamic_skill(**kwargs):
        logger.info(f"Executing marketplace skill: {skill_def.get('name')} with args: {kwargs}")
        return f"Executed remote skill '{skill_def.get('name')}' successfully."

    # Optionally attach metadata to the function
    dynamic_skill.__name__ = skill_def.get("name", "unknown_skill")
    dynamic_skill.__doc__ = skill_def.get("description", "No description provided.")

    return dynamic_skill

async def fetch_and_register_skills(url: str, registry: SkillRegistry) -> List[str]:
    """
    Fetches an agentskills.io compliant JSON specification from the given URL
    and dynamically registers the skills into the provided SkillRegistry.

    Args:
        url (str): The URL of the marketplace JSON endpoint.
        registry (SkillRegistry): The registry where new skills will be registered.

    Returns:
        List[str]: A list of skill names that were successfully registered.
    """
    registered_skills = []
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()

                # Assume standard agentskills.io definition has a "skills" list
                skills_list = data.get("skills", [])
                for skill_def in skills_list:
                    name = skill_def.get("name")
                    description = skill_def.get("description", "Dynamic marketplace skill")

                    if name:
                        func = _create_dynamic_skill(skill_def)
                        registry.register_skill(name=name, func=func, description=description)
                        registered_skills.append(name)
                        logger.info(f"Successfully registered marketplace skill: {name}")
                    else:
                        logger.warning(f"Skill definition missing 'name' field: {skill_def}")

    except Exception as e:
        logger.error(f"Failed to fetch and register skills from {url}: {e}")

    return registered_skills
