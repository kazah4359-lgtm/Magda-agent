import logging
import os
import yaml
import importlib.util
from typing import Dict, Any, List, Optional
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.marketplace_importer import SkillMarketplaceImporter

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
    name = skill_def.get("name", "unknown_skill")
    dynamic_skill.__name__ = name
    dynamic_skill.__doc__ = skill_def.get("description", "No description provided.")

    # Attach parameters schema if present
    if "parameters" in skill_def:
        setattr(dynamic_skill, "__mcp_schema__", skill_def["parameters"])

    return dynamic_skill

def load_skill_from_directory(skill_dir: str, registry: SkillRegistry) -> Optional[str]:
    """
    Loads a skill from a directory following the agentskills.io specification.
    The directory must contain a SKILL.md file with YAML frontmatter.

    Args:
        skill_dir: Path to the skill directory.
        registry: The SkillRegistry to register the skill in.

    Returns:
        The name of the registered skill if successful, None otherwise.
    """
    skill_md_path = os.path.join(skill_dir, "SKILL.md")
    if not os.path.exists(skill_md_path):
        logger.error(f"SKILL.md not found in {skill_dir}")
        return None

    try:
        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                metadata = yaml.safe_load(parts[1])
                instructions = parts[2].strip()
            else:
                logger.error(f"Invalid SKILL.md format in {skill_dir}")
                return None
        else:
            logger.error(f"No frontmatter found in SKILL.md in {skill_dir}")
            return None

        name = metadata.get("name")
        description = metadata.get("description", "")
        if not name:
            logger.error(f"Skill name missing in frontmatter in {skill_dir}")
            return None

        # Look for scripts in scripts/ directory
        scripts_dir = os.path.join(skill_dir, "scripts")
        skill_func = None
        if os.path.isdir(scripts_dir):
            for filename in os.listdir(scripts_dir):
                if filename.endswith(".py"):
                    script_path = os.path.join(scripts_dir, filename)
                    spec = importlib.util.spec_from_file_location(name, script_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        # Try to find a function matching the skill name
                        if hasattr(module, name) and callable(getattr(module, name)):
                            skill_func = getattr(module, name)
                            break
                        # Otherwise take the first callable found
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if callable(attr) and not attr_name.startswith("__"):
                                skill_func = attr
                                break
                    if skill_func:
                        break

        if not skill_func:
            # If no script found, create a placeholder that returns the instructions
            def placeholder_skill(**kwargs):
                return f"Skill '{name}' instructions: {instructions}"
            placeholder_skill.__name__ = name
            placeholder_skill.__doc__ = description
            skill_func = placeholder_skill

        # Attach parameters schema if present in metadata
        if "parameters" in metadata:
            setattr(skill_func, "__mcp_schema__", metadata["parameters"])

        registry.register_skill(name=name, func=skill_func, description=description)
        logger.info(f"Loaded skill '{name}' from directory {skill_dir}")
        return name

    except Exception as e:
        logger.error(f"Error loading skill from {skill_dir}: {e}")
        return None

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

async def load_skill_from_marketplace(skill_name: str, registry: SkillRegistry, importer: Optional[SkillMarketplaceImporter] = None) -> bool:
    """
    Loads a single skill from the agentskills.io marketplace using the SkillMarketplaceImporter v2.

    Args:
        skill_name: The name of the skill to fetch.
        registry: The SkillRegistry to register the skill in.
        importer: Optional SkillMarketplaceImporter instance. If None, one is created.

    Returns:
        True if the skill was successfully fetched and registered, False otherwise.
    """
    if importer is None:
        importer = SkillMarketplaceImporter()

    skill_def = await importer.get_skill(skill_name)
    if not skill_def:
        logger.error(f"Could not load skill '{skill_name}' from marketplace.")
        return False

    name = skill_def.get("name")
    if not name:
        logger.error(f"Marketplace skill definition missing 'name' field.")
        return False

    description = skill_def.get("description", "Dynamic marketplace skill via v2 importer")
    func = _create_dynamic_skill(skill_def)
    registry.register_skill(name=name, func=func, description=description)
    logger.info(f"Successfully registered marketplace skill via importer v2: {name}")
    return True

async def search_marketplace_skills(url: str, query: str) -> List[Dict[str, Any]]:
    """
    Searches an agentskills.io compliant marketplace for skills matching the given query.

    Args:
        url (str): The URL of the marketplace JSON endpoint.
        query (str): The search query to filter skills by name or description.

    Returns:
        List[Dict[str, Any]]: A list of skill definitions matching the query.
    """
    matched_skills = []
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()

                skills_list = data.get("skills", [])
                query_lower = query.lower()

                for skill_def in skills_list:
                    name = skill_def.get("name", "").lower()
                    description = skill_def.get("description", "").lower()

                    if query_lower in name or query_lower in description:
                        matched_skills.append(skill_def)

    except Exception as e:
        logger.error(f"Failed to search skills from {url}: {e}")

    return matched_skills
