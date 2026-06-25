import json
import logging
import urllib.parse
import os
import shutil
import aiohttp
import aiofiles
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SkillMarketplaceImporter:
    """
    Module for pulling down and caching skills dynamically from the agentskills.io marketplace.
    """

    def __init__(self, base_url: str = "https://registry.agentskills.io/api/v1/skills", cache_dir: str = ".skill_cache") -> None:
        """
        Initialize the Skill Marketplace Importer.

        Args:
            base_url (str): The base URL of the skill marketplace.
            cache_dir (str): The directory to use for disk caching of skills.
        """
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, skill_name: str) -> str:
        """
        Get the file path for a cached skill on disk.

        Args:
            skill_name (str): The name of the skill.

        Returns:
            str: The path to the cached JSON file.
        """
        safe_name = urllib.parse.quote_plus(skill_name)
        return os.path.join(self.cache_dir, f"{safe_name}.json")

    async def _fetch_from_network(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the skill definition from the remote marketplace.

        Args:
            skill_name (str): The name of the skill to fetch.

        Returns:
            Optional[Dict[str, Any]]: The skill definition if found, None otherwise.
        """
        safe_name = urllib.parse.quote(skill_name)
        url = f"{self.base_url}/{safe_name}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10, headers={'User-Agent': 'MagdaAgent/1.0'}) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch skill '{skill_name}' from network: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse skill '{skill_name}' JSON: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching skill '{skill_name}': {e}")
        return None

    async def get_skill(self, skill_name: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve a skill definition, utilizing memory and disk caches unless forced to refresh.

        Args:
            skill_name (str): The name of the skill to retrieve.
            force_refresh (bool): If True, bypass caches and fetch from network.

        Returns:
            Optional[Dict[str, Any]]: The unpacked skill definition, or None if not found.
        """
        if not force_refresh:
            # Check memory cache
            if skill_name in self.memory_cache:
                logger.info(f"Skill '{skill_name}' found in memory cache.")
                return self.memory_cache[skill_name]

            # Check disk cache
            cache_path = self._get_cache_path(skill_name)
            if os.path.exists(cache_path):
                try:
                    async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
                        contents = await f.read()
                        skill_data = json.loads(contents)
                        self.memory_cache[skill_name] = skill_data
                        logger.info(f"Skill '{skill_name}' loaded from disk cache.")
                        return skill_data
                except Exception as e:
                    logger.error(f"Failed to read disk cache for '{skill_name}': {e}")

        # Fetch from network
        logger.info(f"Fetching skill '{skill_name}' from marketplace...")
        skill_data = await self._fetch_from_network(skill_name)

        if skill_data:
            # Update caches
            self.memory_cache[skill_name] = skill_data
            try:
                async with aiofiles.open(self._get_cache_path(skill_name), 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(skill_data, indent=2))
            except Exception as e:
                logger.error(f"Failed to write disk cache for '{skill_name}': {e}")
            return skill_data

        return None

    async def invalidate_cache(self, skill_name: Optional[str] = None) -> None:
        """
        Invalidate the cache for a specific skill or all skills.

        Args:
            skill_name (Optional[str]): The name of the skill to invalidate. If None, clears all cache.
        """
        if skill_name:
            if skill_name in self.memory_cache:
                del self.memory_cache[skill_name]
            cache_path = self._get_cache_path(skill_name)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    logger.error(f"Failed to remove cache file for '{skill_name}': {e}")
            logger.info(f"Cache invalidated for skill '{skill_name}'.")
        else:
            self.memory_cache.clear()
            if os.path.exists(self.cache_dir):
                try:
                    shutil.rmtree(self.cache_dir)
                    os.makedirs(self.cache_dir, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to clear cache directory: {e}")
            logger.info("All skill caches invalidated.")
