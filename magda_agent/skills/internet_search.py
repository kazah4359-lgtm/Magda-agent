import logging
from typing import List, Dict, Any

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

logger = logging.getLogger(__name__)

def search_internet(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the internet using DuckDuckGo.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).

    Returns:
        A list of dictionaries containing 'title', 'href', and 'body' for each search result.
    """
    if DDGS is None:
        logger.error("duckduckgo-search package is not installed. Please install it using `pip install duckduckgo-search`.")
        return []

    try:
        results = list(DDGS().text(query, max_results=max_results))
        return results
    except Exception as e:
        logger.error(f"Error during internet search: {e}")
        return []
