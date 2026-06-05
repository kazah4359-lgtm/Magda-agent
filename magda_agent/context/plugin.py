from typing import Any, Dict, List, Optional

class ContextPlugin:
    """
    Base class for Context Engine plugins.
    Inspired by OpenClaw's architecture.
    """
    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Process incoming content before it is stored or used."""
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble the context string from retrieved items for the LLM."""
        return "\n".join([str(item) for item in context_items])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Compact or summarize the context when limits are reached."""
        return context_items
