import logging
from typing import Any, Dict, List
from magda_agent.memory.context_engine import ContextPlugin


class TelemetryPlugin(ContextPlugin):
    """
    A Context Engine Plugin that provides telemetry tracking over the context hook pipeline.
    It tracks the number of executions and the context sizes during retrieval operations.
    """

    def __init__(self) -> None:
        """Initialize the TelemetryPlugin."""
        self.metrics: Dict[str, Any] = {
            "before_retrieval_count": 0,
            "after_retrieval_count": 0,
            "total_context_items_retrieved": 0,
            "bootstrap_count": 0,
            "ingest_count": 0,
            "assemble_count": 0,
            "compact_count": 0,
            "on_context_update_count": 0
        }
        logging.info("TelemetryPlugin initialized.")

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.

        Args:
            config: A dictionary containing configuration parameters.
        """
        self.metrics["bootstrap_count"] += 1
        logging.debug(f"TelemetryPlugin bootstrap called. Metrics: {self.metrics}")

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Process incoming content before it is stored or used.

        Args:
            content: The incoming content to process.
            metadata: Metadata associated with the content.

        Returns:
            The processed content.
        """
        self.metrics["ingest_count"] += 1
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """
        Assemble the context string from retrieved items for the LLM.

        Args:
            context_items: The list of context items.
            metadata: Metadata associated with the assembly process.

        Returns:
            An empty string as this plugin doesn't modify the assembled text,
            or rely on other plugins to provide the actual string. Actually it should
            return a reasonable default if it was the only one, but ContextEngine handles that.
            Since ContextEngine overwrites, we just return the joined string to be safe.
        """
        self.metrics["assemble_count"] += 1
        return "\n".join([str(item) for item in context_items])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        Compact or summarize the context when limits are reached.

        Args:
            context_items: The list of context items.
            metadata: Metadata associated with the compaction process.

        Returns:
            The list of context items.
        """
        self.metrics["compact_count"] += 1
        return context_items

    def before_retrieval(self, query: str, user_id: int) -> str:
        """
        Called before context is retrieved.

        Args:
            query: The query string used for retrieval.
            user_id: The ID of the user performing the retrieval.

        Returns:
            The original query string.
        """
        self.metrics["before_retrieval_count"] += 1
        logging.debug(f"TelemetryPlugin before_retrieval count: {self.metrics['before_retrieval_count']}")
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """
        Called after context is retrieved.

        Args:
            context: The retrieved context items.
            query: The query string used for retrieval.
            user_id: The ID of the user performing the retrieval.

        Returns:
            The original list of context items.
        """
        self.metrics["after_retrieval_count"] += 1
        items_count = len(context)
        self.metrics["total_context_items_retrieved"] += items_count
        logging.debug(
            f"TelemetryPlugin after_retrieval. Count: {self.metrics['after_retrieval_count']}, "
            f"Items added: {items_count}, Total items: {self.metrics['total_context_items_retrieved']}"
        )
        return context

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """
        Called when the overall context is updated.

        Args:
            new_context: The updated context.
            user_id: The ID of the user.
        """
        self.metrics["on_context_update_count"] += 1
        logging.debug(f"TelemetryPlugin on_context_update count: {self.metrics['on_context_update_count']}")
