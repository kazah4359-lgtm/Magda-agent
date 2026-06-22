from typing import Any, List, Dict
import logging

from magda_agent.memory.context_engine import ContextPlugin

class OrderedContextPluginV2(ContextPlugin):
    """
    A V2 Context Engine Plugin that explicitly tracks execution order
    and implements before_retrieval and after_retrieval lifecycle hooks.
    """
    def __init__(self) -> None:
        self.execution_order: List[str] = []
        logging.debug("Initialized OrderedContextPluginV2")

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self.execution_order.append("bootstrap")

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Process incoming content."""
        self.execution_order.append("ingest")
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble context string."""
        self.execution_order.append("assemble")
        return ""

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Compact context when limits are reached."""
        self.execution_order.append("compact")
        return context_items

    def before_retrieval(self, query: str, user_id: int) -> str:
        """
        Called before context is retrieved.
        Modifies the query to indicate V2 pre-retrieval processing.
        """
        self.execution_order.append("before_retrieval")
        return f"{query} [v2_pre_retrieval_modified]"

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """
        Called after context is retrieved.
        Appends a metadata item to the retrieved context.
        """
        self.execution_order.append("after_retrieval")
        context.append(f"metadata: v2_post_retrieval executed for user {user_id}")
        return context

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """Triggered when overall context is updated."""
        self.execution_order.append("on_context_update")
