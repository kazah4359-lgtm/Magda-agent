from typing import Any, List, Dict
import logging

from magda_agent.memory.context_engine import ContextPlugin

class OrderedContextPluginV5(ContextPlugin):
    """
    A V5 Context Engine Plugin that explicitly tracks execution order
    and implements before_retrieval and after_retrieval lifecycle hooks.
    Inspired by OpenClaw trends for memory context hooks.
    """
    def __init__(self) -> None:
        """Initialize the plugin and set up execution order tracking."""
        super().__init__()
        self.execution_order: List[str] = []
        logging.debug("Initialized OrderedContextPluginV5")

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.
        """
        self.execution_order.append("bootstrap")

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Process incoming content before it is stored or used.
        """
        self.execution_order.append("ingest")
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """
        Assemble the context string from retrieved items for the LLM.
        """
        self.execution_order.append("assemble")
        return ""

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        Compact or summarize the context when limits are reached.
        """
        self.execution_order.append("compact")
        return context_items

    def before_retrieval(self, query: str, user_id: int) -> str:
        """
        Called before context is retrieved.
        Modifies the query to indicate V5 pre-retrieval processing.
        """
        self.execution_order.append("before_retrieval")
        return f"{query} [v5_pre_retrieval_modified]"

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """
        Called after context is retrieved.
        Appends a metadata item to the retrieved context.
        """
        self.execution_order.append("after_retrieval")
        context.append(f"metadata: v5_post_retrieval executed for user {user_id}")
        return context

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """
        Triggered when overall context is updated.
        """
        self.execution_order.append("on_context_update")
