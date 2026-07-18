import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class CanvasLoggerPlugin:
    """
    A context engine plugin that intercepts lifecycle hooks
    (bootstrap, before_retrieval, after_retrieval) and logs
    the events for Canvas visualization.
    """

    def __init__(self) -> None:
        self.logs: List[Dict[str, Any]] = []

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin and log the config."""
        event = {"event": "bootstrap", "config": config}
        self.logs.append(event)
        logger.debug(f"CanvasLoggerPlugin.bootstrap: {event}")

    def before_retrieval(self, query: str, user_id: int) -> str:
        """Log the retrieval query."""
        event = {"event": "before_retrieval", "query": query, "user_id": user_id}
        self.logs.append(event)
        logger.debug(f"CanvasLoggerPlugin.before_retrieval: {event}")
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """Log the retrieved context length."""
        event = {
            "event": "after_retrieval",
            "context_length": len(context),
            "query": query,
            "user_id": user_id
        }
        self.logs.append(event)
        logger.debug(f"CanvasLoggerPlugin.after_retrieval: {event}")
        return context

    def get_logs(self) -> List[Dict[str, Any]]:
        """Return the captured logs."""
        return self.logs

    def clear_logs(self) -> None:
        """Clear the captured logs."""
        self.logs.clear()
