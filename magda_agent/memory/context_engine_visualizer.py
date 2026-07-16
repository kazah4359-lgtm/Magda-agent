import logging
import json
import time
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

class ContextEngineVisualizer:
    """
    ContextEngineVisualizer is a read-only observer plugin for the ContextEngine
    that captures context lifecycle events and exports the state matching the
    OpenClaw visualization schema.
    """

    def __init__(self, schema_version: str = "openclaw_v1") -> None:
        """
        Initializes the ContextEngineVisualizer.

        Args:
            schema_version (str): The OpenClaw schema version string.
        """
        self.schema_version: str = schema_version
        self._latest_context: Dict[int, Any] = {}
        self._callbacks: List[Callable[[Dict[str, Any]], Any]] = []
        logging.info(f"Initialized ContextEngineVisualizer with schema {schema_version}")

    def subscribe(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """
        Registers a callback that will receive formatted state updates.

        Args:
            callback (Callable): Callback function called with the formatted state dict.
        """
        self._callbacks.append(callback)
        logging.debug("New subscriber registered to ContextEngineVisualizer")

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """
        Unregisters an existing callback.

        Args:
            callback (Callable): The callback function to remove.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logging.debug("Subscriber unregistered from ContextEngineVisualizer")

    def _broadcast(self, state: Dict[str, Any]) -> None:
        """
        Broadcasts the formatted state to all registered subscribers.

        Args:
            state (Dict[str, Any]): The formatted state dictionary to broadcast.
        """
        for callback in self._callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Error executing visualizer broadcast callback: {e}")

    def format_context_state(self, user_id: int, new_context: Any, status: str = "updated") -> Dict[str, Any]:
        """
        Formats the context state into a dictionary matching the OpenClaw schema.

        Args:
            user_id (int): The ID of the user associated with this context.
            new_context (Any): The raw or list-based context entries.
            status (str): Current status of the context (e.g. 'updated', 'assembled').

        Returns:
            Dict[str, Any]: A dictionary structured in the OpenClaw context visualization format.
        """
        items: List[Dict[str, Any]] = []

        # Parse context items if it is a list or similar collection
        if isinstance(new_context, list):
            for item in new_context:
                item_dict = {
                    "content": getattr(item, "content", str(item)),
                    "importance": getattr(item, "importance", 0.5),
                    "tags": getattr(item, "tags", [])
                }
                # Handle emotional state if available in metadata/attributes
                if hasattr(item, "emotional_state") and item.emotional_state:
                    item_dict["emotional_state"] = {
                        "pleasure": getattr(item.emotional_state, "pleasure", 0.0),
                        "arousal": getattr(item.emotional_state, "arousal", 0.0),
                        "dominance": getattr(item.emotional_state, "dominance", 0.0)
                    }
                items.append(item_dict)
        elif new_context is not None:
            items.append({
                "content": getattr(new_context, "content", str(new_context)),
                "importance": getattr(new_context, "importance", 0.5),
                "tags": getattr(new_context, "tags", [])
            })

        formatted = {
            "schema_version": self.schema_version,
            "user_id": user_id,
            "status": status,
            "timestamp": time.time(),
            "items": items,
            "item_count": len(items)
        }
        return formatted

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """
        Bootstrap lifecycle hook.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        logging.info("ContextEngineVisualizer bootstrapped.")

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """
        Lifecycle hook called when the context has been updated.
        Triggers a broadcast of the newly updated context state.

        Args:
            new_context (Any): The newly updated context object or list of items.
            user_id (int): The ID of the user.
        """
        self._latest_context[user_id] = new_context
        state = self.format_context_state(user_id, new_context, status="updated")
        self._broadcast(state)

    def after_write(self, context: Any, user_id: int) -> None:
        """
        Lifecycle hook called after context is written.
        Triggers a broadcast of the written state.

        Args:
            context (Any): The context that was written.
            user_id (int): The ID of the user.
        """
        self._latest_context[user_id] = context
        state = self.format_context_state(user_id, context, status="saved")
        self._broadcast(state)

    def get_latest_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest captured context state for a user.

        Args:
            user_id (int): The user ID.

        Returns:
            Optional[Dict[str, Any]]: Latest formatted state or None.
        """
        if user_id in self._latest_context:
            return self.format_context_state(user_id, self._latest_context[user_id])
        return None

    def get_state_json(self, user_id: int) -> str:
        """
        Returns the formatted state as a JSON string.

        Args:
            user_id (int): The user ID.

        Returns:
            str: JSON encoded string of the formatted state, or empty JSON.
        """
        state = self.get_latest_state(user_id)
        if state:
            return json.dumps(state)
        return json.dumps({
            "schema_version": self.schema_version,
            "user_id": user_id,
            "status": "empty",
            "items": [],
            "item_count": 0
        })
