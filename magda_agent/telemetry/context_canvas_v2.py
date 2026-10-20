import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from magda_agent.memory.context_engine import ContextEngine

logger = logging.getLogger(__name__)


class ContextCanvasVisualizerV2:
    """
    Serializes Context Engine state into a live JSON payload
    for front-end rendering in OpenClaw Canvas UI V2.
    """

    def __init__(self, context_engine: Optional[ContextEngine] = None, websocket: Optional[Any] = None) -> None:
        """
        Initialize the ContextCanvasVisualizerV2.

        Args:
            context_engine (Optional[ContextEngine]): Reference to the ContextEngine instance.
            websocket (Optional[Any]): Optional WebSocket connection for streaming events.
        """
        self.context_engine = context_engine
        self.websocket = websocket
        self.logger = logging.getLogger(__name__)

    def serialize_item(self, item: Any) -> Dict[str, Any]:
        """
        Serialize an individual context memory entry or item into a dictionary.

        Args:
            item (Any): A context item, memory entry, or dictionary.

        Returns:
            Dict[str, Any]: Serialized item dictionary.
        """
        if isinstance(item, dict):
            return item

        serialized: Dict[str, Any] = {
            "content": getattr(item, "content", str(item)),
            "importance": getattr(item, "importance", getattr(item, "score", 0.5)),
            "user_id": getattr(item, "user_id", None),
            "tags": getattr(item, "tags", []),
            "timestamp": getattr(item, "timestamp", None),
            "emotional_state": getattr(item, "emotional_state", None),
        }

        if hasattr(item, "metadata") and isinstance(item.metadata, dict):
            serialized["metadata"] = item.metadata

        return serialized

    def get_formatted_state(
        self,
        context_items: Optional[List[Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract and serialize the ContextEngine state into a structured dictionary.

        Args:
            context_items (Optional[List[Any]]): Active context items to include.
            extra_metadata (Optional[Dict[str, Any]]): Additional metadata annotations.

        Returns:
            Dict[str, Any]: Structured JSON-serializable dictionary representing the ContextEngine state.
        """
        plugins_info: List[Dict[str, Any]] = []
        registered_hooks: Dict[str, int] = {}

        if self.context_engine is not None:
            # Extract plugin details
            plugins = getattr(self.context_engine, "_plugins", [])
            for plugin in plugins:
                plugins_info.append({
                    "name": plugin.__class__.__name__,
                    "module": plugin.__class__.__module__,
                })

            # Extract hook registry info
            hook_registry = getattr(self.context_engine, "hook_registry", None)
            if hook_registry and hasattr(hook_registry, "_hooks"):
                for hook_name, hooks_list in hook_registry._hooks.items():
                    registered_hooks[hook_name] = len(hooks_list)

        items_to_serialize = context_items if context_items is not None else []
        serialized_items = [self.serialize_item(item) for item in items_to_serialize]

        total_chars = sum(len(str(item.get("content", ""))) for item in serialized_items)
        estimated_tokens = total_chars // 4

        payload: Dict[str, Any] = {
            "type": "context_engine_canvas_v2",
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "data": {
                "plugins": plugins_info,
                "registered_hooks": registered_hooks,
                "items": serialized_items,
                "item_count": len(serialized_items),
                "stats": {
                    "total_characters": total_chars,
                    "estimated_tokens": estimated_tokens,
                    "plugin_count": len(plugins_info),
                },
                "metadata": extra_metadata or {},
            },
        }

        return payload

    def get_state_json(
        self,
        context_items: Optional[List[Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Returns the formatted state as a JSON string.

        Args:
            context_items (Optional[List[Any]]): Active context items.
            extra_metadata (Optional[Dict[str, Any]]): Extra metadata.

        Returns:
            str: JSON payload string.
        """
        return json.dumps(
            self.get_formatted_state(context_items=context_items, extra_metadata=extra_metadata),
            default=str,
        )

    async def broadcast_state(
        self,
        context_items: Optional[List[Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Broadcast the context engine state payload over WebSocket if connected.

        Args:
            context_items (Optional[List[Any]]): Active context items to broadcast.
            extra_metadata (Optional[Dict[str, Any]]): Extra metadata annotations.
        """
        if self.websocket is None:
            self.logger.debug("Skipping Canvas V2 broadcast: No active WebSocket connection.")
            return

        payload = self.get_formatted_state(context_items=context_items, extra_metadata=extra_metadata)
        try:
            message = json.dumps(payload, default=str)
            if hasattr(self.websocket, "send_text"):
                await self.websocket.send_text(message)
            else:
                await self.websocket.send(message)
            self.logger.debug("Successfully broadcasted Context Engine Canvas V2 state.")
        except Exception as e:
            self.logger.error(f"Failed to broadcast Context Engine Canvas V2 state: {e}")
