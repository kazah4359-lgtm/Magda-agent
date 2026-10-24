import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional


class CanvasReflectionTelemetryV2:
    """
    An export module to stream V2 subconsciousness reflection events
    dynamically to a live dashboard using WebSockets, inspired by OpenClaw Canvas.
    """

    def __init__(self, websocket: Optional[Any] = None) -> None:
        """
        Initialize the CanvasReflectionTelemetryV2.

        Args:
            websocket (Optional[Any]): An open asynchronous websocket connection.
        """
        self.websocket = websocket
        self.logger = logging.getLogger(__name__)

    async def broadcast_reflection_event(
        self,
        summary: str,
        lessons: Optional[List[str]] = None,
        anti_patterns: Optional[List[str]] = None,
        proposed_tasks: Optional[List[str]] = None,
        new_facts: Optional[List[str]] = None,
        pad_adjustment: Optional[Dict[str, float]] = None,
        reflection_id: Optional[str] = None,
    ) -> None:
        """
        Broadcast a subconsciousness reflection V2 event to the connected Canvas client.

        Args:
            summary (str): Textual summary of the self-reflection.
            lessons (Optional[List[str]]): List of reusable lessons identified.
            anti_patterns (Optional[List[str]]): List of anti-patterns identified.
            proposed_tasks (Optional[List[str]]): List of proposed tasks generated.
            new_facts (Optional[List[str]]): List of new semantic facts extracted.
            pad_adjustment (Optional[Dict[str, float]]): Emotional PAD state adjustment.
            reflection_id (Optional[str]): Unique identifier for the reflection event.
        """
        payload = {
            "type": "subconsciousness_reflection_v2",
            "version": "v2",
            "timestamp": time.time(),
            "data": {
                "reflection_id": reflection_id or str(uuid.uuid4()),
                "summary": summary,
                "lessons": lessons or [],
                "anti_patterns": anti_patterns or [],
                "proposed_tasks": proposed_tasks or [],
                "new_facts": new_facts or [],
                "pad_adjustment": pad_adjustment or {"p": 0.0, "a": 0.0, "d": 0.0},
            },
        }
        await self._broadcast(payload)

    async def broadcast_reflection_v2(
        self,
        summary: str,
        lessons: Optional[List[str]] = None,
        anti_patterns: Optional[List[str]] = None,
        proposed_tasks: Optional[List[str]] = None,
        new_facts: Optional[List[str]] = None,
        pad_adjustment: Optional[Dict[str, float]] = None,
        reflection_id: Optional[str] = None,
    ) -> None:
        """
        Alias for broadcast_reflection_event for convenience.
        """
        await self.broadcast_reflection_event(
            summary=summary,
            lessons=lessons,
            anti_patterns=anti_patterns,
            proposed_tasks=proposed_tasks,
            new_facts=new_facts,
            pad_adjustment=pad_adjustment,
            reflection_id=reflection_id,
        )

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        """
        Helper method to format and send JSON payload over the websocket.

        Args:
            payload (Dict[str, Any]): The payload of the event.
        """
        if self.websocket is None:
            self.logger.debug(
                f"Skipping broadcast for {payload.get('type')} - no websocket connected."
            )
            return

        try:
            message = json.dumps(payload)
            if hasattr(self.websocket, "send_text"):
                await self.websocket.send_text(message)
            else:
                await self.websocket.send(message)
            self.logger.debug(f"Successfully broadcasted {payload.get('type')} event.")
        except Exception as e:
            self.logger.error(f"Failed to broadcast {payload.get('type')} event: {e}")
