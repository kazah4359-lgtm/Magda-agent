import asyncio
import json
from typing import Dict, Any, List

class CanvasVisualizer:
    """
    OpenClaw trend: Canvas live visualization.
    Maintains agent state and streams updates to subscribed clients.
    """
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._subscribers: List[asyncio.Queue] = []

    def get_state(self) -> Dict[str, Any]:
        """Returns the current canvas state."""
        return dict(self._state)

    async def update_state(self, key: str, value: Any) -> None:
        """Updates a state variable and broadcasts to subscribers."""
        self._state[key] = value
        await self._broadcast({"type": "state_update", "key": key, "value": value})

    async def _broadcast(self, message: Dict[str, Any]) -> None:
        """Sends message to all active subscribers."""
        for q in self._subscribers:
            await q.put(message)

    def subscribe(self) -> asyncio.Queue:
        """Subscribes a new client and returns a queue for reading updates."""
        q = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Unsubscribes a client queue."""
        if q in self._subscribers:
            self._subscribers.remove(q)
