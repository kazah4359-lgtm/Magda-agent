import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import WebSocket

from magda_agent.consciousness.core import Consciousness

logger = logging.getLogger(__name__)

def compute_diff(old_state: str, new_state: str) -> Optional[Dict[str, Any]]:
    """
    Computes a simple dictionary diff between old_state and new_state (JSON strings).
    Returns a dict with the changed keys or None if no changes.
    """
    if old_state == new_state:
        return None

    try:
        old_dict = json.loads(old_state) if old_state else {}
        new_dict = json.loads(new_state) if new_state else {}
    except json.JSONDecodeError:
        return {"_raw": new_state}

    diff = {}
    for key, value in new_dict.items():
        if key not in old_dict or old_dict[key] != value:
            diff[key] = value

    # Note: For simplicity, we just send keys that are updated or new.
    # Deleted keys might require a more complex representation (e.g. {"key": None}).
    # We will assume dict updates are additive or overwrite.
    for key in old_dict:
        if key not in new_dict:
            diff[key] = None # Marker for deleted keys

    if not diff:
        return None

    return diff

class CanvasServerV2:
    """
    WebSocket server for streaming live visualization of Magda's internal cognitive state.
    Enhances V1 by supporting partial state diff updates.
    """
    def __init__(self, consciousness: Consciousness, interval: float = 1.0):
        # Dynamically import to avoid circular dependencies if necessary
        from magda_agent.visualization.canvas_v3 import CanvasVisualizerV3
        self.visualizer = CanvasVisualizerV3(consciousness)
        self.active_connections: List[WebSocket] = []
        self.consciousness = consciousness
        self.interval = interval
        self._streaming_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_broadcasted_state: str = ""

    async def connect(self, websocket: WebSocket):
        """Accept a websocket connection, add it, and send full current state."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Canvas client connected. Total clients: {len(self.active_connections)}")

        # On fresh connection, we should ideally send the full state.
        # But to keep logic simple across all clients, the broadcast loop sends diffs globally.
        # Here we just send the current full state to this specific client.
        try:
            current_state = self.visualizer.get_state_json()
            full_update = {"type": "full", "data": json.loads(current_state)}
            await websocket.send_text(json.dumps(full_update))
        except Exception as e:
            logger.error(f"Failed to send initial full state: {e}")

    def disconnect(self, websocket: WebSocket):
        """Remove a websocket connection from the pool."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Canvas client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Broadcast a message to all active websocket connections."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to canvas client: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def start_streaming(self):
        """Start the background task that periodically broadcasts the cognitive state."""
        self._running = True
        logger.info("Canvas visualization streaming (V2) started.")
        self._last_broadcasted_state = ""

        while self._running:
            try:
                if self.active_connections:
                    current_state = self.visualizer.get_state_json()
                    diff = compute_diff(self._last_broadcasted_state, current_state)

                    if diff:
                        update_payload = {"type": "diff", "data": diff}
                        # If there was no previous state, we might as well just send full or diff (diff is full here)
                        if not self._last_broadcasted_state:
                           update_payload["type"] = "full"

                        await self.broadcast(json.dumps(update_payload))
                        self._last_broadcasted_state = current_state

            except Exception as e:
                logger.error(f"Error while streaming canvas state V2: {e}")
            await asyncio.sleep(self.interval)

    async def stop_streaming(self):
        """Stop the background streaming task."""
        self._running = False
        logger.info("Canvas visualization streaming (V2) stopped.")
