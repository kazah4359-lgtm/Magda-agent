import asyncio
import logging
from typing import List
from fastapi import WebSocket, WebSocketDisconnect
from magda_agent.consciousness.core import Consciousness

logger = logging.getLogger(__name__)

class CanvasServer:
    """
    WebSocket server for streaming live visualization of Magda's internal cognitive state.
    """
    def __init__(self, consciousness: Consciousness, interval: float = 1.0):
        from magda_agent.visualization.canvas_v3 import CanvasVisualizerV3
        self.visualizer = CanvasVisualizerV3(consciousness)
        self.active_connections: List[WebSocket] = []
        self.consciousness = consciousness
        self.interval = interval
        self._streaming_task: asyncio.Task | None = None
        self._running = False

    async def connect(self, websocket: WebSocket):
        """Accept a websocket connection and add it to the active pool."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Canvas client connected. Total clients: {len(self.active_connections)}")

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
        logger.info("Canvas visualization streaming started.")
        while self._running:
            try:
                if self.active_connections:
                    state = self.visualizer.get_state_json()
                    await self.broadcast(state)
            except Exception as e:
                logger.error(f"Error while streaming canvas state: {e}")
            await asyncio.sleep(self.interval)

    async def stop_streaming(self):
        """Stop the background streaming task."""
        self._running = False
        logger.info("Canvas visualization streaming stopped.")
        # We don't cancel the task directly here, we let the loop finish gracefully.
