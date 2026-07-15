import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import WebSocket

from magda_agent.consciousness.core import Consciousness
from magda_agent.visualization.canvas_v3 import CanvasVisualizerV3

logger = logging.getLogger(__name__)

class LiveCanvasStreamer:
    """
    LiveCanvasStreamer manages WebSocket streaming of Magda's internal cognitive state,
    providing realtime visualizations of memory layers and attention focus.
    """
    def __init__(self, consciousness: Consciousness, interval: float = 1.0) -> None:
        """
        Initializes the LiveCanvasStreamer with consciousness and interval.

        Args:
            consciousness: The main Consciousness instance.
            interval: The periodic interval (in seconds) at which state is streamed.
        """
        self.consciousness: Consciousness = consciousness
        self.interval: float = interval
        self.visualizer: CanvasVisualizerV3 = CanvasVisualizerV3(consciousness)
        self.active_connections: List[WebSocket] = []
        self._running: bool = False
        self._streaming_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket) -> None:
        """
        Accepts an incoming WebSocket connection and adds it to the active pool,
        sending the initial full state immediately.

        Args:
            websocket: The FastAPI WebSocket connection.
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Canvas client connected. Active connections: {len(self.active_connections)}")
        try:
            current_state = self.visualizer.get_state_json()
            await websocket.send_text(current_state)
        except Exception as e:
            logger.error(f"Failed to send initial state to canvas client: {e}")

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Removes a WebSocket connection from the active pool.

        Args:
            websocket: The WebSocket connection to remove.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Canvas client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: str) -> None:
        """
        Broadcasts a text message to all active WebSocket connections.

        Args:
            message: The message string to broadcast.
        """
        failed_connections: List[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to canvas client: {e}")
                failed_connections.append(connection)

        for conn in failed_connections:
            self.disconnect(conn)

    async def start_streaming(self) -> None:
        """
        Starts the background loop that periodically broadcasts the cognitive state.
        """
        self._running = True
        logger.info("Live Canvas streaming started.")
        while self._running:
            try:
                if self.active_connections:
                    state_json = self.visualizer.get_state_json()
                    await self.broadcast(state_json)
            except Exception as e:
                logger.error(f"Error in Live Canvas streaming loop: {e}")
            await asyncio.sleep(self.interval)

    async def stop_streaming(self) -> None:
        """
        Stops the periodic streaming of cognitive state.
        """
        self._running = False
        logger.info("Live Canvas streaming stopped.")
