"""
Canvas Live Visualization API v2

Implements API endpoints for streaming the OpenClaw Canvas Live Visualization state.
"""

import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from magda_agent.visualization.server import CanvasServer

logger = logging.getLogger(__name__)

def get_canvas_v2_router(canvas_server: CanvasServer, token: Optional[str] = None) -> APIRouter:
    """
    Returns an APIRouter providing endpoints for live canvas streaming.

    Args:
        canvas_server (CanvasServer): The initialized canvas server that broadcasts states.
        token (Optional[str]): Optional auth token for basic websocket authentication.

    Returns:
        APIRouter: The FastAPI router containing canvas V2 endpoints.
    """
    router = APIRouter(prefix="/api/v2/canvas", tags=["Canvas V2"])

    @router.websocket("/stream")
    async def stream_canvas_v2(websocket: WebSocket, auth_token: Optional[str] = None) -> None:
        """
        WebSocket endpoint for streaming canvas updates to connected clients.
        """
        if token and auth_token != token:
            await websocket.close(code=1008)
            return

        await canvas_server.connect(websocket)
        try:
            while True:
                # Keep the connection open and wait for client messages
                # CanvasServer's background task will send the state updates automatically
                _data = await websocket.receive_text()
        except WebSocketDisconnect:
            canvas_server.disconnect(websocket)
        except Exception as e:
            logger.error(f"Canvas stream error: {e}")
            canvas_server.disconnect(websocket)

    return router
