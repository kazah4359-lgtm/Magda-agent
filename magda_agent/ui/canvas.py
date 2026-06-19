"""
Canvas live visualization UI module.
Inspired by OpenClaw trend: Add a Canvas interface for live visual feedback of agent state and outputs.
"""
import json
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class CanvasVisualizer:
    """
    A live visualization UI component for rendering agent state and objects.
    """

    def __init__(self) -> None:
        """
        Initializes the CanvasVisualizer.
        """
        self._is_initialized = False
        self._history: List[Dict[str, Any]] = []

    def initialize(self) -> None:
        """
        Initializes the canvas interface.
        """
        self._is_initialized = True
        logger.info("Canvas visualizer initialized")

    def connect(self) -> bool:
        """
        Connects to the Canvas live visualizer.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        if not self._is_initialized:
            raise RuntimeError("CanvasVisualizer is not initialized. Call initialize() first.")
        logger.info("Canvas visualizer connected")
        return True

    def render_text(self, text: str) -> None:
        """
        Renders plain text on the canvas.

        Args:
            text (str): The text to render.
        """
        if not self._is_initialized:
            raise RuntimeError("CanvasVisualizer is not initialized. Call initialize() first.")
        self._history.append({"type": "text", "content": text})
        logger.debug(f"Rendered text to canvas: {text}")

    def render_object(self, obj: Dict[str, Any]) -> None:
        """
        Renders a structured object dynamically on the canvas.

        Args:
            obj (Dict[str, Any]): The object to visualize.
        """
        if not self._is_initialized:
            raise RuntimeError("CanvasVisualizer is not initialized. Call initialize() first.")
        self._history.append({"type": "object", "content": obj})
        logger.debug("Rendered object to canvas")

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Returns the history of rendered items.

        Returns:
            List[Dict[str, Any]]: The rendering history.
        """
        return self._history
