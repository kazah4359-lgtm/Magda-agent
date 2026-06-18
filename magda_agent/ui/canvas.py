"""
Canvas live visualization UI module.
Inspired by OpenClaw trend: Add a Canvas interface for live visual feedback of agent state and outputs.
"""
import json
from typing import Any, Dict, List, Optional


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

    def render_text(self, text: str) -> None:
        """
        Renders plain text on the canvas.

        Args:
            text (str): The text to render.
        """
        if not self._is_initialized:
            raise RuntimeError("CanvasVisualizer is not initialized. Call initialize() first.")
        self._history.append({"type": "text", "content": text})

    def render_object(self, obj: Dict[str, Any]) -> None:
        """
        Renders a structured object dynamically on the canvas.

        Args:
            obj (Dict[str, Any]): The object to visualize.
        """
        if not self._is_initialized:
            raise RuntimeError("CanvasVisualizer is not initialized. Call initialize() first.")
        self._history.append({"type": "object", "content": obj})

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Returns the history of rendered items.

        Returns:
            List[Dict[str, Any]]: The rendering history.
        """
        return self._history
