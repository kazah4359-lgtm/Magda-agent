import pytest

from magda_agent.ui.canvas import CanvasVisualizer

def test_canvas_initialization() -> None:
    """
    Test that the CanvasVisualizer initializes correctly.
    """
    canvas = CanvasVisualizer()
    assert canvas._is_initialized is False
    canvas.initialize()
    assert canvas._is_initialized is True

def test_canvas_render_uninitialized() -> None:
    """
    Test that rendering fails before initialization.
    """
    canvas = CanvasVisualizer()
    with pytest.raises(RuntimeError, match="CanvasVisualizer is not initialized. Call initialize.. first."):
        canvas.render_text("Hello")

    with pytest.raises(RuntimeError, match="CanvasVisualizer is not initialized. Call initialize.. first."):
        canvas.render_object({"hello": "world"})

def test_canvas_render_text() -> None:
    """
    Test rendering text on the canvas.
    """
    canvas = CanvasVisualizer()
    canvas.initialize()
    canvas.render_text("Test rendering")

    history = canvas.get_history()
    assert len(history) == 1
    assert history[0]["type"] == "text"
    assert history[0]["content"] == "Test rendering"

def test_canvas_render_object() -> None:
    """
    Test rendering an object on the canvas.
    """
    canvas = CanvasVisualizer()
    canvas.initialize()

    obj = {"key": "value", "nested": {"id": 123}}
    canvas.render_object(obj)

    history = canvas.get_history()
    assert len(history) == 1
    assert history[0]["type"] == "object"
    assert history[0]["content"] == obj
