import pytest
import time
from magda_agent.homeostasis.hypothalamus import Hypothalamus

def test_hypothalamus_initialization():
    hypothalamus = Hypothalamus()
    assert hypothalamus.needs["social"] == 0.5
    assert hypothalamus.needs["rest"] == 0.0
    assert hypothalamus.needs["curiosity"] == 0.5

def test_hypothalamus_update_needs():
    hypothalamus = Hypothalamus()

    # Test positive deltas
    hypothalamus.update_needs(delta_social=0.2, delta_rest=0.1, delta_curiosity=0.3)
    assert hypothalamus.needs["social"] == pytest.approx(0.7, abs=0.01)
    assert hypothalamus.needs["rest"] == pytest.approx(0.1, abs=0.01)
    assert hypothalamus.needs["curiosity"] == pytest.approx(0.8, abs=0.01)

    # Test negative deltas
    hypothalamus.update_needs(delta_social=-0.5, delta_rest=-0.1, delta_curiosity=-0.9)
    assert hypothalamus.needs["social"] == pytest.approx(0.2, abs=0.01)
    assert hypothalamus.needs["rest"] == pytest.approx(0.0, abs=0.01)
    assert hypothalamus.needs["curiosity"] == pytest.approx(0.0, abs=0.01)

def test_hypothalamus_time_decay(monkeypatch):
    hypothalamus = Hypothalamus()

    # Mock time.time() to simulate passage of 1 hour (3600 seconds)
    current_time = time.time()

    def mock_time():
        return current_time + 3600

    monkeypatch.setattr(time, 'time', mock_time)

    hypothalamus.update_needs()
    # Base: social=0.5, rest=0.0, curiosity=0.5
    # Max time_factor = 1.0 (3600/3600)
    # Social + 0.1 = 0.6
    # Rest + 0.05 = 0.05
    # Curiosity + 0.05 = 0.55

    assert hypothalamus.needs["social"] == pytest.approx(0.6, abs=0.01)
    assert hypothalamus.needs["rest"] == pytest.approx(0.05, abs=0.01)
    assert hypothalamus.needs["curiosity"] == pytest.approx(0.55, abs=0.01)

def test_hypothalamus_bounds():
    hypothalamus = Hypothalamus()
    # Try to push beyond 1.0
    hypothalamus.update_needs(delta_social=1.5, delta_rest=2.0, delta_curiosity=1.0)
    assert hypothalamus.needs["social"] == 1.0
    assert hypothalamus.needs["rest"] == 1.0
    assert hypothalamus.needs["curiosity"] == 1.0

    # Try to push below 0.0
    hypothalamus.update_needs(delta_social=-2.0, delta_rest=-2.0, delta_curiosity=-2.0)
    assert hypothalamus.needs["social"] == 0.0
    assert hypothalamus.needs["rest"] == 0.0
    assert hypothalamus.needs["curiosity"] == 0.0

def test_hypothalamus_summary():
    hypothalamus = Hypothalamus()
    summary = hypothalamus.get_summary()
    assert "Needs State:" in summary
    assert "Social: 0.50" in summary
    assert "Rest: 0.00" in summary
    assert "Curiosity: 0.50" in summary
