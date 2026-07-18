import pytest
from unittest.mock import MagicMock
from magda_agent.learning.rl_stability_v1 import RLSkillWeightStabilityTracker
from magda_agent.evaluation.longitudinal_metrics import LongitudinalMetricsTracker

@pytest.fixture
def mock_metrics_tracker():
    tracker = MagicMock(spec=LongitudinalMetricsTracker)
    return tracker

def test_rl_stability_tracker_initialization(mock_metrics_tracker):
    tracker = RLSkillWeightStabilityTracker(metrics_tracker=mock_metrics_tracker)
    assert tracker.window_size == 5
    assert tracker.oscillation_threshold == 3
    assert tracker.skill_weights == {}
    assert tracker.skill_deltas == {}

def test_rl_stability_tracker_no_oscillation(mock_metrics_tracker):
    tracker = RLSkillWeightStabilityTracker(metrics_tracker=mock_metrics_tracker, window_size=5, oscillation_threshold=3)

    skill_id = "test_skill"

    # Initial weight
    is_oscillating = tracker.record_weight_update(skill_id, 1.0)
    assert not is_oscillating
    assert mock_metrics_tracker.record_metric.call_count == 0

    # Gradual increase (no direction changes)
    is_oscillating = tracker.record_weight_update(skill_id, 1.1)
    assert not is_oscillating
    mock_metrics_tracker.record_metric.assert_called_with("rl_stability_test_skill", 1.0)

    is_oscillating = tracker.record_weight_update(skill_id, 1.2)
    assert not is_oscillating

    is_oscillating = tracker.record_weight_update(skill_id, 1.3)
    assert not is_oscillating

    is_oscillating = tracker.record_weight_update(skill_id, 1.4)
    assert not is_oscillating

    # Verify the state
    assert tracker.skill_weights[skill_id] == 1.4
    assert len(tracker.skill_deltas[skill_id]) == 4
    # All deltas should be positive
    for d in tracker.skill_deltas[skill_id]:
        assert d > 0

def test_rl_stability_tracker_detects_oscillation(mock_metrics_tracker):
    tracker = RLSkillWeightStabilityTracker(metrics_tracker=mock_metrics_tracker, window_size=5, oscillation_threshold=3)

    skill_id = "oscillating_skill"

    # 0. Initial weight
    tracker.record_weight_update(skill_id, 1.0)

    # 1. Delta: +0.2
    is_oscillating = tracker.record_weight_update(skill_id, 1.2)
    assert not is_oscillating

    # 2. Delta: -0.2 (direction change 1)
    is_oscillating = tracker.record_weight_update(skill_id, 1.0)
    assert not is_oscillating

    # 3. Delta: +0.2 (direction change 2)
    is_oscillating = tracker.record_weight_update(skill_id, 1.2)
    assert not is_oscillating

    # 4. Delta: -0.2 (direction change 3) - threshold reached
    is_oscillating = tracker.record_weight_update(skill_id, 1.0)
    assert is_oscillating

    # Verify metrics logged
    mock_metrics_tracker.record_metric.assert_called_with("rl_stability_oscillating_skill", 0.0)

def test_rl_stability_tracker_sliding_window(mock_metrics_tracker):
    tracker = RLSkillWeightStabilityTracker(metrics_tracker=mock_metrics_tracker, window_size=3, oscillation_threshold=3)

    skill_id = "window_skill"

    # Insert more than window_size updates
    tracker.record_weight_update(skill_id, 1.0)
    tracker.record_weight_update(skill_id, 1.1)
    tracker.record_weight_update(skill_id, 0.9)
    tracker.record_weight_update(skill_id, 1.0)
    tracker.record_weight_update(skill_id, 0.8)

    assert len(tracker.skill_deltas[skill_id]) == 3
