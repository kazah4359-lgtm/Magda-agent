import pytest
from unittest.mock import MagicMock
from magda_agent.learning.metrics import RLMetricsSystem
from magda_agent.metacognition.tracker import QualityTracker

@pytest.fixture
def mock_quality_tracker():
    """Provides a mocked QualityTracker."""
    return MagicMock(spec=QualityTracker)

@pytest.fixture
def rl_metrics_system(mock_quality_tracker):
    """Provides an RLMetricsSystem instance with a mocked tracker."""
    return RLMetricsSystem(quality_tracker=mock_quality_tracker)

def test_capture_next_state_signal_empty(rl_metrics_system, mock_quality_tracker):
    """Test that empty replies are ignored."""
    rl_metrics_system.capture_next_state_signal("", "context", "user_1")
    mock_quality_tracker.log_metric.assert_not_called()

def test_capture_next_state_signal_explicit_score(rl_metrics_system, mock_quality_tracker):
    """Test capturing an explicit score."""
    reply = "I give this a 8.5/10, good job."
    rl_metrics_system.capture_next_state_signal(reply, "test_action", "user_1")

    # Check explicit score log
    mock_quality_tracker.log_metric.assert_any_call(
        "rl_explicit_score",
        8.5,
        {"user_id": "user_1", "action_context": "test_action", "is_correction": False}
    )

    # Check quality score log (8.5 * 10 = 85.0)
    mock_quality_tracker.log_metric.assert_any_call(
        "rl_quality_score",
        85.0,
        {"user_id": "user_1", "action_context": "test_action", "is_correction": False}
    )

def test_capture_next_state_signal_positive_implicit(rl_metrics_system, mock_quality_tracker):
    """Test capturing positive implicit sentiment."""
    reply = "That is perfect, thanks!"
    rl_metrics_system.capture_next_state_signal(reply, "test_action", "user_2")

    # Extract args of the log_metric calls
    calls = mock_quality_tracker.log_metric.call_args_list

    # Should have logged rl_implicit_sentiment and rl_quality_score
    assert len(calls) == 2

    sentiment_call = calls[0]
    assert sentiment_call[0][0] == "rl_implicit_sentiment"
    assert sentiment_call[0][1] > 0.0  # sentiment should be positive

    quality_call = calls[1]
    assert quality_call[0][0] == "rl_quality_score"
    assert quality_call[0][1] > 50.0  # quality should be > 50

def test_capture_next_state_signal_correction(rl_metrics_system, mock_quality_tracker):
    """Test capturing a negative correction."""
    reply = "No, that is wrong, fix it."
    rl_metrics_system.capture_next_state_signal(reply, "test_action", "user_3")

    # Check for correction and sentiment logs
    calls = mock_quality_tracker.log_metric.call_args_list
    assert len(calls) == 2

    sentiment_call = calls[0]
    assert sentiment_call[0][0] == "rl_implicit_sentiment"
    assert sentiment_call[0][1] < 0.0  # sentiment should be negative

    quality_call = calls[1]
    assert quality_call[0][0] == "rl_quality_score"
    # Quality should be significantly lower (starts at 50, -sentiment penalty, -correction penalty)
    assert quality_call[0][1] < 50.0

def test_capture_next_state_signal_with_tool_output(rl_metrics_system, mock_quality_tracker):
    """Test with tool output included."""
    reply = "perfect"
    tool_out = "Some expected result"
    rl_metrics_system.capture_next_state_signal(reply, "test_action", "user_4", tool_output=tool_out)

    calls = mock_quality_tracker.log_metric.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == "rl_implicit_sentiment"
    assert calls[1][0][0] == "rl_quality_score"

    # Verify metadata is correct
    assert calls[0][0][2]["user_id"] == "user_4"
