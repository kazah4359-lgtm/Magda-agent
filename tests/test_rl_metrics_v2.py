import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.learning.metrics_v2 import RLMetricsSystemV2
from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_quality_tracker():
    """Provides a mocked QualityTracker."""
    return MagicMock(spec=QualityTracker)

@pytest.fixture
def mock_llm_client():
    """Provides a mocked LLMClient."""
    llm = MagicMock(spec=LLMClient)
    llm.generate = AsyncMock()
    return llm

@pytest.fixture
def rl_metrics_system_v2(mock_quality_tracker, mock_llm_client):
    """Provides an RLMetricsSystemV2 instance with mocked dependencies."""
    return RLMetricsSystemV2(quality_tracker=mock_quality_tracker, llm_client=mock_llm_client)

@pytest.mark.asyncio
async def test_capture_next_state_signal_empty(rl_metrics_system_v2, mock_quality_tracker):
    """Test that empty replies are ignored."""
    await rl_metrics_system_v2.capture_next_state_signal("", "context", "user_1")
    mock_quality_tracker.log_metric.assert_not_called()

@pytest.mark.asyncio
async def test_capture_next_state_signal_explicit_score(rl_metrics_system_v2, mock_quality_tracker):
    """Test capturing an explicit score."""
    reply = "I give this a 8.5/10, good job."
    await rl_metrics_system_v2.capture_next_state_signal(reply, "test_action", "user_1")

    # Check explicit score log
    mock_quality_tracker.log_metric.assert_any_call(
        "rl_explicit_score_v2",
        8.5,
        {"user_id": "user_1", "action_context": "test_action", "is_correction": False, "intent": "praise"}
    )

    # Check quality score log (8.5 * 10 = 85.0)
    mock_quality_tracker.log_metric.assert_any_call(
        "rl_quality_score_v2",
        85.0,
        {"user_id": "user_1", "action_context": "test_action", "is_correction": False, "intent": "praise"}
    )

@pytest.mark.asyncio
async def test_capture_next_state_signal_with_llm_fallback(rl_metrics_system_v2, mock_quality_tracker, mock_llm_client):
    """Test falling back to the LLM for neutral/ambiguous feedback."""
    # A reply that doesn't trigger keywords in the formatter
    reply = "hmm, let's look at this differently"
    mock_llm_client.generate.return_value = "this is a correction"

    await rl_metrics_system_v2.capture_next_state_signal(reply, "test_action", "user_2")

    mock_llm_client.generate.assert_called_once()
    calls = mock_quality_tracker.log_metric.call_args_list

    # Should have logged rl_implicit_sentiment_v2 and rl_quality_score_v2
    assert len(calls) == 2

    # Intent should have been updated to criticism due to LLM
    metadata = calls[0][0][2]
    assert metadata["intent"] == "criticism"
    assert metadata["is_correction"] == True

@pytest.mark.asyncio
async def test_capture_next_state_signal_positive_implicit(rl_metrics_system_v2, mock_quality_tracker):
    """Test capturing positive implicit sentiment."""
    reply = "That is perfect, thanks!"
    await rl_metrics_system_v2.capture_next_state_signal(reply, "test_action", "user_3")

    calls = mock_quality_tracker.log_metric.call_args_list
    assert len(calls) == 2

    sentiment_call = calls[0]
    assert sentiment_call[0][0] == "rl_implicit_sentiment_v2"
    assert sentiment_call[0][1] > 0.0

    quality_call = calls[1]
    assert quality_call[0][0] == "rl_quality_score_v2"
    assert quality_call[0][1] > 50.0
