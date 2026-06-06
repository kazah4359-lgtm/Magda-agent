import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.metacognition.confidence import ConfidenceCalibrator
from magda_agent.llm_client import LLMClient
from magda_agent.metacognition.tracker import QualityTracker

@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMClient)
    llm.chat_completion.return_value = "0.9"
    return llm

@pytest.fixture
def mock_tracker():
    tracker = MagicMock(spec=QualityTracker)
    return tracker

@pytest.fixture
def calibrator(mock_llm, mock_tracker):
    return ConfidenceCalibrator(llm=mock_llm, tracker=mock_tracker)

@pytest.mark.asyncio
async def test_estimate_confidence_high(calibrator, mock_llm):
    score = await calibrator.estimate_confidence("Hello", "Hi there!")
    assert score == 0.9
    assert calibrator.last_confidence == 0.9
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_estimate_confidence_low(calibrator, mock_llm):
    mock_llm.chat_completion.return_value = "0.3"
    score = await calibrator.estimate_confidence("Explain quantum gravity", "I think it is magic.")
    assert score == 0.3
    assert calibrator.last_confidence == 0.3

@pytest.mark.asyncio
async def test_estimate_confidence_exception(calibrator, mock_llm):
    mock_llm.chat_completion.side_effect = Exception("API Error")
    score = await calibrator.estimate_confidence("Hello", "Hi there!")
    assert score == 0.5
    assert calibrator.last_confidence == 0.5

def test_add_caveat_high_confidence(calibrator):
    response = calibrator.add_caveat_if_needed("Direct answer.", 0.9)
    assert "Direct answer." == response
    assert "verify the details" not in response

def test_add_caveat_low_confidence(calibrator):
    response = calibrator.add_caveat_if_needed("Guess answer.", 0.4)
    assert "Guess answer." in response
    assert "verify the details" in response

def test_track_calibration(calibrator, mock_tracker):
    # Predicted confidence: 0.8
    # Actual score: 6.0 (scaled to 0.6)
    # Error: |0.8 - 0.6| = 0.2
    calibrator.track_calibration(0.8, 6.0)
    mock_tracker.log_metric.assert_called_once()

    args, kwargs = mock_tracker.log_metric.call_args
    assert args[0] == "calibration_error"
    assert pytest.approx(args[1]) == 0.2
    assert kwargs["metadata"]["confidence"] == 0.8
    assert kwargs["metadata"]["actual_score"] == 6.0
