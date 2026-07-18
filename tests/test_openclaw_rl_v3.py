import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.learning.openclaw_rl_v3 import OnlineRLFeedbackLoop
from magda_agent.llm_client import LLMClient

def test_map_reply_to_reward_positive() -> None:
    """Tests parsing positive user replies to reward scores."""
    loop = OnlineRLFeedbackLoop()
    assert loop._map_reply_to_reward("That's great") == 1.0
    assert loop._map_reply_to_reward("good job") == 1.0
    assert loop._map_reply_to_reward("awesome!") == 1.0
    assert loop._map_reply_to_reward("Yes please") == 1.0
    assert loop._map_reply_to_reward("thanks") == 1.0

def test_map_reply_to_reward_negative() -> None:
    """Tests parsing negative user replies to reward scores."""
    loop = OnlineRLFeedbackLoop()
    assert loop._map_reply_to_reward("That's bad") == -1.0
    assert loop._map_reply_to_reward("terrible") == -1.0
    assert loop._map_reply_to_reward("no") == -1.0
    assert loop._map_reply_to_reward("wrong") == -1.0

def test_map_reply_to_reward_neutral() -> None:
    """Tests parsing neutral user replies to reward scores."""
    loop = OnlineRLFeedbackLoop()
    assert loop._map_reply_to_reward("Okay") == 0.0
    assert loop._map_reply_to_reward("Maybe") == 0.0
    assert loop._map_reply_to_reward("I don't know") == 0.0

def test_process_feedback() -> None:
    """Tests processing feedback and verifying Q-value updates."""
    loop = OnlineRLFeedbackLoop()

    # Test positive feedback
    loop.process_feedback("skill_1", "Great")
    assert loop.get_q_value("skill_1") == 0.1  # 0 + 0.1 * (1 - 0)

    # Test multiple positive feedback
    loop.process_feedback("skill_1", "Good")
    assert loop.get_q_value("skill_1") == 0.19 # 0.1 + 0.1 * (1 - 0.1)

    # Test negative feedback
    loop.process_feedback("skill_2", "Bad")
    assert loop.get_q_value("skill_2") == -0.1 # 0 + 0.1 * (-1 - 0)

@pytest.mark.asyncio
async def test_extract_reward_llm_positive() -> None:
    """Tests that a mocked LLM returning a positive reward is processed correctly."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate = AsyncMock(return_value="0.85")

    loop = OnlineRLFeedbackLoop(llm_client=mock_llm)
    reward = await loop.extract_reward_llm("The response was fantastic!", "Successfully fetched data")

    assert reward == 0.85
    mock_llm.generate.assert_called_once()
    assert "The response was fantastic!" in mock_llm.generate.call_args[0][0]
    assert "Successfully fetched data" in mock_llm.generate.call_args[0][0]

@pytest.mark.asyncio
async def test_extract_reward_llm_negative_out_of_bounds() -> None:
    """Tests that a mocked LLM returning an out of bounds value gets capped correctly."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate = AsyncMock(return_value="-1.5")

    loop = OnlineRLFeedbackLoop(llm_client=mock_llm)
    reward = await loop.extract_reward_llm("This is total garbage.", "Error: 500")

    assert reward == -1.0
    mock_llm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_extract_reward_llm_malformed_text() -> None:
    """Tests that a mocked LLM returning text with a number embedded gets extracted correctly."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate = AsyncMock(return_value="The reward is: 0.5 points.")

    loop = OnlineRLFeedbackLoop(llm_client=mock_llm)
    reward = await loop.extract_reward_llm("Nice work", "Printed output")

    assert reward == 0.5

@pytest.mark.asyncio
async def test_extract_reward_llm_error_fallback() -> None:
    """Tests that the system falls back to heuristics if the LLM generate call fails."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate = AsyncMock(side_effect=Exception("API Error"))

    loop = OnlineRLFeedbackLoop(llm_client=mock_llm)
    reward = await loop.extract_reward_llm("thanks", "some output")

    # Should fall back to heuristic, "thanks" maps to 1.0
    assert reward == 1.0

@pytest.mark.asyncio
async def test_process_feedback_async() -> None:
    """Tests async processing of feedback and verifying Q-value updates."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate = AsyncMock(return_value="0.5")

    loop = OnlineRLFeedbackLoop(llm_client=mock_llm)

    # First async feedback: reward is 0.5
    await loop.process_feedback_async("skill_async_1", "quite good", "success context")
    assert loop.get_q_value("skill_async_1") == 0.05  # 0 + 0.1 * (0.5 - 0)

    # Second async feedback: reward is 0.5
    await loop.process_feedback_async("skill_async_1", "quite good", "success context")
    assert abs(loop.get_q_value("skill_async_1") - 0.095) < 1e-6  # 0.05 + 0.1 * (0.5 - 0.05)
