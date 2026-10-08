import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.learning.interactive_rl_v4 import (
    InteractiveRLLoopV4,
    OpenClawInteractiveLearnerV4,
)


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Provides a mocked LLMClient."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Score: 9.0")
    return llm


@pytest.fixture
def mock_habit_tracker() -> MagicMock:
    """Provides a mocked HabitTracker."""
    return MagicMock()


@pytest.fixture
def mock_user_model() -> MagicMock:
    """Provides a mocked UserModel."""
    um = MagicMock()
    um.get_model.return_value = {"communication_style": "default"}
    return um


def test_alias_equivalence() -> None:
    """Ensures OpenClawInteractiveLearnerV4 is an alias for InteractiveRLLoopV4."""
    assert OpenClawInteractiveLearnerV4 is InteractiveRLLoopV4


def test_heuristic_analysis() -> None:
    """Tests fallback heuristic feedback analysis."""
    loop = InteractiveRLLoopV4()

    # Positive user reply
    score_pos = loop.analyze_feedback_heuristic("That's awesome, thank you!", tool_output="Output")
    assert score_pos == 9.5  # 8.5 + 1.0 bonus

    # Negative user reply
    score_neg = loop.analyze_feedback_heuristic("No, that is completely wrong.", tool_output=None)
    assert score_neg == 2.0

    # Neutral reply
    score_neu = loop.analyze_feedback_heuristic("Okay, I see.", tool_output=None)
    assert score_neu == 5.0


@pytest.mark.asyncio
async def test_llm_analysis_success(mock_llm: AsyncMock) -> None:
    """Tests LLM feedback analysis when LLM returns a valid rating."""
    loop = InteractiveRLLoopV4(llm_client=mock_llm)
    score = await loop.analyze_feedback_llm(
        reply_text="Great work!",
        action_context="Generated code snippet",
        tool_output="Snippet created",
    )
    assert score == 9.0
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_llm_analysis_failure_fallback(mock_llm: AsyncMock) -> None:
    """Tests fallback to heuristic analysis when LLM raises an error."""
    mock_llm.generate.side_effect = RuntimeError("API rate limit")
    loop = InteractiveRLLoopV4(llm_client=mock_llm)

    score = await loop.analyze_feedback_llm(
        reply_text="Perfect answer, thanks!",
        action_context="Answered question",
    )
    assert score == 8.5  # Heuristic positive score


@pytest.mark.asyncio
async def test_process_interaction(
    mock_llm: AsyncMock,
    mock_habit_tracker: MagicMock,
    mock_user_model: MagicMock,
) -> None:
    """Tests full interaction processing cycle."""
    loop = InteractiveRLLoopV4(
        llm_client=mock_llm,
        habit_tracker=mock_habit_tracker,
        user_model=mock_user_model,
        initial_weights={"code_generator": 1.0},
        learning_rate=0.5,
    )

    reward = await loop.process_interaction(
        user_reply="That worked perfectly!",
        action_context="Fix bug in parser",
        skill_used="code_generator",
        tool_output="Tests passed",
        user_id=123,
    )

    assert reward == 9.0
    # Current weight: 1.0 + ((9.0 - 5.0) / 5.0 * 0.5) = 1.0 + 0.4 = 1.4
    state = loop.get_state()
    assert abs(state["code_generator"] - 1.4) < 1e-5

    # Habit tracker recorded usage
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="Fix bug in parser",
        skill_used="code_generator",
        evaluation_score=9.0,
        user_id=123,
    )

    # User model updated
    mock_user_model.save_model.assert_called_once()
    saved_model = mock_user_model.save_model.call_args[0][1]
    assert "(reinforced)" in saved_model["communication_style"]
    assert len(saved_model["rl_v4_history"]) == 1


@pytest.mark.asyncio
async def test_weight_clamping() -> None:
    """Tests that skill weights are properly clamped between 0.1 and 10.0."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Score: 10.0")

    loop = InteractiveRLLoopV4(
        llm_client=llm,
        initial_weights={"super_skill": 9.9},
        learning_rate=1.0,
    )

    await loop.process_interaction(
        user_reply="Amazing",
        action_context="Do something",
        skill_used="super_skill",
    )

    # Max clamp at 10.0
    assert loop.get_state()["super_skill"] == 10.0
