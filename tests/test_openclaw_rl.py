import pytest
import os
import json
import tempfile
from unittest.mock import MagicMock
from magda_agent.learning.openclaw_rl import OpenClawInteractiveLearner
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.user_model.model import UserModel
from magda_agent.learning.lessons import TaskRecoveryLessons

@pytest.fixture
def mock_habit_tracker():
    tracker = MagicMock(spec=HabitTracker)
    return tracker

@pytest.fixture
def mock_mirror_neurons():
    neurons = MagicMock(spec=MirrorNeurons)
    return neurons

@pytest.fixture
def mock_recovery_lessons():
    lessons = MagicMock(spec=TaskRecoveryLessons)
    return lessons

@pytest.fixture
def temp_user_model_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def user_model(temp_user_model_dir):
    model = UserModel(persist_dir=temp_user_model_dir, llm=None)
    return model

@pytest.mark.asyncio
async def test_openclaw_rl_positive_signal(mock_habit_tracker: MagicMock, mock_mirror_neurons: MagicMock, user_model: UserModel) -> None:
    """Tests that a positive next-state signal reinforces habits and updates preferences."""
    learner = OpenClawInteractiveLearner(mock_habit_tracker, mock_mirror_neurons, user_model)

    # Mock positive empathize
    mock_mirror_neurons.empathize.return_value = (0.2, 0.1, 0.0)

    await learner.process_next_state_signal("Great job", "test_context", 1, skills_used=["skill_a", "skill_b"])

    # Check habit tracker called for both skills
    assert mock_habit_tracker.record_usage.call_count == 2
    mock_habit_tracker.record_usage.assert_any_call(
        input_text="test_context", skill_used="skill_a", evaluation_score=10.0, user_id=1
    )
    mock_habit_tracker.record_usage.assert_any_call(
        input_text="test_context", skill_used="skill_b", evaluation_score=10.0, user_id=1
    )

    # Check user model modification
    model_data = user_model.get_model(1)
    assert model_data["preferences"]["last_p_shift"] == 0.2
    assert "(friendly)" in model_data["communication_style"]

@pytest.mark.asyncio
async def test_openclaw_rl_negative_signal(mock_habit_tracker: MagicMock, mock_mirror_neurons: MagicMock, user_model: UserModel, mock_recovery_lessons: MagicMock) -> None:
    """Tests that a negative next-state signal generates recovery lessons and updates preferences."""
    learner = OpenClawInteractiveLearner(
        mock_habit_tracker, mock_mirror_neurons, user_model, recovery_lessons=mock_recovery_lessons
    )

    # Mock negative empathize
    mock_mirror_neurons.empathize.return_value = (-0.3, 0.1, 0.1)

    await learner.process_next_state_signal("This is terrible", "test_context", 2)

    # Check habit tracker NOT called
    mock_habit_tracker.record_usage.assert_not_called()

    # Check recovery lesson generation
    mock_recovery_lessons.generate_and_store_lesson.assert_called_once_with(
        task_description="test_context",
        failure_reason="This is terrible",
        user_id=2
    )

    # Check user model modification
    model_data = user_model.get_model(2)
    assert model_data["preferences"]["last_p_shift"] == -0.3
    assert "(cautious)" in model_data["communication_style"]

@pytest.mark.asyncio
async def test_openclaw_rl_tool_output(mock_habit_tracker: MagicMock, mock_mirror_neurons: MagicMock, user_model: UserModel) -> None:
    """Tests that the OpenClawInteractiveLearner correctly processes next-state signals with tool outputs."""
    learner = OpenClawInteractiveLearner(mock_habit_tracker, mock_mirror_neurons, user_model)

    # Mock positive empathize
    mock_mirror_neurons.empathize.return_value = (0.5, 0.1, 0.0)

    await learner.process_next_state_signal("This is fine", "test_context", 3, tool_output="Success: data saved")

    # Check mirror neurons called with both
    mock_mirror_neurons.empathize.assert_called_once_with("This is fine [Tool Output: Success: data saved]")

    # Check habit tracker
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="test_context", skill_used="rl_skill", evaluation_score=10.0, user_id=3
    )

    # Check user model modification
    model_data = user_model.get_model(3)
    assert model_data["preferences"]["last_p_shift"] == 0.5
    assert "(friendly)" in model_data["communication_style"]

@pytest.mark.asyncio
async def test_openclaw_rl_dynamic_behavior_weights(mock_habit_tracker: MagicMock, mock_mirror_neurons: MagicMock, user_model: UserModel) -> None:
    """Tests that dynamic behavior weights are adjusted based on PAD shifts."""
    learner = OpenClawInteractiveLearner(mock_habit_tracker, mock_mirror_neurons, user_model)

    # Initialize a user model with specific weights
    user_model.save_model(4, {"behavior_weights": {"exploration": 1.0, "verbosity": 1.0, "directness": 1.0}})

    # Mock empathize
    # p_shift = 0.4, a_shift = 0.2, d_shift = -0.1
    mock_mirror_neurons.empathize.return_value = (0.4, 0.2, -0.1)

    await learner.process_next_state_signal("This is great but confusing", "test_context", 4)

    # Verify model modifications
    model_data = user_model.get_model(4)
    weights = model_data["behavior_weights"]

    # Exploration: 1.0 + 0.4 * 0.5 = 1.2
    assert abs(weights["exploration"] - 1.2) < 1e-6
    # Verbosity: 1.0 + 0.2 = 1.2
    assert abs(weights["verbosity"] - 1.2) < 1e-6
    # Directness: 1.0 - 0.1 = 0.9
    assert abs(weights["directness"] - 0.9) < 1e-6

    # Test bounding logic
    # Mock empathize with large shifts
    mock_mirror_neurons.empathize.return_value = (3.0, 3.0, -3.0)
    await learner.process_next_state_signal("Huge shift", "test_context", 4)
    model_data2 = user_model.get_model(4)
    weights2 = model_data2["behavior_weights"]

    assert weights2["exploration"] <= 2.0
    assert weights2["verbosity"] <= 2.0
    assert weights2["directness"] >= 0.5

@pytest.mark.asyncio
async def test_openclaw_rl_integration_weights_to_planner(mock_habit_tracker: MagicMock, mock_mirror_neurons: MagicMock, user_model: UserModel) -> None:
    """Tests that learned weights are correctly passed to the Planner through Consciousness."""
    from magda_agent.consciousness.core import Consciousness
    from magda_agent.llm_client import LLMClient
    from magda_agent.emotions.engine import EmotionalEngine
    from magda_agent.memory.storage import MemorySystem
    from magda_agent.skills.registry import SkillRegistry
    from magda_agent.planning.planner import Planner
    from unittest.mock import AsyncMock

    # Setup mocks
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock(return_value=json.dumps({
        "goal": "test", "constraints": [], "risk": "low", "steps": [], "acceptance": []
    }))
    llm.get_system_prompt.return_value = "System prompt"

    emotions = MagicMock(spec=EmotionalEngine)
    memory = MagicMock(spec=MemorySystem)
    memory.working_memory = MagicMock()
    memory.working_memory.get_entries.return_value = []
    memory.episodic_memory = MagicMock()
    memory.episodic_memory.recall_events.return_value = []
    memory.retrieve_relevant.return_value = []
    skills = MagicMock(spec=SkillRegistry)
    planner = Planner(llm=llm, skills=skills)

    learner = OpenClawInteractiveLearner(mock_habit_tracker, mock_mirror_neurons, user_model)

    # Initialize Consciousness
    consciousness = Consciousness(
        llm=llm,
        emotions=emotions,
        memory=memory,
        skills=skills,
        planner=planner,
        user_model=user_model,
        openclaw_rl=learner
    )

    # Simulate feedback to set weights
    # p_shift=0.8, a_shift=0.8, d_shift=0.8
    mock_mirror_neurons.empathize.return_value = (0.8, 0.8, 0.8)
    await consciousness.process_input("You are doing a great job!", user_id=123)

    # Verify weights were set in user model
    model_data = user_model.get_model(123)
    assert model_data["behavior_weights"]["exploration"] > 1.3

    # Reset chat_completion mock to track the call during next process_input
    llm.chat_completion.reset_mock()

    # Process next input, which should trigger plan generation with weights
    await consciousness.process_input("What is the next task?", user_id=123)

    # Verify generate_plan was called with behavior_weights (via llm.chat_completion tracking in Planner)
    # The system prompt generated in Planner.generate_plan should contain "High exploration mode"
    # Note: chat_completion is called twice: once for planning, once for response generation.
    # We want to check the planning call.
    plan_call = next(c for c in llm.chat_completion.call_args_list if "Prefrontal Cortex" in c[0][0][0]["content"])
    system_msg = plan_call[0][0][0]["content"]
    assert "High exploration mode" in system_msg
    assert "Behavioral Parameters:" in system_msg
