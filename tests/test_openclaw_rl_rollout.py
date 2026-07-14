import pytest
from magda_agent.learning.openclaw_rl_rollout import OpenClawRLTrajectoryRolloutV6, ConversationStep

def test_trajectory_buffer_turn_persistence() -> None:
    """
    Tests that trajectory steps are persisted correctly across turns in the user buffer,
    including limit eviction and clear capabilities.
    """
    learner = OpenClawRLTrajectoryRolloutV6(initial_q_values={"search": 1.0})
    user_id = "user_abc_123"

    # Initially, buffer is empty
    buffer = learner.get_buffer(user_id)
    assert len(buffer.steps) == 0

    # Record first step
    learner.record_step(user_id, "search", "search state context")
    assert len(buffer.steps) == 1
    assert buffer.steps[0].skill_id == "search"
    assert buffer.steps[0].state_context == "search state context"
    assert buffer.steps[0].q_value == 1.0

    # Record second step
    learner.record_step(user_id, "math", "math state context")
    assert len(buffer.steps) == 2
    assert buffer.steps[1].skill_id == "math"
    assert buffer.steps[1].state_context == "math state context"
    assert buffer.steps[1].q_value == 0.0  # default Q-value is 0.0

    # Check that another user has a separate buffer
    other_user_id = "user_def_456"
    other_buffer = learner.get_buffer(other_user_id)
    assert len(other_buffer.steps) == 0

    # Test buffer eviction with custom max_size
    buffer.max_size = 2
    learner.record_step(user_id, "code", "code state context")
    # Buffer has max_size = 2, so "search" should be evicted
    assert len(buffer.steps) == 2
    assert buffer.steps[0].skill_id == "math"
    assert buffer.steps[1].skill_id == "code"


def test_trajectory_delayed_reward_backpropagation() -> None:
    """
    Tests that the delayed reward is properly discounted and backpropagated to previous steps
    and updates the Q-table correctly.
    """
    learner = OpenClawRLTrajectoryRolloutV6(initial_q_values={"skill_a": 0.5, "skill_b": 0.3})
    user_id = "user_xyz"

    # Record 3 steps
    learner.record_step(user_id, "skill_a", "state 1")
    learner.record_step(user_id, "skill_b", "state 2")
    learner.record_step(user_id, "skill_a", "state 3")

    # Apply delayed reward
    # final_reward = 1.0, discount_factor = 0.9, learning_rate = 0.1
    # n = 3 steps
    # Step 0 (idx 0, distance 2 from end): skill_a
    #   discounted_reward = 1.0 * (0.9 ** 2) = 0.81
    #   Q_new = 0.5 + 0.1 * (0.81 - 0.5) = 0.5 + 0.031 = 0.531
    # Step 1 (idx 1, distance 1 from end): skill_b
    #   discounted_reward = 1.0 * (0.9 ** 1) = 0.9
    #   Q_new = 0.3 + 0.1 * (0.9 - 0.3) = 0.3 + 0.06 = 0.36
    # Step 2 (idx 2, distance 0 from end): skill_a
    #   discounted_reward = 1.0 * (0.9 ** 0) = 1.0
    #   Q_new = 0.531 + 0.1 * (1.0 - 0.531) = 0.531 + 0.0469 = 0.5779

    updated_values = learner.process_delayed_reward(
        user_id,
        final_reward=1.0,
        discount_factor=0.9,
        learning_rate=0.1
    )

    # Check returned updated values
    assert pytest.approx(updated_values["skill_b"], abs=1e-5) == 0.36
    assert pytest.approx(updated_values["skill_a"], abs=1e-5) == 0.5779

    # Check Q-table
    assert pytest.approx(learner.get_q_value("skill_a"), abs=1e-5) == 0.5779
    assert pytest.approx(learner.get_q_value("skill_b"), abs=1e-5) == 0.36

    # Verify buffer is cleared after processing
    buffer = learner.get_buffer(user_id)
    assert len(buffer.steps) == 0


def test_trajectory_rollout_empty_buffer() -> None:
    """
    Tests delayed reward processing on an empty buffer does not crash and returns empty.
    """
    learner = OpenClawRLTrajectoryRolloutV6()
    updated = learner.process_delayed_reward("empty_user", 1.0)
    assert updated == {}
