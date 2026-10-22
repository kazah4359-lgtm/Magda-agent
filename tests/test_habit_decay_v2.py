import pytest
import time
from magda_agent.learning.habit_decay_v2 import (
    OpenClawHabitDecayManagerV2,
    HabitDecayConfig
)


@pytest.fixture
def manager():
    config = HabitDecayConfig(
        decay_rate=0.05,
        half_life_days=10.0,
        min_weight=0.2,
        decay_mode="exponential",
        feedback_sensitivity=0.1
    )
    return OpenClawHabitDecayManagerV2(persist_directory=":memory:", config=config)


def test_config_parameterization(manager: OpenClawHabitDecayManagerV2):
    """Test dynamic config updating."""
    manager.set_config(
        decay_rate=0.1,
        half_life_days=7.0,
        min_weight=0.3,
        decay_mode="linear",
        feedback_sensitivity=0.2
    )

    assert manager.config.decay_rate == 0.1
    assert manager.config.half_life_days == 7.0
    assert manager.config.min_weight == 0.3
    assert manager.config.decay_mode == "linear"
    assert manager.config.feedback_sensitivity == 0.2


def test_exponential_decay_application(manager: OpenClawHabitDecayManagerV2):
    """Test exponential decay after simulated time passes."""
    now = time.time()
    one_day = 24 * 3600

    # Record habit 10 days ago (with half_life_days=10.0, weight should halve)
    manager.record_habit(
        input_text="deploy application to prod",
        skill_used="deploy_skill",
        initial_weight=1.0,
        timestamp=now - (10 * one_day)
    )

    summary = manager.apply_decay(current_time=now)
    assert summary["updated_count"] == 1
    assert summary["deleted_count"] == 0
    assert summary["total_remaining"] == 1

    # Inspect updated metadata
    results = manager.collection.get(include=["metadatas"])
    updated_meta = results["metadatas"][0]
    assert pytest.approx(updated_meta["weight"], abs=1e-2) == 0.5


def test_min_weight_pruning(manager: OpenClawHabitDecayManagerV2):
    """Test pruning when decayed weight falls below min_weight."""
    now = time.time()
    one_day = 24 * 3600

    # Record habit 30 days ago (with half_life_days=10.0, weight will be 1.0 * (0.5^3) = 0.125 < min_weight 0.2)
    manager.record_habit(
        input_text="old task",
        skill_used="legacy_skill",
        initial_weight=1.0,
        timestamp=now - (30 * one_day)
    )

    summary = manager.apply_decay(current_time=now)
    assert summary["deleted_count"] == 1
    assert summary["total_remaining"] == 0


def test_linear_decay_mode(manager: OpenClawHabitDecayManagerV2):
    """Test linear decay mode calculation."""
    manager.set_config(decay_mode="linear", decay_rate=0.1, min_weight=0.1)

    now = time.time()
    one_day = 24 * 3600

    # Record habit 5 days ago: weight = 1.0 - (0.1 * 5) = 0.5
    manager.record_habit(
        input_text="run benchmark",
        skill_used="perf_skill",
        initial_weight=1.0,
        timestamp=now - (5 * one_day)
    )

    summary = manager.apply_decay(current_time=now)
    assert summary["updated_count"] == 1

    results = manager.collection.get(include=["metadatas"])
    assert pytest.approx(results["metadatas"][0]["weight"], abs=1e-2) == 0.5


def test_explicit_feedback_processing(manager: OpenClawHabitDecayManagerV2):
    """Test positive and negative explicit feedback loops."""
    now = time.time()

    manager.record_habit("code review", "refactor_skill", initial_weight=1.0, timestamp=now)

    # Positive feedback (+1.0) -> increases weight by feedback_sensitivity (0.1)
    res_pos = manager.process_explicit_feedback("refactor_skill", feedback_score=1.0)
    assert res_pos["updated_count"] == 1

    results = manager.collection.get(include=["metadatas"])
    assert pytest.approx(results["metadatas"][0]["weight"], abs=1e-2) == 1.1

    # Negative feedback (-1.0) on scale 0-10 (e.g., score 0.0 -> normalized -1.0)
    res_neg = manager.process_explicit_feedback("refactor_skill", feedback_score=0.0)
    assert res_neg["updated_count"] == 1

    results = manager.collection.get(include=["metadatas"])
    # 1.1 - 0.1 = 1.0
    assert pytest.approx(results["metadatas"][0]["weight"], abs=1e-2) == 1.0
