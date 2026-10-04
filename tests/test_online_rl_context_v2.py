import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.learning.online_rl_context_v2 import OpenClawContextOnlineRLV2


def test_initial_default_weights():
    rl_module = OpenClawContextOnlineRLV2()
    weights = rl_module.get_retrieval_weights()

    assert weights["episodic"] == 1.0
    assert weights["semantic"] == 1.0
    assert weights["procedural"] == 1.0
    assert weights["working"] == 1.0


def test_adjust_weights_and_clamping():
    rl_module = OpenClawContextOnlineRLV2(min_weight=0.1, max_weight=3.0)

    # Increase weight
    new_weight = rl_module.adjust_weights_for_context("episodic", 0.5, user_id="user_1")
    assert new_weight == 1.5
    assert rl_module.get_retrieval_weights("user_1")["episodic"] == 1.5

    # Check upper clamp
    clamped_max = rl_module.adjust_weights_for_context("episodic", 5.0, user_id="user_1")
    assert clamped_max == 3.0

    # Check lower clamp
    clamped_min = rl_module.adjust_weights_for_context("semantic", -10.0, user_id="user_1")
    assert clamped_min == 0.1


@pytest.mark.asyncio
async def test_process_feedback_explicit_reward():
    rl_module = OpenClawContextOnlineRLV2(learning_rate=0.2)

    # Positive explicit reward
    reward = await rl_module.process_feedback(
        user_feedback="Great answer!",
        user_id="user_2",
        reward=1.0,
        retrieved_context=[{"category": "episodic"}, {"category": "semantic"}],
    )
    assert reward == 1.0
    weights = rl_module.get_retrieval_weights("user_2")
    assert weights["episodic"] == pytest.approx(1.2)
    assert weights["semantic"] == pytest.approx(1.2)
    assert weights["procedural"] == 1.0  # Unaffected

    # Negative explicit reward
    reward_neg = await rl_module.process_feedback(
        user_feedback="Wrong context",
        user_id="user_2",
        reward=-0.5,
        retrieved_context=[{"category": "semantic"}],
    )
    assert reward_neg == -0.5
    weights_after = rl_module.get_retrieval_weights("user_2")
    assert weights_after["semantic"] == pytest.approx(1.1)  # 1.2 + (-0.5 * 0.2) = 1.1


@pytest.mark.asyncio
async def test_process_feedback_with_mirror_neurons_mock():
    mock_mirror = MagicMock()
    mock_mirror.empathize.return_value = (0.4, 0.1, 0.0)

    rl_module = OpenClawContextOnlineRLV2(mirror_neurons=mock_mirror, learning_rate=0.1)

    reward = await rl_module.process_feedback(
        user_feedback="That was fantastic and wonderful!",
        user_id="user_3",
    )

    mock_mirror.empathize.assert_called_once_with("That was fantastic and wonderful!")
    assert reward == 0.4
    weights = rl_module.get_retrieval_weights("user_3")
    assert weights["episodic"] == pytest.approx(1.04)


def test_rank_and_filter_context():
    rl_module = OpenClawContextOnlineRLV2()
    rl_module.adjust_weights_for_context("semantic", 1.0, user_id="user_4")  # semantic weight = 2.0
    rl_module.adjust_weights_for_context("episodic", -0.5, user_id="user_4")  # episodic weight = 0.5

    context_items = [
        {"id": 1, "category": "episodic", "relevance": 1.0},
        {"id": 2, "category": "semantic", "relevance": 0.8},
        {"id": 3, "category": "working", "relevance": 0.9},
    ]

    ranked = rl_module.rank_and_filter_context(context_items, user_id="user_4")

    # Semantic score: 0.8 * 2.0 = 1.6
    # Working score: 0.9 * 1.0 = 0.9
    # Episodic score: 1.0 * 0.5 = 0.5
    assert [item["id"] for item in ranked] == [2, 3, 1]
    assert ranked[0]["_weighted_score"] == pytest.approx(1.6)


@pytest.mark.asyncio
async def test_context_plugin_lifecycle_hooks():
    rl_module = OpenClawContextOnlineRLV2()
    rl_module.adjust_weights_for_context("semantic", 0.5, user_id="101")

    context = [
        {"id": "ep1", "category": "episodic", "score": 1.0},
        {"id": "sem1", "category": "semantic", "score": 0.8},
    ]

    # Test after_retrieval hook
    result = rl_module.after_retrieval(context, query="test query", user_id=101)
    # sem1 weighted score: 0.8 * 1.5 = 1.2
    # ep1 weighted score: 1.0 * 1.0 = 1.0
    assert result[0]["id"] == "sem1"
    assert result[1]["id"] == "ep1"

    # Test on_context_update with AsyncMock
    with patch.object(rl_module, "process_feedback", new_callable=AsyncMock) as mock_process:
        rl_module.on_context_update(
            {"user_reply": "Good job!", "reward": 0.5, "retrieved_context": context},
            user_id=101,
        )
        mock_process.assert_called_once_with(
            user_feedback="Good job!",
            retrieved_context=context,
            user_id="101",
            reward=0.5,
        )
