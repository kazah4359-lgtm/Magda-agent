import pytest
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.learning.reward_propagator_v6 import (
    OpenClawRLRewardPropagatorV6,
    RewardPropagatorV6,
    ExecutionNode
)


def test_register_execution_and_lookup():
    propagator = OpenClawRLRewardPropagatorV6()

    node1 = propagator.register_execution(
        node_id="root_1",
        agent_id="orchestrator",
        skill_id="plan_task",
        context_metadata={"session_id": "sess_123", "step": 1}
    )

    node2 = propagator.register_execution(
        node_id="sub_1",
        agent_id="subagent_a",
        skill_id="code_gen",
        parent_id="root_1",
        context_metadata={"session_id": "sess_123", "step": 2}
    )

    assert node1.node_id == "root_1"
    assert node1.depth == 0
    assert node1.children_ids == ["sub_1"]
    assert node2.parent_id == "root_1"
    assert node2.depth == 1

    fetched_node = propagator.get_node("sub_1")
    assert fetched_node is not None
    assert fetched_node.agent_id == "subagent_a"

    nodes_by_meta = propagator.find_nodes_by_metadata("session_id", "sess_123")
    assert len(nodes_by_meta) == 2


def test_pad_signal_extraction_and_reward_computation():
    propagator = OpenClawRLRewardPropagatorV6()

    # Positive user reply
    pos_pad = propagator.extract_pad_signal("Great job! Excellent result.")
    pos_reward = propagator.compute_reward(pos_pad, tool_output="Success")
    assert pos_reward > 0.0

    # Negative user reply
    neg_pad = propagator.extract_pad_signal("Bad error! Terrible outcome.")
    neg_reward = propagator.compute_reward(neg_pad)
    assert neg_reward < 0.0

    # Neutral / empty reply
    neutral_pad = propagator.extract_pad_signal("")
    assert neutral_pad == (0.0, 0.0, 0.0)
    assert propagator.compute_reward(neutral_pad) == 0.0


def test_recursive_propagation_upward_and_downward():
    propagator = OpenClawRLRewardPropagatorV6(learning_rate=0.2, decay_factor=0.5)

    # Setup tree:
    # parent (orch, skill_plan)
    #   ├── child1 (sub_a, skill_code)
    #   └── child2 (sub_b, skill_test)
    propagator.register_execution("parent", "orch", "skill_plan")
    propagator.register_execution("child1", "sub_a", "skill_code", parent_id="parent")
    propagator.register_execution("child2", "sub_b", "skill_test", parent_id="parent")

    # Initial weights
    assert propagator.get_agent_weight("orch") == 1.0
    assert propagator.get_agent_weight("sub_a") == 1.0
    assert propagator.get_agent_weight("sub_b") == 1.0

    # Feedback on child1 (positive)
    updated = propagator.propagate_feedback("child1", "Excellent work!", tool_output="Done")

    # child1 should get full positive update
    # parent should get decayed update
    # child2 (sibling) should get further decayed update
    assert propagator.get_agent_weight("sub_a") > 1.0
    assert propagator.get_agent_weight("orch") > 1.0
    assert propagator.get_agent_weight("sub_b") > 1.0

    # Direct child1 weight > parent weight > sibling sub_b weight
    w_sub_a = propagator.get_agent_weight("sub_a")
    w_orch = propagator.get_agent_weight("orch")
    w_sub_b = propagator.get_agent_weight("sub_b")

    assert w_sub_a > w_orch > w_sub_b > 1.0
    assert "agent:sub_a" in updated
    assert "skill:skill_code" in updated


def test_weight_clamping_and_edge_cases():
    propagator = OpenClawRLRewardPropagatorV6(
        learning_rate=1.0,
        min_weight=0.1,
        max_weight=2.0
    )

    propagator.register_execution("n1", "ag1", "sk1")

    # Repeated high positive feedback
    for _ in range(5):
        propagator.propagate_feedback("n1", "Excellent amazing perfect fantastic!")

    assert propagator.get_agent_weight("ag1") == 2.0
    assert propagator.get_skill_weight("sk1") == 2.0

    # Repeated negative feedback
    for _ in range(10):
        propagator.propagate_feedback("n1", "Bad sad terrible awful fail!")

    assert propagator.get_agent_weight("ag1") == 0.1
    assert propagator.get_skill_weight("sk1") == 0.1

    # Non-existent node ID
    empty_result = propagator.propagate_feedback("non_existent", "Great job!")
    assert empty_result == {}


def test_alias_compatibility():
    propagator = RewardPropagatorV6()
    assert isinstance(propagator, OpenClawRLRewardPropagatorV6)
