import pytest
from magda_agent.learning.online_learning import OnlineLearnerModule

def test_online_learning_extracts_constraint():
    learner = OnlineLearnerModule()

    user_msg = "Please don't use markdown in your replies."
    agent_msg = "I will format my text accordingly."

    learner.process_dialogue(user_msg, agent_msg)

    insights = learner.get_recent_insights()
    assert len(insights) == 1
    assert insights[0]["type"] == "constraint"
    assert insights[0]["content"] == user_msg

def test_online_learning_extracts_preference():
    learner = OnlineLearnerModule()

    user_msg = "You must always be polite."
    agent_msg = "I will be polite."

    learner.process_dialogue(user_msg, agent_msg)

    insights = learner.get_recent_insights()
    assert len(insights) == 1
    assert insights[0]["type"] == "preference"
    assert insights[0]["content"] == user_msg

def test_online_learning_no_insight():
    learner = OnlineLearnerModule()

    user_msg = "What is the weather?"
    agent_msg = "It is sunny."

    learner.process_dialogue(user_msg, agent_msg)

    insights = learner.get_recent_insights()
    assert len(insights) == 0
