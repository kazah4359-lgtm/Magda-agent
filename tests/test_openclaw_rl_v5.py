import pytest
import asyncio
from magda_agent.learning.openclaw_rl_v5 import OnlineRLIntegrator

@pytest.fixture
def integrator():
    return OnlineRLIntegrator(initial_weights={"search": 1.0})

def test_parse_feedback_positive(integrator):
    assert integrator.parse_feedback("That was good, thanks!") == 1.0
    assert integrator.parse_feedback("Excellent job") == 1.0

def test_parse_feedback_negative(integrator):
    assert integrator.parse_feedback("That was terrible") == -1.0
    assert integrator.parse_feedback("No, that is wrong") == -1.0

def test_parse_feedback_neutral(integrator):
    assert integrator.parse_feedback("Okay") == 0.0

@pytest.mark.asyncio
async def test_process_feedback_positive(integrator):
    await integrator.process_feedback(
        user_feedback="great",
        action_context="search context",
        user_id=123,
        skill_used="search"
    )
    assert integrator.skill_weights["search"] == 1.1

@pytest.mark.asyncio
async def test_process_feedback_negative(integrator):
    await integrator.process_feedback(
        user_feedback="bad",
        action_context="search context",
        user_id=123,
        skill_used="search"
    )
    assert integrator.skill_weights["search"] == 0.9

@pytest.mark.asyncio
async def test_process_feedback_new_skill(integrator):
    await integrator.process_feedback(
        user_feedback="good",
        action_context="math context",
        user_id=123,
        skill_used="math_skill"
    )
    assert integrator.skill_weights["math_skill"] == 1.1

@pytest.mark.asyncio
async def test_process_feedback_minimum_threshold(integrator):
    integrator.skill_weights["poor_skill"] = 0.15
    await integrator.process_feedback(
        user_feedback="bad",
        action_context="poor context",
        user_id=123,
        skill_used="poor_skill"
    )
    assert integrator.skill_weights["poor_skill"] == 0.1
