import pytest
from magda_agent.emotions.mental_states import MentalStates

def test_mental_states_initialization():
    ms = MentalStates()
    summary = ms.get_summary(user_id=123)
    assert "Fear: 0.00" in summary
    assert "Desire: 0.50" in summary
    assert "Tension: 0.00" in summary
    assert "Satisfied" in summary

def test_update_from_action_result_success():
    ms = MentalStates()
    user_id = 1
    # Success: fear -, desire +, tension --
    ms.update_from_action_result(success=True, user_id=user_id)
    state = ms._get_state(user_id)
    assert state.fear == 0.0
    assert state.desire == 0.65
    assert state.tension == 0.0

    # Multiple successes to reach "Determined"
    for _ in range(3):
        ms.update_from_action_result(success=True, user_id=user_id)

    assert ms.get_state_label(user_id) == "Determined"
    assert "Determined" in ms.get_summary(user_id)

def test_update_from_action_result_failure():
    ms = MentalStates()
    user_id = 2
    # Failure: fear +, tension +, desire -
    ms.update_from_action_result(success=False, user_id=user_id)
    state = ms._get_state(user_id)
    assert state.fear == 0.2
    assert state.tension == 0.15
    assert state.desire == 0.35

    # Multiple failures to reach "Anxious"
    for _ in range(5):
        ms.update_from_action_result(success=False, user_id=user_id)

    assert ms.get_state_label(user_id) == "Anxious"

def test_satisfied_label():
    ms = MentalStates()
    user_id = 3
    state = ms._get_state(user_id)
    state.fear = 0.1
    state.tension = 0.1
    state.desire = 0.5
    assert ms.get_state_label(user_id) == "Satisfied"

def test_focused_label():
    ms = MentalStates()
    user_id = 4
    state = ms._get_state(user_id)
    state.tension = 0.7
    assert ms.get_state_label(user_id) == "Focused"
