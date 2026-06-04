import pytest
from magda_agent.action.selector import BasalGanglia

def test_basal_ganglia_initialization():
    bg = BasalGanglia()
    assert bg is not None

def test_select_action_empty():
    bg = BasalGanglia()
    assert bg.select_action([]) is None

def test_select_action_with_priorities():
    bg = BasalGanglia()
    actions = [
        {"name": "idle", "priority": 1},
        {"name": "respond", "priority": 10},
        {"name": "search", "priority": 5}
    ]
    selected = bg.select_action(actions)
    assert selected is not None
    assert selected["name"] == "respond"
    assert selected["priority"] == 10

def test_select_action_missing_priority():
    bg = BasalGanglia()
    actions = [
        {"name": "unknown"},  # default priority 0
        {"name": "low_prio", "priority": -5},
        {"name": "med_prio", "priority": 2}
    ]
    selected = bg.select_action(actions)
    assert selected is not None
    assert selected["name"] == "med_prio"

def test_select_action_all_missing_priority():
    bg = BasalGanglia()
    actions = [
        {"name": "one"},
        {"name": "two"}
    ]
    selected = bg.select_action(actions)
    assert selected is not None
    # max() returns the first item if all have the same key value (0)
    assert selected["name"] == "one"
