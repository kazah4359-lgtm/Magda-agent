import pytest
from magda_agent.attention.salience import SalienceNetwork
from magda_agent.attention.workspace import GlobalWorkspace

class MockPlanner:
    def __init__(self):
        self.current_plan = [{"description": "step 1"}]
        self.completed_steps = []
        self.current_risk = "medium"
        self.paused = False

    def pause_current_plan(self):
        self.paused = True
        self.current_plan = []

class MockSalienceNetwork(SalienceNetwork):
    def score_event(self, event):
        score = event.get("mock_score", 0.1)
        return score, "mock explanation"

def test_evaluate_interrupt_high_risk_plan():
    sn = MockSalienceNetwork()

    # Event score 0.7, plan risk high (needs 0.8)
    event1 = {"mock_score": 0.7}
    assert not sn.evaluate_interrupt(event1, current_plan_risk="high")

    # Event score 0.9, plan risk high (needs 0.8)
    event2 = {"mock_score": 0.9}
    assert sn.evaluate_interrupt(event2, current_plan_risk="high")

def test_evaluate_interrupt_explicit_urgency():
    sn = MockSalienceNetwork()

    event = {"mock_score": 0.1, "urgency": 0.9}
    assert sn.evaluate_interrupt(event, current_plan_risk="critical")

def test_process_interruption_interrupts():
    sn = MockSalienceNetwork()
    gw = GlobalWorkspace(salience_network=sn)
    planner = MockPlanner()

    # Event score 0.7, plan risk medium (needs 0.6)
    event = {"mock_score": 0.7}

    interrupted = gw.process_interruption(event, planner)

    assert interrupted is True
    assert planner.paused is True
    assert len(gw.candidates) == 1
    assert gw.candidates[0] == event

def test_process_interruption_does_not_interrupt():
    sn = MockSalienceNetwork()
    gw = GlobalWorkspace(salience_network=sn)
    planner = MockPlanner()

    # Event score 0.5, plan risk medium (needs 0.6)
    event = {"mock_score": 0.5}

    interrupted = gw.process_interruption(event, planner)

    assert interrupted is False
    assert planner.paused is False
    assert len(gw.candidates) == 1
    assert gw.candidates[0] == event
