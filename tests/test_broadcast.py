import pytest
from magda_agent.attention.workspace import GlobalWorkspace
from magda_agent.attention.salience import SalienceNetwork

class MockSalienceNetwork(SalienceNetwork):
    def score_event(self, event):
        score = event.get("mock_score", 0.1)
        return score, "mock explanation"

def test_workspace_broadcasts_to_listeners():
    salience_network = MockSalienceNetwork()
    workspace = GlobalWorkspace(salience_network=salience_network)

    received_events = []

    def mock_listener(event):
        received_events.append(event)

    workspace.register_listener(mock_listener)

    event1 = {"id": 1, "mock_score": 0.5}
    event2 = {"id": 2, "mock_score": 0.9}

    workspace.add_candidate(event1)
    workspace.add_candidate(event2)

    focused = workspace.select_focus()

    assert focused is not None
    assert focused["id"] == 2
    assert len(received_events) == 1
    assert received_events[0]["id"] == 2
    assert received_events[0]["_salience_score"] == 0.9

def test_workspace_multiple_listeners():
    salience_network = MockSalienceNetwork()
    workspace = GlobalWorkspace(salience_network=salience_network)

    listener1_received = []
    listener2_received = []

    workspace.register_listener(lambda e: listener1_received.append(e))
    workspace.register_listener(lambda e: listener2_received.append(e))

    workspace.add_candidate({"id": 1, "mock_score": 0.8})
    workspace.select_focus()

    assert len(listener1_received) == 1
    assert len(listener2_received) == 1
    assert listener1_received[0]["id"] == 1
    assert listener2_received[0]["id"] == 1
