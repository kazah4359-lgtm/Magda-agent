import pytest
from magda_agent.attention.workspace import GlobalWorkspace
from magda_agent.attention.salience import SalienceNetwork


class MockSalienceNetwork(SalienceNetwork):
    def score_event(self, event):
        # Allow testing by providing a fixed score in the event
        score = event.get("mock_score", 0.1)
        explanation = "mock explanation"
        return score, explanation


def test_global_workspace_selects_highest_salience():
    salience_network = MockSalienceNetwork()
    workspace = GlobalWorkspace(salience_network=salience_network)

    event1 = {"id": 1, "mock_score": 0.2}
    event2 = {"id": 2, "mock_score": 0.9}
    event3 = {"id": 3, "mock_score": 0.5}

    workspace.add_candidate(event1)
    workspace.add_candidate(event2)
    workspace.add_candidate(event3)

    focused = workspace.select_focus()

    assert focused is not None
    assert focused["id"] == 2
    assert focused["_salience_score"] == 0.9


def test_global_workspace_stores_suppressed_candidates():
    salience_network = MockSalienceNetwork()
    workspace = GlobalWorkspace(salience_network=salience_network)

    event1 = {"id": 1, "mock_score": 0.8}
    event2 = {"id": 2, "mock_score": 0.4}
    event3 = {"id": 3, "mock_score": 0.6}

    workspace.add_candidate(event1)
    workspace.add_candidate(event2)
    workspace.add_candidate(event3)

    focused = workspace.select_focus()

    assert focused is not None
    assert focused["id"] == 1

    suppressed = workspace.get_suppressed_candidates()
    assert len(suppressed) == 2
    suppressed_ids = [e["id"] for e in suppressed]
    assert 2 in suppressed_ids
    assert 3 in suppressed_ids


def test_global_workspace_handles_empty_candidates():
    salience_network = MockSalienceNetwork()
    workspace = GlobalWorkspace(salience_network=salience_network)

    focused = workspace.select_focus()
    assert focused is None
    assert len(workspace.get_suppressed_candidates()) == 0


def test_global_workspace_clear():
    salience_network = MockSalienceNetwork()
    workspace = GlobalWorkspace(salience_network=salience_network)

    workspace.add_candidate({"id": 1, "mock_score": 0.5})
    workspace.clear()

    focused = workspace.select_focus()
    assert focused is None
