import json
from unittest.mock import MagicMock

from magda_agent.architecture.agent_teams_v3 import AgentTeamManagerV3, AgentWorktreeIsolationV3
from magda_agent.visualization.multi_agent_stream import MultiAgentStreamServer

def test_get_multi_agent_state_json_empty():
    mock_isolation = MagicMock(spec=AgentWorktreeIsolationV3)
    mock_isolation.base_dir = "/mock/base"
    mock_isolation.active_worktrees = {}

    mock_team_manager = MagicMock(spec=AgentTeamManagerV3)
    mock_team_manager.isolation_manager = mock_isolation

    server = MultiAgentStreamServer(team_manager=mock_team_manager)
    state_json = server.get_state_json()
    state = json.loads(state_json)

    assert state["isolation_metrics"]["base_dir"] == "/mock/base"
    assert state["isolation_metrics"]["active_worktrees"] == 0
    assert len(state["agents"]) == 0

def test_get_multi_agent_state_json_with_agents():
    mock_isolation = MagicMock(spec=AgentWorktreeIsolationV3)
    mock_isolation.base_dir = "/mock/base"
    mock_isolation.active_worktrees = {
        "agent-1": "/mock/base/agent-1-env",
        "agent-2": "/mock/base/agent-2-env"
    }

    mock_team_manager = MagicMock(spec=AgentTeamManagerV3)
    mock_team_manager.isolation_manager = mock_isolation

    server = MultiAgentStreamServer(team_manager=mock_team_manager)
    state_json = server.get_state_json()
    state = json.loads(state_json)

    assert state["isolation_metrics"]["base_dir"] == "/mock/base"
    assert state["isolation_metrics"]["active_worktrees"] == 2
    assert len(state["agents"]) == 2

    agent_ids = [agent["agent_id"] for agent in state["agents"]]
    assert "agent-1" in agent_ids
    assert "agent-2" in agent_ids

    agent_1 = next(agent for agent in state["agents"] if agent["agent_id"] == "agent-1")
    assert agent_1["worktree"] == "/mock/base/agent-1-env"
    assert agent_1["status"] == "active"

def test_get_multi_agent_state_json_error_handling():
    mock_team_manager = MagicMock(spec=AgentTeamManagerV3)
    # Simulate an error by having accessing isolation_manager raise an exception
    type(mock_team_manager).isolation_manager = property(lambda self: (_ for _ in ()).throw(ValueError("Mock Exception")))

    server = MultiAgentStreamServer(team_manager=mock_team_manager)
    state_json = server.get_state_json()
    state = json.loads(state_json)

    assert "error" in state
    assert "Mock Exception" in state["error"]
