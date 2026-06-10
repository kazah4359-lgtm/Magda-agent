import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.architecture.sub_agents import SubAgentRPCManager, SubAgentRPCServer
from magda_agent.llm_client import LLMClient
from fastapi import Request
from fastapi.testclient import TestClient

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock(return_value="Task executed successfully.")
    return llm

@pytest.fixture
def manager(mock_llm):
    return SubAgentRPCManager(llm=mock_llm)

@pytest.fixture
def rpc_server(manager):
    return SubAgentRPCServer(manager=manager)

@pytest.fixture
def test_client(rpc_server):
    return TestClient(rpc_server.app)

@pytest.mark.asyncio
async def test_spawn_and_execute_task(manager, mock_llm):
    # Mock GitWorktreeManager to avoid actual git operations in tests
    with patch("magda_agent.agents.sub_agent.GitWorktreeManager") as mock_git:
        mock_worktree_manager = MagicMock()
        mock_worktree_manager.create_worktree_async = AsyncMock(return_value="/tmp/worktree1")
        mock_worktree_manager.remove_worktree_async = AsyncMock()
        mock_git.return_value = mock_worktree_manager

        await manager.spawn_agent("agent_1", context={"role": "developer"})

        assert "agent_1" in manager.active_agents
        agent = manager.active_agents["agent_1"]
        assert "developer" in agent.system_prompt
        assert agent.use_isolation is True

        result = await manager.execute_task_rpc("agent_1", "Write a test")

        assert result["status"] == "success"
        assert result["agent_id"] == "agent_1"
        assert result["result"] == "Task executed successfully."
        mock_llm.chat_completion.assert_called_once()

        await manager.kill_agent("agent_1")
        assert "agent_1" not in manager.active_agents

@pytest.mark.asyncio
async def test_execute_task_agent_not_found(manager):
    result = await manager.execute_task_rpc("non_existent", "Task")
    assert result["status"] == "error"
    assert "not found" in result["error"]

@pytest.mark.asyncio
async def test_execute_parallel_rpc(manager, mock_llm):
    with patch("magda_agent.agents.sub_agent.GitWorktreeManager") as mock_git:
        mock_worktree_manager = MagicMock()
        mock_worktree_manager.create_worktree_async = AsyncMock(return_value="/tmp/worktree")
        mock_worktree_manager.remove_worktree_async = AsyncMock()
        mock_git.return_value = mock_worktree_manager

        await manager.spawn_agent("agent_1")
        await manager.spawn_agent("agent_2")

        results = await manager.execute_parallel_rpc(
            ["agent_1", "agent_2"],
            ["task 1", "task 2"]
        )

        assert len(results) == 2
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "success"
        assert mock_llm.chat_completion.call_count == 2

@pytest.mark.asyncio
async def test_rpc_server_spawn_and_execute(test_client, mock_llm):
    with patch("magda_agent.agents.sub_agent.GitWorktreeManager") as mock_git:
        mock_worktree_manager = MagicMock()
        mock_worktree_manager.create_worktree_async = AsyncMock(return_value="/tmp/worktree")
        mock_git.return_value = mock_worktree_manager

        # Spawn agent
        response = test_client.post("/rpc", json={
            "jsonrpc": "2.0",
            "method": "spawn_agent",
            "params": {"agent_id": "test_agent", "context": {"env": "test"}},
            "id": 1
        })
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["status"] == "spawned"

        # Execute task
        response = test_client.post("/rpc", json={
            "jsonrpc": "2.0",
            "method": "execute_task",
            "params": {"agent_id": "test_agent", "task": "hello"},
            "id": 2
        })
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["status"] == "success"
        assert data["result"]["result"] == "Task executed successfully."

        # Kill agent
        response = test_client.post("/rpc", json={
            "jsonrpc": "2.0",
            "method": "kill_agent",
            "params": {"agent_id": "test_agent"},
            "id": 3
        })
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "killed"

def test_rpc_server_invalid_json(test_client):
    response = test_client.post("/rpc", content="not json")
    assert response.status_code == 200 # FastAPI still returns 200 but with JSON-RPC error
    assert response.json()["error"]["code"] == -32700

def test_rpc_server_method_not_found(test_client):
    response = test_client.post("/rpc", json={
        "jsonrpc": "2.0",
        "method": "invalid_method",
        "id": 1
    })
    assert response.json()["error"]["code"] == -32601
