import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from magda_agent.integration.a2a_server import A2AServer
from magda_agent.planning.planner import Planner

@pytest.fixture
def mock_planner():
    planner = MagicMock(spec=Planner)
    planner.generate_plan = AsyncMock()
    return planner

@pytest.fixture
def test_client(mock_planner):
    server = A2AServer(planner=mock_planner)
    return TestClient(server.app)

def test_parse_error(test_client):
    response = test_client.post("/rpc", data="invalid json")
    assert response.status_code == 400
    res = response.json()
    assert res["error"]["code"] == -32700

def test_invalid_request(test_client):
    response = test_client.post("/rpc", json={"method": "delegate_task"})
    assert response.status_code == 400
    res = response.json()
    assert res["error"]["code"] == -32600

def test_method_not_found(test_client):
    response = test_client.post("/rpc", json={"jsonrpc": "2.0", "method": "unknown", "id": 1})
    assert response.status_code == 404
    res = response.json()
    assert res["error"]["code"] == -32601

def test_delegate_task_missing_params(test_client):
    response = test_client.post("/rpc", json={"jsonrpc": "2.0", "method": "delegate_task", "params": {}, "id": 2})
    assert response.status_code == 400
    res = response.json()
    assert res["error"]["code"] == -32602

def test_delegate_task_success(test_client, mock_planner):
    response = test_client.post("/rpc", json={
        "jsonrpc": "2.0",
        "method": "delegate_task",
        "params": {"task": "do something"},
        "id": 3
    })
    assert response.status_code == 200
    res = response.json()
    assert res["result"]["status"] == "accepted"
    assert res["result"]["task"] == "do something"
    mock_planner.generate_plan.assert_called_once()

def test_delegate_task_invalid_params_type(test_client):
    response = test_client.post("/rpc", json={"jsonrpc": "2.0", "method": "delegate_task", "params": ["task"], "id": 4})
    assert response.status_code == 400
    res = response.json()
    assert res["error"]["code"] == -32602

def test_notification(test_client, mock_planner):
    response = test_client.post("/rpc", json={
        "jsonrpc": "2.0",
        "method": "delegate_task",
        "params": {"task": "do something silent"}
    })
    assert response.status_code == 204
    mock_planner.generate_plan.assert_called_once()
