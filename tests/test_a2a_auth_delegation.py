import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.integration.a2a_auth_delegation import A2AWorkflowAuthDelegator


def test_issue_and_validate_token():
    delegator = A2AWorkflowAuthDelegator(default_ttl_seconds=100)
    token = delegator.issue_token(
        workflow_id="wf_123",
        task_id="task_001",
        scopes=["read", "execute", "write"],
        target_agent_id="agent_beta",
    )

    assert token.startswith("a2a_wftk_")
    assert delegator.validate_token(token) is True
    assert delegator.validate_token(token, required_scope="read") is True
    assert delegator.validate_token(token, required_scope="admin") is False
    assert delegator.validate_token(token, expected_workflow_id="wf_123") is True
    assert delegator.validate_token(token, expected_workflow_id="wf_other") is False
    assert delegator.validate_token(token, agent_id="agent_beta") is True
    assert delegator.validate_token(token, agent_id="agent_gamma") is False


def test_token_expiration():
    delegator = A2AWorkflowAuthDelegator()
    token = delegator.issue_token(
        workflow_id="wf_exp",
        task_id="task_exp",
        ttl_seconds=-10,  # Already expired
    )

    assert delegator.validate_token(token) is False
    assert delegator.get_token_meta(token) is None


def test_token_revocation():
    delegator = A2AWorkflowAuthDelegator()
    token = delegator.issue_token(workflow_id="wf_1", task_id="task_1")

    assert delegator.validate_token(token) is True
    assert delegator.revoke_token(token) is True
    assert delegator.validate_token(token) is False
    assert delegator.revoke_token("non_existent_token") is False


def test_subtask_delegation_scopes():
    delegator = A2AWorkflowAuthDelegator()
    parent_token = delegator.issue_token(
        workflow_id="wf_parent",
        task_id="task_parent",
        scopes=["read", "execute"],
    )

    # Valid subtask delegation with subset of scopes
    sub_token = delegator.delegate_subtask_token(
        parent_token=parent_token,
        subtask_id="subtask_1",
        subtask_scopes=["read"],
        target_agent_id="agent_sub",
    )
    assert sub_token is not None
    assert sub_token.startswith("a2a_subtk_")
    assert delegator.validate_token(sub_token, required_scope="read") is True
    assert delegator.validate_token(sub_token, required_scope="execute") is False

    # Invalid subtask delegation requesting unauthorized scope
    invalid_sub_token = delegator.delegate_subtask_token(
        parent_token=parent_token,
        subtask_id="subtask_2",
        subtask_scopes=["read", "admin_write"],
    )
    assert invalid_sub_token is None


def test_subtask_delegation_with_invalid_or_expired_parent():
    delegator = A2AWorkflowAuthDelegator()

    # Invalid parent token
    assert delegator.delegate_subtask_token("invalid_parent", "sub_1") is None

    # Expired parent token
    expired_parent = delegator.issue_token("wf_1", "task_1", ttl_seconds=-5)
    assert delegator.delegate_subtask_token(expired_parent, "sub_2") is None


@pytest.mark.asyncio
async def test_pass_task_over_mesh_success():
    delegator = A2AWorkflowAuthDelegator()
    token = delegator.issue_token(
        workflow_id="wf_mesh",
        task_id="task_mesh",
        scopes=["execute"],
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"jsonrpc": "2.0", "result": {"status": "accepted"}, "id": "1"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        res = await delegator.pass_task_over_mesh(
            peer_endpoint="http://peer-agent:8080/rpc",
            token=token,
            task_payload={"action": "process_data", "input": [1, 2, 3]},
        )

        assert res["success"] is True
        assert res["code"] == 200
        assert res["response"]["result"]["status"] == "accepted"
        mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_pass_task_over_mesh_invalid_token():
    delegator = A2AWorkflowAuthDelegator()

    res = await delegator.pass_task_over_mesh(
        peer_endpoint="http://peer-agent:8080/rpc",
        token="invalid_token",
        task_payload={"action": "test"},
    )

    assert res["success"] is False
    assert res["code"] == 401
    assert "Invalid or expired" in res["error"]
