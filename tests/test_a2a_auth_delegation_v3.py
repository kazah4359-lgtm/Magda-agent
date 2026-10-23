import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from magda_agent.integration.a2a_auth_delegation_v3 import (
    A2AWorkflowAuthDelegatorV3,
    A2AAuthDelegationManagerV3,
)


def test_alias_and_init():
    assert A2AAuthDelegationManagerV3 is A2AWorkflowAuthDelegatorV3
    delegator = A2AWorkflowAuthDelegatorV3(default_ttl_seconds=500)
    assert delegator.default_ttl_seconds == 500


def test_issue_and_validate_token_v3():
    delegator = A2AWorkflowAuthDelegatorV3(default_ttl_seconds=100)
    token = delegator.issue_token(
        workflow_id="wf_123",
        task_id="task_001",
        scopes=["read", "execute", "write"],
        target_agent_id="agent_beta",
        metadata={"priority": "high"},
    )

    assert token.startswith("a2a_wftk_v3_")
    assert delegator.validate_token(token) is True
    assert delegator.validate_token(token, required_scope="read") is True
    assert delegator.validate_token(token, required_scope="admin") is False
    assert delegator.validate_token(token, expected_workflow_id="wf_123") is True
    assert delegator.validate_token(token, expected_workflow_id="wf_other") is False
    assert delegator.validate_token(token, agent_id="agent_beta") is True
    assert delegator.validate_token(token, agent_id="agent_gamma") is False

    meta = delegator.get_token_meta(token)
    assert meta is not None
    assert meta["workflow_id"] == "wf_123"
    assert meta["metadata"] == {"priority": "high"}


def test_list_active_tokens_and_cleanup():
    delegator = A2AWorkflowAuthDelegatorV3()
    t1 = delegator.issue_token("wf_a", "task_1", ttl_seconds=300)
    t2 = delegator.issue_token("wf_a", "task_2", ttl_seconds=-10)  # Already expired
    t3 = delegator.issue_token("wf_b", "task_3", ttl_seconds=300)

    active_all = delegator.list_active_tokens()
    active_tokens = [item["token"] for item in active_all]
    assert t1 in active_tokens
    assert t2 not in active_tokens
    assert t3 in active_tokens

    active_wf_a = delegator.list_active_tokens(workflow_id="wf_a")
    assert len(active_wf_a) == 1
    assert active_wf_a[0]["token"] == t1

    purged = delegator.cleanup_expired_tokens()
    assert purged >= 1


def test_token_revocation_by_workflow():
    delegator = A2AWorkflowAuthDelegatorV3()
    t1 = delegator.issue_token("wf_cancel", "task_1")
    t2 = delegator.issue_token("wf_cancel", "task_2")
    t3 = delegator.issue_token("wf_keep", "task_3")

    assert delegator.validate_token(t1) is True
    assert delegator.validate_token(t2) is True

    revoked_count = delegator.revoke_workflow_tokens("wf_cancel")
    assert revoked_count == 2

    assert delegator.validate_token(t1) is False
    assert delegator.validate_token(t2) is False
    assert delegator.validate_token(t3) is True


def test_subtask_delegation_scopes_v3():
    delegator = A2AWorkflowAuthDelegatorV3()
    parent_token = delegator.issue_token(
        workflow_id="wf_parent",
        task_id="task_parent",
        scopes=["read", "execute"],
        metadata={"tenant": "corp"},
    )

    sub_token = delegator.delegate_subtask_token(
        parent_token=parent_token,
        subtask_id="subtask_1",
        subtask_scopes=["read"],
        target_agent_id="agent_sub",
    )
    assert sub_token is not None
    assert sub_token.startswith("a2a_subtk_v3_")
    assert delegator.validate_token(sub_token, required_scope="read") is True
    assert delegator.validate_token(sub_token, required_scope="execute") is False

    meta = delegator.get_token_meta(sub_token)
    assert meta["metadata"] == {"tenant": "corp"}

    invalid_sub_token = delegator.delegate_subtask_token(
        parent_token=parent_token,
        subtask_id="subtask_2",
        subtask_scopes=["read", "admin_write"],
    )
    assert invalid_sub_token is None


@pytest.mark.asyncio
async def test_verify_peer_token_exchange_success():
    delegator = A2AWorkflowAuthDelegatorV3()
    token = delegator.issue_token("wf_peer", "task_peer")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {"valid": True, "issuer": "local_agent"},
        "id": "1",
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        res = await delegator.verify_peer_token_exchange(
            peer_endpoint="http://peer-agent:8080/verify",
            token=token,
            expected_issuer="local_agent",
        )

        assert res["success"] is True
        assert res["verified"] is True
        assert res["code"] == 200
        mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_verify_peer_token_exchange_http_error():
    delegator = A2AWorkflowAuthDelegatorV3()
    token = delegator.issue_token("wf_peer", "task_peer")

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)

    mock_client = AsyncMock()
    mock_client.post.side_effect = mock_error

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        res = await delegator.verify_peer_token_exchange(
            peer_endpoint="http://peer-agent:8080/verify",
            token=token,
        )

        assert res["success"] is False
        assert res["verified"] is False
        assert res["code"] == 500


@pytest.mark.asyncio
async def test_pass_task_over_mesh_v3_success():
    delegator = A2AWorkflowAuthDelegatorV3()
    token = delegator.issue_token(
        workflow_id="wf_mesh_v3",
        task_id="task_mesh_v3",
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
