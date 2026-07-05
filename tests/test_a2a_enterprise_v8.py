import pytest
from unittest.mock import patch, MagicMock

from magda_agent.integration.a2a_enterprise_v8 import A2AEnterpriseClientV8
from magda_agent.integration.a2a_auth import A2AAuthTokenDelegation


def test_delegate_task_securely_logs_and_manages_token() -> None:
    """Tests that delegate_task securely logs events and handles tokens."""
    # Setup
    auth_delegation = A2AAuthTokenDelegation()
    client = A2AEnterpriseClientV8(auth_delegation=auth_delegation)

    target_agent_id = "agent-xyz"
    capability = "compute_metrics"
    context = {"param1": "value1"}

    with patch("magda_agent.integration.a2a_enterprise_v8.A2AAuthTokenDelegation.generate_token") as mock_generate_token, \
         patch("magda_agent.integration.a2a_enterprise_v8.A2AAuthTokenDelegation.revoke_token") as mock_revoke_token, \
         patch("magda_agent.integration.a2a_enterprise_v8.A2ATracingV4.securely_log_delegation_event") as mock_log_event:

        mock_generate_token.return_value = "mock_token_123"
        mock_log_event.return_value = "trace_abc123"

        # Execute
        trace_id = client.delegate_task(target_agent_id, capability, context)

        # Assert
        assert trace_id == "trace_abc123"

        # Verify token generation was called
        mock_generate_token.assert_called_once()

        # Verify secure logging was called with the correct parameters, including the token
        mock_log_event.assert_called_once_with(
            target_agent_id=target_agent_id,
            capability=capability,
            context=context,
            token="mock_token_123"
        )

        # Verify token was revoked after
        mock_revoke_token.assert_called_once_with("mock_token_123")


def test_delegate_task_with_real_auth_manager() -> None:
    """Tests delegate_task using the real auth manager but mocking the tracer."""
    # Integration style test using the real auth manager but mocking the tracer
    auth_delegation = A2AAuthTokenDelegation()
    client = A2AEnterpriseClientV8(auth_delegation=auth_delegation)

    with patch("magda_agent.integration.a2a_enterprise_v8.A2ATracingV4.securely_log_delegation_event") as mock_log_event:
        mock_log_event.return_value = "trace_real_auth"

        # Execute
        trace_id = client.delegate_task("agent-1", "test_cap", {})

        # Assert trace id returned
        assert trace_id == "trace_real_auth"

        # Assert the logger was called with a real token (starts with a2a_auth_)
        called_args = mock_log_event.call_args[1]
        token_used = called_args["token"]
        assert token_used.startswith("a2a_auth_")

        # The token should have been revoked, so it should not be active
        assert auth_delegation.validate_token(token_used) is False
