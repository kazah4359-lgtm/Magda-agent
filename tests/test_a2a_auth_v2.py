"""Tests for A2A Service Authentication V2 with Token-Based Peer Authentication."""

import pytest
from unittest.mock import patch, MagicMock
from magda_agent.integration.a2a_auth_v2 import A2AAuthV2


def test_a2a_auth_v2_authenticate_success():
    """Test successful mTLS authentication."""
    auth = A2AAuthV2(cert_path="cert.pem", key_path="key.pem")

    with patch("ssl.create_default_context") as mock_ssl:
        mock_context = MagicMock()
        mock_ssl.return_value = mock_context

        assert auth.authenticate() is True
        assert auth.is_authenticated is True
        mock_context.load_cert_chain.assert_called_once_with(certfile="cert.pem", keyfile="key.pem")


def test_a2a_auth_v2_authenticate_failure_empty_paths():
    """Test failed mTLS authentication due to empty paths."""
    auth = A2AAuthV2(cert_path="", key_path="")
    assert auth.authenticate() is False
    assert auth.is_authenticated is False


def test_a2a_auth_v2_authenticate_failure_ssl_error():
    """Test failed mTLS authentication due to SSL error."""
    auth = A2AAuthV2(cert_path="cert.pem", key_path="key.pem")

    with patch("ssl.create_default_context") as mock_ssl:
        mock_context = MagicMock()
        mock_context.load_cert_chain.side_effect = Exception("SSL Error")
        mock_ssl.return_value = mock_context

        assert auth.authenticate() is False
        assert auth.is_authenticated is False


def test_register_and_verify_peer_token():
    """Test registering and verifying peer authentication tokens."""
    auth = A2AAuthV2()
    peer_id = "agent_alpha"
    token = "secret_peer_token_123"

    auth.register_peer_token(peer_id, token)

    assert auth.verify_peer_token(peer_id, token) is True
    assert auth.verify_peer_token(peer_id, "wrong_token") is False
    assert auth.verify_peer_token("unknown_peer", token) is False


def test_generate_peer_token():
    """Test generating peer token and verifying it."""
    auth = A2AAuthV2()
    peer_id = "agent_beta"

    generated_token = auth.generate_peer_token(peer_id)

    assert generated_token.startswith("a2a_peer_")
    assert len(generated_token) > 20
    assert auth.verify_peer_token(peer_id, generated_token) is True


def test_revoke_peer_token():
    """Test token revocation."""
    auth = A2AAuthV2()
    peer_id = "agent_gamma"
    token = auth.generate_peer_token(peer_id)

    assert auth.verify_peer_token(peer_id, token) is True
    assert auth.revoke_peer_token(peer_id) is True
    assert auth.verify_peer_token(peer_id, token) is False
    assert auth.revoke_peer_token(peer_id) is False


def test_authenticate_peer_card():
    """Test authenticating peer using agent card payload."""
    auth = A2AAuthV2()
    peer_id = "agent_delta"
    token = auth.generate_peer_token(peer_id)

    peer_card = {"agent_id": peer_id, "name": "Delta Agent", "capabilities": ["tool1"]}

    assert auth.authenticate_peer(peer_card, token) is True
    assert auth.authenticate_peer(peer_card, "invalid_token") is False
    assert auth.authenticate_peer({}, token) is False


def test_verify_request_headers():
    """Test verifying incoming A2A request payload with HTTP headers."""
    auth = A2AAuthV2()
    peer_id = "agent_epsilon"
    token = auth.generate_peer_token(peer_id)

    request_valid = {
        "headers": {
            "X-Peer-ID": peer_id,
            "Authorization": f"Bearer {token}",
        }
    }
    request_invalid_token = {
        "headers": {
            "X-Peer-ID": peer_id,
            "Authorization": "Bearer bad_token",
        }
    }
    request_missing_header = {
        "headers": {
            "X-Peer-ID": peer_id,
        }
    }

    assert auth.verify_request(request_valid) is True
    assert auth.verify_request(request_invalid_token) is False
    assert auth.verify_request(request_missing_header) is False


def test_verify_request_params_or_top_level():
    """Test verifying incoming A2A request payload with top-level or params dict."""
    auth = A2AAuthV2()
    peer_id = "agent_zeta"
    token = auth.generate_peer_token(peer_id)

    top_level_request = {"peer_id": peer_id, "token": token}
    params_request = {"params": {"agent_id": peer_id, "auth_token": token}}

    assert auth.verify_request(top_level_request) is True
    assert auth.verify_request(params_request) is True
    assert auth.verify_request({"peer_id": peer_id, "token": "invalid"}) is False


def test_verify_request_edge_cases():
    """Test verify_request with invalid input types."""
    auth = A2AAuthV2()
    assert auth.verify_request("not_a_dict") is False
    assert auth.verify_request({}) is False


def test_empty_input_handling():
    """Test input validations for register, generate, and verify peer token methods."""
    auth = A2AAuthV2()

    with pytest.raises(ValueError):
        auth.register_peer_token("", "token")

    with pytest.raises(ValueError):
        auth.register_peer_token("peer", "")

    with pytest.raises(ValueError):
        auth.generate_peer_token("")

    assert auth.verify_peer_token("", "token") is False
    assert auth.verify_peer_token("peer", "") is False
