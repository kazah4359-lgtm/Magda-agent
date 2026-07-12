import pytest
from magda_agent.integration.mcp_client_auth_v3 import MCPClientAuthV3

def test_mcp_client_auth_no_credentials():
    auth = MCPClientAuthV3()
    headers = auth.get_auth_headers()
    assert headers == {}

def test_mcp_client_auth_with_api_key():
    auth = MCPClientAuthV3(api_key="test_api_key")
    headers = auth.get_auth_headers()
    assert headers == {"x-api-key": "test_api_key"}

def test_mcp_client_auth_with_oauth_token():
    auth = MCPClientAuthV3(oauth_token="test_oauth_token")
    headers = auth.get_auth_headers()
    assert headers == {"Authorization": "Bearer test_oauth_token"}

def test_mcp_client_auth_priority():
    auth = MCPClientAuthV3(api_key="test_api_key", oauth_token="test_oauth_token")
    headers = auth.get_auth_headers()
    assert headers == {"Authorization": "Bearer test_oauth_token"}

def test_mcp_client_auth_setters():
    auth = MCPClientAuthV3()
    auth.set_api_key("new_api_key")
    headers = auth.get_auth_headers()
    assert headers == {"x-api-key": "new_api_key"}

    auth.set_oauth_token("new_oauth_token")
    headers = auth.get_auth_headers()
    assert headers == {"Authorization": "Bearer new_oauth_token"}
