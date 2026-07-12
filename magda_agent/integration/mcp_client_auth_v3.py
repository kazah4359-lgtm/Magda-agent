from typing import Dict, Optional
import os

class MCPClientAuthV3:
    """
    Manages authentication configuration for the MCP Client.
    Supports API keys and OAuth tokens for connecting to remote MCP servers.
    Inspired by MCP standards trend.
    """
    def __init__(self, api_key: Optional[str] = None, oauth_token: Optional[str] = None) -> None:
        """
        Initializes the MCPClientAuthV3 with optional authentication credentials.

        Args:
            api_key: An optional API key string.
            oauth_token: An optional OAuth bearer token string.
        """
        self.api_key = api_key
        self.oauth_token = oauth_token

    def set_api_key(self, api_key: str) -> None:
        """
        Sets the API key for authentication.

        Args:
            api_key: The API key to use.
        """
        self.api_key = api_key

    def set_oauth_token(self, token: str) -> None:
        """
        Sets the OAuth token for authentication.

        Args:
            token: The OAuth bearer token to use.
        """
        self.oauth_token = token

    def get_auth_headers(self) -> Dict[str, str]:
        """
        Generates the required HTTP headers for authentication based on configured credentials.
        Prioritizes OAuth token over API key if both are set.

        Returns:
            Dict[str, str]: A dictionary of HTTP headers.
        """
        headers: Dict[str, str] = {}
        if self.oauth_token:
            headers["Authorization"] = f"Bearer {self.oauth_token}"
        elif self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
