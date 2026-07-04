import uuid
import logging
from typing import Set

class A2AAuthTokenDelegation:
    """
    Manages generating and validating secure authentication tokens for A2A communication.
    Provides token auth flow to the Agent-to-Agent protocol integration for secure peer delegation.
    """
    def __init__(self) -> None:
        """Initializes the auth token delegation manager with an empty set of active tokens."""
        self._active_tokens: Set[str] = set()

    def generate_token(self) -> str:
        """
        Generates a secure auth token.

        Returns:
            str: The generated token prefixed with 'a2a_auth_'.
        """
        token = f"a2a_auth_{uuid.uuid4().hex}"
        self._active_tokens.add(token)
        logging.info(f"Generated new A2A auth token (redacted: {token[:12]}...)")
        return token

    def validate_token(self, token: str) -> bool:
        """
        Validates the provided auth token.

        Args:
            token: The token to validate.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        is_valid = token in self._active_tokens
        if is_valid:
            logging.info(f"Token validation successful for token: {token[:12]}...")
        else:
            logging.warning(f"Token validation failed for token: {token[:12]}...")
        return is_valid

    def revoke_token(self, token: str) -> bool:
        """
        Revokes a specific token so it can no longer be used.

        Args:
            token: The token to revoke.

        Returns:
            bool: True if the token was revoked, False if it was not found.
        """
        if token in self._active_tokens:
            self._active_tokens.remove(token)
            logging.info(f"Revoked token: {token[:12]}...")
            return True
        logging.warning(f"Attempted to revoke unknown token: {token[:12]}...")
        return False
