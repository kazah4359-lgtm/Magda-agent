"""A2A Service Authentication V2 with Token-Based Peer Authentication."""

import logging
import secrets
import ssl
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class A2AAuthV2:
    """Handles mTLS and token-based peer authentication for A2A communication."""

    def __init__(self, cert_path: str = "", key_path: str = ""):
        """Initialize the A2AAuthV2 instance."""
        self.cert_path = cert_path
        self.key_path = key_path
        self.is_authenticated = False
        self.ssl_context: Optional[ssl.SSLContext] = None
        self._peer_tokens: Dict[str, str] = {}

    def authenticate(self) -> bool:
        """Authenticate using mTLS."""
        if not self.cert_path or not self.key_path:
            self.is_authenticated = False
            return False

        try:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(certfile=self.cert_path, keyfile=self.key_path)
            self.is_authenticated = True
            return True
        except Exception as exc:
            logger.warning(f"mTLS authentication error: {exc}")
            self.is_authenticated = False
            return False

    def register_peer_token(self, peer_id: str, token: str) -> None:
        """Register an authentication token for a discovered agent card/peer."""
        if not peer_id or not token:
            raise ValueError("peer_id and token must be non-empty strings.")
        self._peer_tokens[peer_id] = token
        logger.info(f"Registered peer auth token for {peer_id}")

    def generate_peer_token(self, peer_id: str) -> str:
        """Generate and register a secure token for a peer agent."""
        if not peer_id:
            raise ValueError("peer_id must be a non-empty string.")
        token = f"a2a_peer_{secrets.token_hex(24)}"
        self.register_peer_token(peer_id, token)
        return token

    def verify_peer_token(self, peer_id: str, token: str) -> bool:
        """Verify if the provided token matches the registered token for peer_id."""
        if not peer_id or not token:
            return False

        registered_token = self._peer_tokens.get(peer_id)
        if not registered_token:
            logger.warning(f"Peer token verification failed: peer {peer_id} not registered.")
            return False

        is_valid = secrets.compare_digest(registered_token, token)
        if is_valid:
            logger.info(f"Peer token verification succeeded for {peer_id}.")
        else:
            logger.warning(f"Peer token verification failed: invalid token for {peer_id}.")
        return is_valid

    def revoke_peer_token(self, peer_id: str) -> bool:
        """Revoke a peer authentication token."""
        if peer_id in self._peer_tokens:
            del self._peer_tokens[peer_id]
            logger.info(f"Revoked peer token for {peer_id}.")
            return True
        return False

    def authenticate_peer(self, peer_card: Dict[str, Any], auth_token: str) -> bool:
        """Authenticate a peer using its agent card dictionary and token."""
        peer_id = (
            peer_card.get("agent_id")
            or peer_card.get("id")
            or peer_card.get("name")
        )
        if not peer_id:
            logger.warning("Peer card missing agent_id or id field.")
            return False
        return self.verify_peer_token(str(peer_id), auth_token)

    def verify_request(self, request_data: Dict[str, Any]) -> bool:
        """
        Verify incoming A2A request data payload for peer authentication.

        Supports request structure with headers, params, or top-level fields:
        - headers: {"Authorization": "Bearer <token>", "X-Peer-ID": "<peer_id>"}
        - top-level/params: {"peer_id": "...", "token": "..."}
        """
        if not isinstance(request_data, dict):
            return False

        headers = request_data.get("headers", {})
        if isinstance(headers, dict):
            peer_id = (
                headers.get("X-Peer-ID")
                or headers.get("X-Agent-ID")
                or headers.get("x-peer-id")
                or headers.get("x-agent-id")
            )
            auth_header = headers.get("Authorization") or headers.get("authorization")
            token = None
            if auth_header and isinstance(auth_header, str):
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:].strip()
                else:
                    token = auth_header.strip()

            if peer_id and token:
                return self.verify_peer_token(str(peer_id), token)

        # Fall back to top-level or params dictionary inspection
        params = request_data.get("params", {})
        if not isinstance(params, dict):
            params = {}

        peer_id = (
            request_data.get("peer_id")
            or request_data.get("agent_id")
            or params.get("peer_id")
            or params.get("agent_id")
        )
        token = (
            request_data.get("token")
            or request_data.get("auth_token")
            or params.get("token")
            or params.get("auth_token")
        )

        if peer_id and token:
            return self.verify_peer_token(str(peer_id), str(token))

        return False
