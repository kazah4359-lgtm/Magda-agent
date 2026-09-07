import uuid
import time
import logging
import httpx
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)

class A2AWorkflowAuthDelegator:
    """
    Manages active authentication and delegation tokens when passing workflow tasks
    across an A2A mesh network to peer agents.
    """

    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        """
        Initializes the workflow auth delegator.

        Args:
            default_ttl_seconds: Default time-to-live in seconds for issued tokens.
        """
        self.default_ttl_seconds = default_ttl_seconds
        # Maps token string to metadata dictionary
        self._tokens: Dict[str, Dict[str, Any]] = {}

    def issue_token(
        self,
        workflow_id: str,
        task_id: str,
        scopes: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        issuer_id: str = "local_agent",
        target_agent_id: Optional[str] = None,
    ) -> str:
        """
        Issues a new delegation token for a specific workflow task.

        Args:
            workflow_id: ID of the parent workflow.
            task_id: ID of the active task.
            scopes: List of allowed permission scopes (e.g., ["read", "execute"]).
            ttl_seconds: Custom TTL in seconds.
            issuer_id: ID of the issuing agent/node.
            target_agent_id: ID of the intended peer agent receiving the task.

        Returns:
            The generated delegation token string.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        now = time.time()
        token = f"a2a_wftk_{uuid.uuid4().hex}"

        self._tokens[token] = {
            "token": token,
            "workflow_id": workflow_id,
            "task_id": task_id,
            "scopes": set(scopes) if scopes else {"read", "execute"},
            "created_at": now,
            "expires_at": now + ttl,
            "issuer_id": issuer_id,
            "target_agent_id": target_agent_id,
            "parent_token": None,
            "active": True,
        }

        logger.info(
            f"Issued A2A workflow delegation token for workflow={workflow_id}, "
            f"task={task_id}, target={target_agent_id}."
        )
        return token

    def delegate_subtask_token(
        self,
        parent_token: str,
        subtask_id: str,
        subtask_scopes: Optional[List[str]] = None,
        target_agent_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Optional[str]:
        """
        Exchanges or derives a subtask delegation token from a parent workflow token.

        Args:
            parent_token: The original active workflow token.
            subtask_id: ID of the subtask.
            subtask_scopes: Scopes for the subtask (must be a subset of parent scopes).
            target_agent_id: Intended peer receiving the delegated subtask.
            ttl_seconds: Optional TTL for the subtask token (cannot exceed parent expiration).

        Returns:
            New derived delegation token string, or None if parent token is invalid/expired.
        """
        parent_meta = self.get_token_meta(parent_token)
        if not parent_meta:
            logger.warning("Subtask delegation failed: Parent token invalid or expired.")
            return None

        # Scope validation: requested subtask scopes must be subset of parent scopes
        requested = set(subtask_scopes) if subtask_scopes else parent_meta["scopes"]
        if not requested.issubset(parent_meta["scopes"]):
            logger.warning(
                f"Subtask delegation failed: Scopes {requested - parent_meta['scopes']} "
                "not allowed by parent token."
            )
            return None

        now = time.time()
        parent_expires = parent_meta["expires_at"]
        max_ttl = max(0, int(parent_expires - now))

        if max_ttl <= 0:
            logger.warning("Subtask delegation failed: Parent token has expired.")
            return None

        token_ttl = min(ttl_seconds, max_ttl) if ttl_seconds is not None else max_ttl
        token = f"a2a_subtk_{uuid.uuid4().hex}"

        self._tokens[token] = {
            "token": token,
            "workflow_id": parent_meta["workflow_id"],
            "task_id": subtask_id,
            "scopes": requested,
            "created_at": now,
            "expires_at": now + token_ttl,
            "issuer_id": parent_meta.get("target_agent_id") or "local_agent",
            "target_agent_id": target_agent_id,
            "parent_token": parent_token,
            "active": True,
        }

        logger.info(f"Delegated subtask token derived from parent token for subtask={subtask_id}.")
        return token

    def validate_token(
        self,
        token: str,
        required_scope: Optional[str] = None,
        expected_workflow_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Validates an incoming delegation token for active status, expiration, scope, and optional workflow/agent match.

        Args:
            token: The token to validate.
            required_scope: Mandatory scope required for operation.
            expected_workflow_id: Optional expected workflow ID.
            agent_id: Optional agent ID verifying receipt.

        Returns:
            True if valid, False otherwise.
        """
        meta = self._tokens.get(token)
        if not meta or not meta["active"]:
            logger.warning("Token validation failed: Token not found or revoked.")
            return False

        if time.time() > meta["expires_at"]:
            logger.warning("Token validation failed: Token expired.")
            meta["active"] = False
            return False

        if required_scope and required_scope not in meta["scopes"]:
            logger.warning(f"Token validation failed: Required scope '{required_scope}' missing.")
            return False

        if expected_workflow_id and meta["workflow_id"] != expected_workflow_id:
            logger.warning(f"Token validation failed: Workflow mismatch ({meta['workflow_id']} vs {expected_workflow_id}).")
            return False

        if agent_id and meta["target_agent_id"] and meta["target_agent_id"] != agent_id:
            logger.warning(f"Token validation failed: Target agent mismatch ({meta['target_agent_id']} vs {agent_id}).")
            return False

        return True

    def revoke_token(self, token: str) -> bool:
        """
        Revokes a delegation token.

        Args:
            token: Token string to revoke.

        Returns:
            True if token was found and revoked, False otherwise.
        """
        if token in self._tokens:
            self._tokens[token]["active"] = False
            logger.info("A2A delegation token successfully revoked.")
            return True
        return False

    def get_token_meta(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Gets token metadata if active and not expired.

        Args:
            token: The token string.

        Returns:
            Dictionary of token metadata or None.
        """
        meta = self._tokens.get(token)
        if not meta or not meta["active"]:
            return None
        if time.time() > meta["expires_at"]:
            meta["active"] = False
            return None
        return dict(meta)

    async def pass_task_over_mesh(
        self,
        peer_endpoint: str,
        token: str,
        task_payload: Dict[str, Any],
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Passes a delegated workflow task and token to a peer agent over the mesh network via HTTP JSON-RPC.

        Args:
            peer_endpoint: HTTP endpoint URL of peer agent.
            token: Delegation token string.
            task_payload: Task parameters and payload.
            timeout: HTTP request timeout.

        Returns:
            Response dictionary with outcome status and data.
        """
        if not self.validate_token(token):
            return {
                "success": False,
                "error": "Invalid or expired delegation token.",
                "code": 401,
            }

        meta = self.get_token_meta(token)
        json_payload = {
            "jsonrpc": "2.0",
            "method": "a2a_receive_workflow_task",
            "params": {
                "delegation_token": token,
                "workflow_id": meta["workflow_id"] if meta else None,
                "task_id": meta["task_id"] if meta else None,
                "payload": task_payload,
            },
            "id": str(uuid.uuid4()),
        }

        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(peer_endpoint, json=json_payload, timeout=timeout)
                res.raise_for_status()
                return {
                    "success": True,
                    "response": res.json(),
                    "code": res.status_code,
                }
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error passing task over mesh to {peer_endpoint}: {e}")
            return {"success": False, "error": f"HTTP status {e.response.status_code}", "code": e.response.status_code}
        except httpx.RequestError as e:
            logger.error(f"Network error passing task over mesh to {peer_endpoint}: {e}")
            return {"success": False, "error": f"Network request error: {str(e)}", "code": 503}
        except Exception as e:
            logger.error(f"Unexpected error passing task over mesh to {peer_endpoint}: {e}")
            return {"success": False, "error": str(e), "code": 500}
