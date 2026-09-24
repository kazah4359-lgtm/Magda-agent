import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from magda_agent.integration.a2a_cards import AgentCardV4, A2ADiscoveryV4
from magda_agent.integration.a2a_delegator_v4 import A2ADelegatorV4

logger = logging.getLogger(__name__)


@dataclass
class TaskEvaluationResult:
    """
    Represents the complexity evaluation result for a given task payload.
    """
    task_id: str
    complexity_score: float  # 0.0 to 10.0 scale
    is_complex: bool
    required_capabilities: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class A2APeerRouterV2:
    """
    Evaluates task complexity and routes/delegates tasks to specialized peer agents
    using Agent Cards (V4) for capability matching and A2ADelegatorV4.
    """

    def __init__(
        self,
        discovery: Optional[A2ADiscoveryV4] = None,
        delegator: Optional[A2ADelegatorV4] = None,
        complexity_threshold: float = 5.0
    ) -> None:
        """
        Initializes A2APeerRouterV2.

        Args:
            discovery: Optional A2ADiscoveryV4 instance for discovering peer agents.
            delegator: Optional A2ADelegatorV4 instance for executing HTTP task delegation.
            complexity_threshold: Score threshold (0-10) above which tasks are considered complex/delegatable.
        """
        self.discovery = discovery
        self.delegator = delegator or A2ADelegatorV4()
        self.complexity_threshold = complexity_threshold

    def evaluate_complexity(self, task_payload: Dict[str, Any]) -> TaskEvaluationResult:
        """
        Evaluates task complexity based on task payload parameters, length, subtasks, or explicit risk.

        Args:
            task_payload: Dictionary containing task details (e.g. 'id', 'description', 'subtasks', 'risk').

        Returns:
            TaskEvaluationResult containing complexity score, whether it's complex, and required capabilities.
        """
        task_id = str(task_payload.get("id", "unknown"))
        description = str(task_payload.get("description", ""))
        subtasks = task_payload.get("subtasks", [])
        risk = str(task_payload.get("risk", "low")).lower()
        capabilities = task_payload.get("required_capabilities") or task_payload.get("capabilities") or []

        if isinstance(capabilities, str):
            capabilities = [capabilities]
        else:
            capabilities = list(capabilities)

        # Baseline score calculation
        score = 1.0

        # Description length factor
        if len(description) > 500:
            score += 3.0
        elif len(description) > 200:
            score += 1.5

        # Subtasks factor
        if isinstance(subtasks, list) and subtasks:
            score += min(len(subtasks) * 1.5, 4.0)

        # Risk factor
        if risk in ("high", "critical"):
            score += 3.0
        elif risk == "medium":
            score += 1.5

        # Explicit complexity flag or level override
        if task_payload.get("is_complex") is True or task_payload.get("complexity") == "high":
            score = max(score, 8.0)

        score = min(max(score, 0.0), 10.0)
        is_complex = score >= self.complexity_threshold

        logger.info(f"Task {task_id} evaluated with complexity score {score:.2f} (is_complex={is_complex})")

        return TaskEvaluationResult(
            task_id=task_id,
            complexity_score=score,
            is_complex=is_complex,
            required_capabilities=capabilities,
            metadata={"risk": risk, "subtask_count": len(subtasks) if isinstance(subtasks, list) else 0}
        )

    def find_matching_peer(self, required_capabilities: List[str]) -> Optional[AgentCardV4]:
        """
        Matches required capabilities against discovered peer agents.

        Args:
            required_capabilities: List of capability strings required by the task.

        Returns:
            Matching AgentCardV4 if found, otherwise None.
        """
        if not self.discovery or not required_capabilities:
            return None

        # Look for agents that match all or any required capability
        all_agents = self.discovery.get_all_agents()
        for agent in all_agents:
            if agent.status != "inactive" and agent.matches_any_capability(required_capabilities):
                return agent

        return None

    async def route_and_delegate(
        self,
        task_payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates task complexity, matches peer capabilities, and delegates task if complex or matching peer found.

        Args:
            task_payload: Task payload to evaluate and route.
            headers: Optional headers for delegation request.

        Returns:
            Result dict from delegation if delegated, or local handling indicator if processed locally.

        Raises:
            ValueError: If task requires peer delegation but no matching peer or endpoint is found.
        """
        eval_result = self.evaluate_complexity(task_payload)

        # Match peer based on capabilities
        matched_peer = self.find_matching_peer(eval_result.required_capabilities)

        if not eval_result.is_complex and not matched_peer:
            logger.info(f"Task {eval_result.task_id} is simple and has no explicit peer requirement. Handling locally.")
            return {
                "status": "local_execution",
                "task_id": eval_result.task_id,
                "complexity_score": eval_result.complexity_score,
                "message": "Task processed locally"
            }

        if not matched_peer and eval_result.required_capabilities:
            # Check if any capability is required but no matching peer
            logger.warning(f"No peer found with required capabilities: {eval_result.required_capabilities}")
            raise ValueError(f"No peer agent found matching required capabilities: {eval_result.required_capabilities}")

        if not matched_peer:
            # Task is complex, but no specific peer found
            all_agents = self.discovery.get_all_agents() if self.discovery else []
            if all_agents:
                matched_peer = all_agents[0]
            else:
                raise ValueError("Task is complex but no peer agents are available for delegation")

        # Select endpoint (rpc or mcp)
        endpoint = matched_peer.endpoints.get("rpc") or matched_peer.endpoints.get("mcp") or matched_peer.endpoints.get("main")
        if not endpoint:
            raise ValueError(f"Matched peer agent {matched_peer.agent_id} has no valid endpoint configured")

        logger.info(f"Delegating task {eval_result.task_id} to peer {matched_peer.agent_id} at {endpoint}")

        delegation_payload = {
            "task_id": eval_result.task_id,
            "payload": task_payload,
            "complexity_score": eval_result.complexity_score,
            "delegated_by": "A2APeerRouterV2"
        }

        response = await self.delegator.delegate_task(
            peer_endpoint=endpoint,
            payload=delegation_payload,
            headers=headers
        )

        return {
            "status": "delegated",
            "task_id": eval_result.task_id,
            "peer_id": matched_peer.agent_id,
            "peer_name": matched_peer.name,
            "result": response
        }
