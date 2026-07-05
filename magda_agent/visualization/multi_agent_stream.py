import json
import logging
from typing import Dict, Any

from magda_agent.architecture.agent_teams_v3 import AgentTeamManagerV3

logger = logging.getLogger(__name__)

class MultiAgentStreamServer:
    """
    Formats multi-agent isolation metrics and active agent states into a structured
    JSON payload suitable for streaming to a live OpenClaw-inspired Canvas UI.
    """

    def __init__(self, team_manager: AgentTeamManagerV3) -> None:
        """
        Initialize the stream server formatter.

        Args:
            team_manager (AgentTeamManagerV3): The team manager coordinating agents.
        """
        self.team_manager = team_manager

    def get_formatted_state(self) -> Dict[str, Any]:
        """
        Retrieves active agents and their isolated worktrees to broadcast status.

        Returns:
            Dict[str, Any]: A dictionary representing the active multi-agent cluster.
        """
        state: Dict[str, Any] = {
            "agents": [],
            "isolation_metrics": {
                "active_worktrees": 0,
                "base_dir": ""
            }
        }

        try:
            isolation = self.team_manager.isolation_manager

            if isolation:
                state["isolation_metrics"]["base_dir"] = isolation.base_dir

                for agent_id, env_path in isolation.active_worktrees.items():
                    agent_info = {
                        "agent_id": agent_id,
                        "worktree": env_path,
                        "status": "active"
                    }
                    state["agents"].append(agent_info)

                state["isolation_metrics"]["active_worktrees"] = len(state["agents"])

        except Exception as e:
            logger.error(f"Error formatting multi-agent canvas state: {e}")
            state["error"] = str(e)

        return state

    def get_state_json(self) -> str:
        """
        Returns the formatted multi-agent state as a JSON string.

        Returns:
            str: JSON-encoded string of the multi-agent state.
        """
        return json.dumps(self.get_formatted_state())
