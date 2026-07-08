import asyncio
import logging
import time
from typing import Dict, List, Optional

class SubagentHeartbeatMonitor:
    """
    SubagentHeartbeatMonitor tracks heartbeat signals from Subagents (e.g. those spawned
    by SubagentSpawnerV2 in isolated git worktrees) to detect deadlocks or crashes.
    """
    def __init__(self) -> None:
        """
        Initializes the SubagentHeartbeatMonitor.
        """
        self._heartbeats: Dict[str, float] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._is_running = False

    def register_agent(self, agent_id: str) -> None:
        """
        Registers an agent to be monitored.

        Args:
            agent_id: The unique identifier of the subagent.
        """
        self._heartbeats[agent_id] = time.time()
        logging.info(f"Registered agent {agent_id} for heartbeat monitoring.")

    def record_heartbeat(self, agent_id: str) -> None:
        """
        Records a heartbeat signal from a registered agent.

        Args:
            agent_id: The unique identifier of the subagent.
        """
        if agent_id in self._heartbeats:
            self._heartbeats[agent_id] = time.time()
            logging.debug(f"Recorded heartbeat for agent {agent_id}.")
        else:
            logging.warning(f"Received heartbeat for unregistered agent {agent_id}.")

    def check_deadlocks(self, timeout_seconds: float) -> List[str]:
        """
        Checks for agents that have not sent a heartbeat within the timeout window.

        Args:
            timeout_seconds: The maximum allowed time (in seconds) between heartbeats.

        Returns:
            A list of agent IDs that have exceeded the timeout.
        """
        deadlocked_agents = []
        current_time = time.time()
        for agent_id, last_heartbeat in list(self._heartbeats.items()):
            if current_time - last_heartbeat > timeout_seconds:
                deadlocked_agents.append(agent_id)

        if deadlocked_agents:
            logging.warning(f"Detected deadlocked agents: {deadlocked_agents}")

        return deadlocked_agents

    async def _monitor_loop(self, timeout_seconds: float, check_interval: float) -> None:
        """
        The background loop that periodically checks for deadlocks.
        """
        self._is_running = True
        try:
            while self._is_running:
                await asyncio.sleep(check_interval)
                deadlocked = self.check_deadlocks(timeout_seconds)
                if deadlocked:
                    self._handle_deadlocked_agents(deadlocked)
        except asyncio.CancelledError:
            self._is_running = False
            logging.info("Heartbeat monitor loop cancelled.")

    def _handle_deadlocked_agents(self, deadlocked_agents: List[str]) -> None:
        """
        Handles alerting for deadlocked agents.

        Args:
            deadlocked_agents: A list of agent IDs that are deadlocked.
        """
        # In a real system, this could trigger recovery, cleanup, or alerts.
        # For this requirement, triggering an alert via logging is sufficient.
        for agent_id in deadlocked_agents:
            logging.error(f"ALERT: Subagent {agent_id} failed to send a heartbeat within the timeout window.")

    def start_monitor(self, timeout_seconds: float = 30.0, check_interval: float = 5.0) -> None:
        """
        Starts the background monitoring task.

        Args:
            timeout_seconds: Maximum time without a heartbeat before alerting.
            check_interval: How often to check the heartbeats.
        """
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop(timeout_seconds, check_interval))
            logging.info(f"Started subagent heartbeat monitor (timeout={timeout_seconds}s, interval={check_interval}s).")

    async def stop_monitor(self) -> None:
        """
        Stops the background monitoring task.
        """
        self._is_running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logging.info("Stopped subagent heartbeat monitor.")
