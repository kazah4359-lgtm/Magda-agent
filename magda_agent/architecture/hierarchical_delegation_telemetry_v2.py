"""
Hierarchical Delegation Telemetry Exporter V2.

Inspired by Magentic-One and Agent Teams trends: Exposes telemetry data
on parallel sub-agent communication, delegations, and results for live visualization tools.
"""

import time
import json
import logging
from collections import deque
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class HierarchicalDelegationTelemetryExporterV2:
    """
    Telemetry exporter for parallel sub-agent hierarchical delegation and message traffic.
    Captures message exchanges, task dispatches, and execution outcomes, generating
    structured visualization payloads and graph topologies for live UI tools.
    """

    def __init__(self, max_events: int = 2000, websocket: Optional[Any] = None) -> None:
        """
        Initialize the Telemetry Exporter.

        Args:
            max_events (int): Maximum number of events retained in memory.
            websocket (Optional[Any]): Optional WebSocket connection for real-time emission.
        """
        self.max_events = max_events
        self.websocket = websocket
        self._events: deque = deque(maxlen=max_events)

    def set_websocket(self, websocket: Any) -> None:
        """
        Set or update the active WebSocket connection.
        """
        self.websocket = websocket

    def capture_message(
        self,
        sender_id: str,
        recipient_id: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture inter-agent or sub-agent message traffic.

        Args:
            sender_id (str): Identifier of the sending agent.
            recipient_id (str): Identifier of the receiving agent.
            content (Any): Message content or payload.
            metadata (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            Dict[str, Any]: The recorded event record.
        """
        event = {
            "event_type": "message",
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self._events.append(event)
        logger.debug(f"Captured message from {sender_id} to {recipient_id}")
        return event

    def capture_delegation(
        self,
        parent_id: str,
        subagent_id: str,
        task: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture task delegation from a parent/orchestrator to a sub-agent.

        Args:
            parent_id (str): Identifier of the parent/orchestrator agent.
            subagent_id (str): Identifier of the target sub-agent.
            task (str): Description of the assigned sub-task.
            metadata (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            Dict[str, Any]: The recorded event record.
        """
        event = {
            "event_type": "delegation",
            "sender_id": parent_id,
            "recipient_id": subagent_id,
            "task": task,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self._events.append(event)
        logger.debug(f"Captured delegation from {parent_id} to {subagent_id}: {task}")
        return event

    def capture_result(
        self,
        subagent_id: str,
        parent_id: str,
        result: Any,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture execution result returned from a sub-agent to the parent orchestrator.

        Args:
            subagent_id (str): Identifier of the sub-agent.
            parent_id (str): Identifier of the parent orchestrator.
            result (Any): The execution outcome or error message.
            status (str): Outcome status ("success", "error", etc.).
            metadata (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            Dict[str, Any]: The recorded event record.
        """
        event = {
            "event_type": "result",
            "sender_id": subagent_id,
            "recipient_id": parent_id,
            "result": result,
            "status": status,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self._events.append(event)
        logger.debug(f"Captured result from {subagent_id} to {parent_id} with status '{status}'")
        return event

    def capture_event(
        self,
        event_type: str,
        source_id: str,
        target_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture a generic telemetry event.

        Args:
            event_type (str): Custom event category.
            source_id (str): Identifier of event source.
            target_id (Optional[str]): Identifier of event target.
            payload (Optional[Dict[str, Any]]): Arbitrary payload data.

        Returns:
            Dict[str, Any]: The recorded event record.
        """
        event = {
            "event_type": event_type,
            "sender_id": source_id,
            "recipient_id": target_id,
            "payload": payload or {},
            "timestamp": time.time()
        }
        self._events.append(event)
        return event

    def get_captured_messages(self) -> List[Dict[str, Any]]:
        """
        Get all captured events/messages.
        """
        return list(self._events)

    def get_traffic_by_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Filter captured traffic involving a specific agent (as sender or recipient).

        Args:
            agent_id (str): The agent identifier to filter by.

        Returns:
            List[Dict[str, Any]]: Matching telemetry events.
        """
        return [
            event for event in self._events
            if event.get("sender_id") == agent_id or event.get("recipient_id") == agent_id
        ]

    def clear(self) -> None:
        """
        Clear internal telemetry event store.
        """
        self._events.clear()

    def export_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics of captured delegation traffic.

        Returns:
            Dict[str, Any]: Aggregated traffic statistics.
        """
        events = list(self._events)
        total_events = len(events)
        delegations = [e for e in events if e.get("event_type") == "delegation"]
        messages = [e for e in events if e.get("event_type") == "message"]
        results = [e for e in events if e.get("event_type") == "result"]

        agents = set()
        for e in events:
            if e.get("sender_id"):
                agents.add(e["sender_id"])
            if e.get("recipient_id"):
                agents.add(e["recipient_id"])

        successful_results = [r for r in results if r.get("status") == "success"]

        return {
            "total_events": total_events,
            "total_delegations": len(delegations),
            "total_messages": len(messages),
            "total_results": len(results),
            "successful_results": len(successful_results),
            "active_agents_count": len(agents),
            "active_agents": sorted(list(agents))
        }

    def export_visualization_payload(self) -> Dict[str, Any]:
        """
        Export a graph topology and timeline payload suitable for live visualization tools (e.g. Canvas UI).

        Returns:
            Dict[str, Any]: Visualization payload containing nodes, edges, timeline events, and summary.
        """
        events = list(self._events)
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: Dict[str, Dict[str, Any]] = {}

        for e in events:
            sender = e.get("sender_id")
            recipient = e.get("recipient_id")
            event_type = e.get("event_type")

            if sender:
                if sender not in nodes:
                    nodes[sender] = {"id": sender, "type": "agent", "sent_count": 0, "received_count": 0}
                nodes[sender]["sent_count"] += 1

            if recipient:
                if recipient not in nodes:
                    nodes[recipient] = {"id": recipient, "type": "agent", "sent_count": 0, "received_count": 0}
                nodes[recipient]["received_count"] += 1

            if sender and recipient:
                edge_key = f"{sender}->{recipient}"
                if edge_key not in edges:
                    edges[edge_key] = {
                        "source": sender,
                        "target": recipient,
                        "message_count": 0,
                        "delegation_count": 0,
                        "result_count": 0
                    }
                if event_type == "message":
                    edges[edge_key]["message_count"] += 1
                elif event_type == "delegation":
                    edges[edge_key]["delegation_count"] += 1
                elif event_type == "result":
                    edges[edge_key]["result_count"] += 1

        return {
            "type": "hierarchical_delegation_telemetry",
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "timeline": events,
            "summary": self.export_summary()
        }

    async def broadcast_to_visualizer(self, websocket: Optional[Any] = None) -> bool:
        """
        Asynchronously broadcast the visualization payload over WebSocket.

        Args:
            websocket (Optional[Any]): Optional WebSocket target; falls back to self.websocket.

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        target_ws = websocket or self.websocket
        if target_ws is None:
            logger.debug("No WebSocket available for broadcasting telemetry.")
            return False

        payload = self.export_visualization_payload()
        try:
            message_str = json.dumps(payload)
            if hasattr(target_ws, "send_text"):
                await target_ws.send_text(message_str)
            elif hasattr(target_ws, "send_json"):
                await target_ws.send_json(payload)
            else:
                await target_ws.send(message_str)
            return True
        except Exception as e:
            logger.error(f"Failed to broadcast delegation telemetry: {e}")
            return False
