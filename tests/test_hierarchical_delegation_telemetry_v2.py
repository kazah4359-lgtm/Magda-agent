import json
from typing import Any
import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.architecture.hierarchical_delegation_telemetry_v2 import (
    HierarchicalDelegationTelemetryExporterV2
)


@pytest.mark.asyncio
async def test_capture_messages_and_delegations():
    exporter = HierarchicalDelegationTelemetryExporterV2()

    # Capture delegation
    del_event = exporter.capture_delegation("Orchestrator", "Worker-1", "Analyze database schema")
    assert del_event["event_type"] == "delegation"
    assert del_event["sender_id"] == "Orchestrator"
    assert del_event["recipient_id"] == "Worker-1"
    assert del_event["task"] == "Analyze database schema"

    # Capture message
    msg_event = exporter.capture_message("Worker-1", "Worker-2", {"query": "Find table names"})
    assert msg_event["event_type"] == "message"
    assert msg_event["sender_id"] == "Worker-1"
    assert msg_event["recipient_id"] == "Worker-2"

    # Capture result
    res_event = exporter.capture_result("Worker-1", "Orchestrator", "Schema analyzed successfully", status="success")
    assert res_event["event_type"] == "result"
    assert res_event["status"] == "success"

    # Capture generic event
    custom_event = exporter.capture_event("custom_step", "Worker-2", payload={"status": "processing"})
    assert custom_event["event_type"] == "custom_step"

    all_messages = exporter.get_captured_messages()
    assert len(all_messages) == 4


def test_get_traffic_by_agent():
    exporter = HierarchicalDelegationTelemetryExporterV2()

    exporter.capture_delegation("Orchestrator", "Agent-A", "Task A")
    exporter.capture_delegation("Orchestrator", "Agent-B", "Task B")
    exporter.capture_message("Agent-A", "Agent-B", "Hello Agent B")

    agent_a_traffic = exporter.get_traffic_by_agent("Agent-A")
    assert len(agent_a_traffic) == 2

    agent_b_traffic = exporter.get_traffic_by_agent("Agent-B")
    assert len(agent_b_traffic) == 2

    orchestrator_traffic = exporter.get_traffic_by_agent("Orchestrator")
    assert len(orchestrator_traffic) == 2

    unknown_traffic = exporter.get_traffic_by_agent("Agent-C")
    assert len(unknown_traffic) == 0


def test_export_summary():
    exporter = HierarchicalDelegationTelemetryExporterV2()

    exporter.capture_delegation("Parent", "Sub-1", "Task 1")
    exporter.capture_delegation("Parent", "Sub-2", "Task 2")
    exporter.capture_message("Sub-1", "Sub-2", "Data sync")
    exporter.capture_result("Sub-1", "Parent", "Done 1", status="success")
    exporter.capture_result("Sub-2", "Parent", "Failed 2", status="error")

    summary = exporter.export_summary()
    assert summary["total_events"] == 5
    assert summary["total_delegations"] == 2
    assert summary["total_messages"] == 1
    assert summary["total_results"] == 2
    assert summary["successful_results"] == 1
    assert summary["active_agents_count"] == 3
    assert summary["active_agents"] == ["Parent", "Sub-1", "Sub-2"]


def test_export_visualization_payload():
    exporter = HierarchicalDelegationTelemetryExporterV2()

    exporter.capture_delegation("MainOrchestrator", "CoderSubagent", "Write function")
    exporter.capture_message("CoderSubagent", "TesterSubagent", "Code ready for test")
    exporter.capture_result("TesterSubagent", "MainOrchestrator", "Tests passed", status="success")

    payload = exporter.export_visualization_payload()
    assert payload["type"] == "hierarchical_delegation_telemetry"
    assert "nodes" in payload
    assert "edges" in payload
    assert "timeline" in payload
    assert "summary" in payload

    nodes = {node["id"]: node for node in payload["nodes"]}
    assert "MainOrchestrator" in nodes
    assert "CoderSubagent" in nodes
    assert "TesterSubagent" in nodes

    assert nodes["MainOrchestrator"]["sent_count"] == 1
    assert nodes["CoderSubagent"]["received_count"] == 1
    assert nodes["CoderSubagent"]["sent_count"] == 1

    edges = {f"{edge['source']}->{edge['target']}": edge for edge in payload["edges"]}
    assert "MainOrchestrator->CoderSubagent" in edges
    assert edges["MainOrchestrator->CoderSubagent"]["delegation_count"] == 1
    assert "CoderSubagent->TesterSubagent" in edges
    assert edges["CoderSubagent->TesterSubagent"]["message_count"] == 1


def test_bounded_history_and_clear():
    exporter = HierarchicalDelegationTelemetryExporterV2(max_events=3)

    for i in range(5):
        exporter.capture_message(f"Agent-{i}", f"Agent-{i+1}", f"Msg {i}")

    messages = exporter.get_captured_messages()
    assert len(messages) == 3
    assert messages[0]["sender_id"] == "Agent-2"
    assert messages[2]["sender_id"] == "Agent-4"

    exporter.clear()
    assert len(exporter.get_captured_messages()) == 0


@pytest.mark.asyncio
async def test_broadcast_to_visualizer_send_text():
    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock()

    exporter = HierarchicalDelegationTelemetryExporterV2(websocket=mock_ws)
    exporter.capture_delegation("P", "S", "Do work")

    success = await exporter.broadcast_to_visualizer()
    assert success is True
    mock_ws.send_text.assert_called_once()
    sent_data = json.loads(mock_ws.send_text.call_args[0][0])
    assert sent_data["type"] == "hierarchical_delegation_telemetry"


@pytest.mark.asyncio
async def test_broadcast_to_visualizer_fallback_send():
    mock_ws = MagicMock(spec=["send"])
    mock_ws.send = AsyncMock()

    exporter = HierarchicalDelegationTelemetryExporterV2()
    exporter.capture_delegation("P", "S", "Do work")

    success = await exporter.broadcast_to_visualizer(websocket=mock_ws)
    assert success is True
    mock_ws.send.assert_called_once()


@pytest.mark.asyncio
async def test_mock_message_passing_integration():
    exporter = HierarchicalDelegationTelemetryExporterV2()

    # Simulate sub-agent message passing pipeline
    class MockSubAgent:
        def __init__(self, agent_id: str, exporter: HierarchicalDelegationTelemetryExporterV2):
            self.agent_id = agent_id
            self.exporter = exporter

        async def send_message(self, recipient_id: str, content: Any):
            self.exporter.capture_message(self.agent_id, recipient_id, content)

        async def return_result(self, parent_id: str, result: Any, status: str = "success"):
            self.exporter.capture_result(self.agent_id, parent_id, result, status)

    class MockOrchestrator:
        def __init__(self, exporter: HierarchicalDelegationTelemetryExporterV2):
            self.agent_id = "Orchestrator-1"
            self.exporter = exporter

        async def delegate(self, subagent: MockSubAgent, task: str):
            self.exporter.capture_delegation(self.agent_id, subagent.agent_id, task)

    orchestrator = MockOrchestrator(exporter)
    sub1 = MockSubAgent("SubAgent-Alpha", exporter)
    sub2 = MockSubAgent("SubAgent-Beta", exporter)

    # 1. Orchestrator delegates tasks to SubAgent-Alpha and SubAgent-Beta
    await orchestrator.delegate(sub1, "Fetch dataset")
    await orchestrator.delegate(sub2, "Prepare feature pipeline")

    # 2. SubAgent-Alpha communicates with SubAgent-Beta
    await sub1.send_message("SubAgent-Beta", {"dataset_id": "ds_101", "records": 500})

    # 3. SubAgent-Beta returns result to Orchestrator
    await sub2.return_result("Orchestrator-1", "Feature pipeline ready", status="success")

    # 4. Verify telemetry output
    payload = exporter.export_visualization_payload()
    assert payload["summary"]["total_delegations"] == 2
    assert payload["summary"]["total_messages"] == 1
    assert payload["summary"]["total_results"] == 1
    assert len(payload["nodes"]) == 3
    assert len(payload["edges"]) == 4
