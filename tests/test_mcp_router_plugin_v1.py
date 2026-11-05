import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from magda_agent.architecture.mcp_router_plugin_v1 import (
    MCPSubagentRouterPluginV1,
    OpenClawMCPSubagentRouterPluginV1,
    SubagentRouteTarget,
    RoutingDecision,
)
from magda_agent.gateway.router import UnifiedMessage


def test_alias_equivalence():
    assert MCPSubagentRouterPluginV1 is OpenClawMCPSubagentRouterPluginV1


def test_register_and_unregister_subagent():
    plugin = MCPSubagentRouterPluginV1()
    target = plugin.register_subagent(
        subagent_id="coder_agent",
        name="Coder Subagent",
        description="Handles python coding tasks",
        capabilities=["coding", "python"],
        keywords=["code", "refactor", "bugfix"],
        endpoint="http://localhost:9000/mcp"
    )

    assert target.subagent_id == "coder_agent"
    assert target.name == "Coder Subagent"
    assert "coder_agent" in plugin.subagents
    assert "coding" in plugin.subagents["coder_agent"].capabilities

    assert plugin.unregister_subagent("coder_agent") is True
    assert "coder_agent" not in plugin.subagents
    assert plugin.unregister_subagent("non_existent") is False


def test_select_subagent_by_explicit_capability():
    plugin = MCPSubagentRouterPluginV1()
    plugin.register_subagent(
        subagent_id="researcher",
        capabilities=["search", "browse"],
        priority=1
    )
    plugin.register_subagent(
        subagent_id="deep_researcher",
        capabilities=["search", "deep_browse"],
        priority=5
    )

    decision = plugin.select_subagent("General message", required_capability="search")
    assert decision.target is not None
    assert decision.target.subagent_id == "deep_researcher"
    assert "explicit_capability:search" in decision.matched_reason
    assert decision.confidence == 1.0


def test_select_subagent_by_keywords_and_name():
    plugin = MCPSubagentRouterPluginV1()
    plugin.register_subagent(
        subagent_id="sql_specialist",
        name="SQL Specialist",
        keywords=["database", "query", "postgres"],
        capabilities=["data"]
    )
    plugin.register_subagent(
        subagent_id="designer",
        name="UI Designer",
        keywords=["frontend", "css", "layout"],
        capabilities=["design"]
    )

    decision = plugin.select_subagent("Can you help me write a postgres database query?")
    assert decision.target is not None
    assert decision.target.subagent_id == "sql_specialist"
    assert decision.confidence > 0.5


def test_select_subagent_fallback_to_default():
    plugin = MCPSubagentRouterPluginV1(default_subagent_id="general_assistant")
    plugin.register_subagent(
        subagent_id="general_assistant",
        name="General Assistant",
        description="Assists users with general tasks"
    )

    decision = plugin.select_subagent("completely unrelated xyz 12345")
    assert decision.target is not None
    assert decision.target.subagent_id == "general_assistant"
    assert decision.matched_reason == "default_fallback"


def test_select_subagent_no_match():
    plugin = MCPSubagentRouterPluginV1()
    decision = plugin.select_subagent("random query with no registered agents")
    assert decision.target is None
    assert decision.matched_reason == "no_route_matched"
    assert decision.confidence == 0.0


@pytest.mark.asyncio
async def test_route_prompt_to_handler():
    plugin = MCPSubagentRouterPluginV1()

    def sync_handler(prompt, **kwargs):
        return f"Handled sync: {prompt}"

    async def async_handler(prompt, **kwargs):
        return f"Handled async: {prompt}"

    plugin.register_subagent(
        subagent_id="sync_agent",
        keywords=["sync"],
        handler=sync_handler
    )
    plugin.register_subagent(
        subagent_id="async_agent",
        keywords=["async"],
        handler=async_handler
    )

    res_sync = await plugin.route_prompt("Run a sync task")
    assert res_sync["status"] == "success"
    assert res_sync["subagent_id"] == "sync_agent"
    assert res_sync["result"] == "Handled sync: Run a sync task"

    res_async = await plugin.route_prompt("Run an async task")
    assert res_async["status"] == "success"
    assert res_async["subagent_id"] == "async_agent"
    assert res_async["result"] == "Handled async: Run an async task"


@pytest.mark.asyncio
async def test_route_prompt_to_mcp_endpoint_success():
    plugin = MCPSubagentRouterPluginV1()
    endpoint = "http://remote-mcp.internal/rpc"

    plugin.register_subagent(
        subagent_id="remote_code_bot",
        endpoint=endpoint,
        keywords=["deploy", "kubernetes"]
    )

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "jsonrpc": "2.0",
        "result": {"deployment_status": "deployed_pod_xyz"},
        "id": "123"
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        res = await plugin.route_prompt("deploy application to kubernetes")

        assert res["status"] == "success"
        assert res["subagent_id"] == "remote_code_bot"
        assert res["result"] == {"deployment_status": "deployed_pod_xyz"}
        assert "routing_decision" in res

        # Verify POST payload sent to endpoint
        mock_client.post.assert_called_once()
        call_args, call_kwargs = mock_client.post.call_args
        assert call_args[0] == endpoint
        sent_json = call_kwargs["json"]
        assert sent_json["jsonrpc"] == "2.0"
        assert sent_json["method"] == "execute_prompt"
        assert sent_json["params"]["subagent_id"] == "remote_code_bot"
        assert sent_json["params"]["prompt"] == "deploy application to kubernetes"


@pytest.mark.asyncio
async def test_route_prompt_to_mcp_endpoint_error_payload():
    plugin = MCPSubagentRouterPluginV1()
    endpoint = "http://remote-mcp.internal/rpc"

    plugin.register_subagent(
        subagent_id="failing_mcp",
        endpoint=endpoint,
        keywords=["fail"]
    )

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "jsonrpc": "2.0",
        "error": {"code": -32603, "message": "Internal error in remote worker"},
        "id": "456"
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        res = await plugin.route_prompt("fail this task now")
        assert res["status"] == "mcp_error"
        assert res["subagent_id"] == "failing_mcp"
        assert res["error"]["message"] == "Internal error in remote worker"


@pytest.mark.asyncio
async def test_route_prompt_unrouted():
    plugin = MCPSubagentRouterPluginV1()
    res = await plugin.route_prompt("random unrouted prompt")
    assert res["status"] == "unrouted"
    assert "No matching subagent" in res["message"]


@pytest.mark.asyncio
async def test_route_prompt_configured_without_executor():
    plugin = MCPSubagentRouterPluginV1()
    plugin.register_subagent(
        subagent_id="empty_agent",
        keywords=["empty"]
    )
    res = await plugin.route_prompt("empty task")
    assert res["status"] == "configured_without_executor"
    assert res["subagent_id"] == "empty_agent"


@pytest.mark.asyncio
async def test_middleware_delegation():
    plugin = MCPSubagentRouterPluginV1()
    plugin.register_subagent(
        subagent_id="support_bot",
        keywords=["billing", "refund"],
        handler=lambda prompt, **kwargs: {"response": f"Support reply for {prompt}"}
    )

    # When matching subagent found
    msg1 = UnifiedMessage(
        channel="telegram",
        text="I need a refund for billing issue",
        user_id="user_123"
    )
    next_call = AsyncMock(return_value="next_handler_called")

    res1 = await plugin.middleware(msg1, next_call)
    assert res1["status"] == "success"
    assert res1["subagent_id"] == "support_bot"
    next_call.assert_not_called()

    # When no matching subagent found, forward to next_call
    msg2 = UnifiedMessage(
        channel="telegram",
        text="Hello world general message",
        user_id="user_123"
    )
    res2 = await plugin.middleware(msg2, next_call)
    assert res2 == "next_handler_called"
    next_call.assert_called_once_with(msg2)
