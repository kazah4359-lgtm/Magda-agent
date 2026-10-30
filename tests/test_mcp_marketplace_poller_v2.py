import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from magda_agent.integration.mcp_marketplace_poller_v2 import MCPMarketplacePollerV2
from magda_agent.skills.mcp_registry_v5 import MCPRegistryV5
from magda_agent.scheduler.cron_scheduler_v2 import CronSchedulerV2


@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=MCPRegistryV5)
    registry.load_tool.return_value = True
    return registry


@pytest.fixture
def mock_scheduler():
    scheduler = MagicMock(spec=CronSchedulerV2)
    return scheduler


@pytest.mark.asyncio
async def test_schedule_sync(mock_registry, mock_scheduler):
    """Test that schedule_sync registers the task with the scheduler."""
    poller = MCPMarketplacePollerV2(mock_registry, mock_scheduler, cron_expr="*/15 * * * *")
    poller.schedule_sync()

    mock_scheduler.add_task.assert_called_once_with(
        "mcp_marketplace_sync_v2",
        "*/15 * * * *",
        poller.sync_marketplace,
    )
    await poller.close()


@pytest.mark.asyncio
@patch("magda_agent.integration.mcp_marketplace_poller_v2.httpx.AsyncClient.get")
async def test_sync_marketplace_direct_list_success(mock_get, mock_registry, mock_scheduler):
    """Test successful fetching and parsing of tools when response is a direct JSON list."""
    poller = MCPMarketplacePollerV2(mock_registry, mock_scheduler)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value=[
            {
                "name": "search_docs",
                "description": "Searches documentation",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            {
                "name": "invalid_item",
                "description": "Missing parameters and inputSchema",
            },
        ]
    )
    mock_get.return_value = mock_response

    registered_count = await poller.sync_marketplace()

    assert registered_count == 1
    mock_get.assert_called_once_with(poller.marketplace_url, timeout=10.0)
    mock_response.raise_for_status.assert_called_once()
    mock_registry.load_tool.assert_called_once()

    call_arg = mock_registry.load_tool.call_args[0][0]
    assert call_arg["name"] == "search_docs"
    assert "inputSchema" in call_arg
    assert "parameters" in call_arg

    await poller.close()


@pytest.mark.asyncio
@patch("magda_agent.integration.mcp_marketplace_poller_v2.httpx.AsyncClient.get")
async def test_sync_marketplace_wrapped_dict_success(mock_get, mock_registry, mock_scheduler):
    """Test parsing wrapped agentskills.io format dict containing 'skills' or 'tools' keys."""
    poller = MCPMarketplacePollerV2(mock_registry, mock_scheduler)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value={
            "skills": [
                {
                    "name": "code_executor",
                    "description": "Executes Python code",
                    "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}}},
                }
            ]
        }
    )
    mock_get.return_value = mock_response

    registered_count = await poller.sync_marketplace()

    assert registered_count == 1
    mock_registry.load_tool.assert_called_once()
    call_arg = mock_registry.load_tool.call_args[0][0]
    assert call_arg["name"] == "code_executor"
    assert call_arg["parameters"] == call_arg["inputSchema"]

    await poller.close()


@pytest.mark.asyncio
@patch("magda_agent.integration.mcp_marketplace_poller_v2.httpx.AsyncClient.get")
async def test_sync_marketplace_network_error(mock_get, mock_registry, mock_scheduler):
    """Test that httpx network errors are caught gracefully."""
    poller = MCPMarketplacePollerV2(mock_registry, mock_scheduler)
    mock_get.side_effect = httpx.RequestError("Connection timeout", request=MagicMock())

    registered_count = await poller.sync_marketplace()

    assert registered_count == 0
    mock_registry.load_tool.assert_not_called()

    await poller.close()


@pytest.mark.asyncio
@patch("magda_agent.integration.mcp_marketplace_poller_v2.httpx.AsyncClient.get")
async def test_sync_marketplace_http_status_error(mock_get, mock_registry, mock_scheduler):
    """Test that HTTP status errors (e.g. 500 or 404) are caught gracefully."""
    poller = MCPMarketplacePollerV2(mock_registry, mock_scheduler)

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Internal Server Error", request=MagicMock(), response=MagicMock()
    )
    mock_get.return_value = mock_response

    registered_count = await poller.sync_marketplace()

    assert registered_count == 0
    mock_registry.load_tool.assert_not_called()

    await poller.close()


@pytest.mark.asyncio
@patch("magda_agent.integration.mcp_marketplace_poller_v2.httpx.AsyncClient.get")
async def test_sync_marketplace_invalid_json_format(mock_get, mock_registry, mock_scheduler):
    """Test handling of unexpected JSON data types (e.g., dict without skills/tools list)."""
    poller = MCPMarketplacePollerV2(mock_registry, mock_scheduler)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"status": "ok", "message": "No tools available"})
    mock_get.return_value = mock_response

    registered_count = await poller.sync_marketplace()

    assert registered_count == 0
    mock_registry.load_tool.assert_not_called()

    await poller.close()


@pytest.mark.asyncio
async def test_alternative_registry_methods():
    """Test compatibility with registries using register_tool or register_skill."""
    mock_reg = MagicMock(spec=["register_tool"])
    mock_reg.register_tool.return_value = True

    poller = MCPMarketplacePollerV2(mock_reg, MagicMock())

    with patch("magda_agent.integration.mcp_marketplace_poller_v2.httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(
            return_value=[
                {
                    "name": "custom_tool",
                    "description": "Custom test tool",
                    "parameters": {"type": "object"},
                }
            ]
        )
        mock_get.return_value = mock_resp

        count = await poller.sync_marketplace()

        assert count == 1
        mock_reg.register_tool.assert_called_once()

    await poller.close()
