import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from magda_agent.multi_agent.a2a_delegator import A2ADelegator

@pytest.mark.asyncio
async def test_delegate_task_success():
    delegator = A2ADelegator()
    peer_url = "http://peer-agent/api/v1/a2a"
    task = "analyze data"

    expected_response = {
        "jsonrpc": "2.0",
        "result": {"status": "completed", "output": "analysis done"},
        "id": 1
    }

    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.json.return_value = expected_response
        mock_response.raise_for_status = MagicMock()

        mock_post.return_value.__aenter__.return_value = mock_response

        result = await delegator.delegate_task(task, peer_url)

        assert result == expected_response
        mock_post.assert_called_once_with(
            peer_url,
            json={"jsonrpc": "2.0", "method": "execute_task", "params": {"task": task}, "id": 1}
        )

@pytest.mark.asyncio
async def test_delegate_task_failure():
    delegator = A2ADelegator()
    peer_url = "http://peer-agent/api/v1/a2a"
    task = "analyze data"

    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.side_effect = Exception("Connection error")

        result = await delegator.delegate_task(task, peer_url)

        assert result["status"] == "failed"
        assert "Connection error" in result["error"]
