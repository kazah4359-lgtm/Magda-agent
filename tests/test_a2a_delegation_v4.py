import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from typing import Any, Dict

from magda_agent.integration.a2a_delegation_v4 import A2ADelegatorV4
from magda_agent.integration.a2a_cards import AgentCardV3
from magda_agent.integration.a2a_security import A2ASecurityContext


@pytest.fixture
def mock_agent_card() -> AgentCardV3:
    return AgentCardV3(
        agent_id="test-agent-id",
        name="Test Agent",
        description="A test agent",
        capabilities=["code_execution"],
        endpoints={"rpc": "http://test-agent/rpc", "mcp": "http://test-agent/mcp"},
        protocol_version="v3"
    )

@pytest.fixture
def mock_security_context() -> A2ASecurityContext:
    context = MagicMock(spec=A2ASecurityContext)
    context.generate_token.return_value = "fake-token"
    return context

@pytest.mark.asyncio
async def test_delegate_task_success(mock_agent_card: AgentCardV3, mock_security_context: A2ASecurityContext) -> None:
    """Test successful task delegation using mocked httpx.AsyncClient."""
    delegator = A2ADelegatorV4(security_context=mock_security_context)
    task_payload = {"task": "test_task"}
    expected_response = {"status": "success", "result": "done"}

    mock_response = MagicMock()
    mock_response.json.return_value = expected_response
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await delegator.delegate_task(mock_agent_card, task_payload)

        assert result == expected_response
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://test-agent/rpc"
        assert kwargs['json'] == task_payload
        assert 'Authorization' in kwargs['headers']
        assert kwargs['headers']['Authorization'] == "Bearer fake-token"
        assert kwargs['timeout'] == 10.0


@pytest.mark.asyncio
async def test_delegate_task_no_endpoint() -> None:
    """Test task delegation fails when the agent has no endpoint."""
    card = AgentCardV3(
        agent_id="test-agent-id",
        name="Test Agent",
        description="A test agent",
        capabilities=["code_execution"],
        endpoints={},
        protocol_version="v3"
    )
    delegator = A2ADelegatorV4()

    with pytest.raises(ValueError, match="Agent test-agent-id missing endpoint"):
        await delegator.delegate_task(card, {"task": "test_task"})


@pytest.mark.asyncio
async def test_delegate_task_http_error(mock_agent_card: AgentCardV3) -> None:
    """Test task delegation handles HTTP errors properly."""
    delegator = A2ADelegatorV4()
    task_payload = {"task": "test_task"}

    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=MagicMock())

        with pytest.raises(httpx.HTTPError):
            await delegator.delegate_task(mock_agent_card, task_payload)
