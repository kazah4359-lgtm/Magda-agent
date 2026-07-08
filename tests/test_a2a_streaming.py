import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.integration.a2a_streaming import A2AStreamingDelegator
from magda_agent.integration.a2a_cards import AgentCardV3
from magda_agent.integration.a2a_security import A2ASecurityContext

@pytest.fixture
def target_agent() -> AgentCardV3:
    """Provides a dummy target agent card."""
    return AgentCardV3(
        agent_id="agent-v3-001",
        name="TargetAgent",
        description="Worker for testing",
        capabilities=["code_execution"],
        endpoints={"rpc": "http://192.168.1.10:8080/rpc"}
    )

class MockAsyncContextManager:
    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.mark.asyncio
async def test_stream_delegation_success(target_agent: AgentCardV3) -> None:
    """Tests successful streaming delegation with chunked updates."""
    security_context = A2ASecurityContext()
    delegator = A2AStreamingDelegator(security_context=security_context)

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()

    async def mock_aiter_lines():
        yield json.dumps({"status": "processing", "progress": 50})
        yield "" # empty chunk should be ignored
        yield "invalid json chunk"
        yield json.dumps({"status": "completed", "result": "done"})

    mock_response.aiter_lines = mock_aiter_lines

    class MockClient:
        def stream(self, *args, **kwargs):
            return MockAsyncContextManager(mock_response)

    with patch('httpx.AsyncClient', return_value=MockAsyncContextManager(MockClient())):
        chunks = []
        async for chunk in delegator.stream_delegation(target_agent, {"task": "test"}):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0]["status"] == "processing"
        assert "error" in chunks[1]
        assert chunks[1]["raw"] == "invalid json chunk"
        assert chunks[2]["result"] == "done"

@pytest.mark.asyncio
async def test_stream_delegation_missing_endpoint(target_agent: AgentCardV3) -> None:
    """Tests streaming delegation when the endpoint is missing."""
    target_agent.endpoints = {}
    delegator = A2AStreamingDelegator()

    chunks = []
    async for chunk in delegator.stream_delegation(target_agent, {"task": "test"}):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert "missing" in chunks[0]["error"]

@pytest.mark.asyncio
async def test_stream_delegation_http_error(target_agent: AgentCardV3) -> None:
    """Tests streaming delegation when an HTTP error occurs."""
    delegator = A2AStreamingDelegator()

    class ErrorContextManager:
        async def __aenter__(self):
            raise Exception("Connection Refused")

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch('httpx.AsyncClient', return_value=ErrorContextManager()):
        chunks = []
        async for chunk in delegator.stream_delegation(target_agent, {"task": "test"}):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "error" in chunks[0]
        assert "Connection Refused" in chunks[0]["error"]
