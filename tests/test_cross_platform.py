import pytest
import asyncio
from magda_agent.integration.cross_platform import CrossPlatformDispatcher

class MockClient:
    """Mock client for testing."""
    def __init__(self, should_fail=False):
        self.messages = []
        self.should_fail = should_fail

    async def send_message(self, user_id: str, message: str) -> None:
        if self.should_fail:
            raise Exception("Mock failure")
        self.messages.append((user_id, message))

class BadMockClient:
    """Mock client without send_message method."""
    pass

@pytest.mark.asyncio
async def test_cross_platform_dispatcher_success():
    dispatcher = CrossPlatformDispatcher()
    client1 = MockClient()
    client2 = MockClient()

    dispatcher.register_platform("telegram", client1)
    dispatcher.register_platform("discord", client2)

    results = await dispatcher.broadcast("user123", "Hello World!")

    assert results == {"telegram": "success", "discord": "success"}
    assert client1.messages == [("user123", "Hello World!")]
    assert client2.messages == [("user123", "Hello World!")]

@pytest.mark.asyncio
async def test_cross_platform_dispatcher_partial_failure():
    dispatcher = CrossPlatformDispatcher()
    client_success = MockClient()
    client_fail = MockClient(should_fail=True)
    client_bad = BadMockClient()

    dispatcher.register_platform("slack", client_success)
    dispatcher.register_platform("whatsapp", client_fail)
    dispatcher.register_platform("signal", client_bad)

    results = await dispatcher.broadcast("user456", "Test message")

    assert results["slack"] == "success"
    assert results["whatsapp"] == "error: Mock failure"
    assert "failed" in results["signal"]

    assert client_success.messages == [("user456", "Test message")]

@pytest.mark.asyncio
async def test_cross_platform_dispatcher_empty():
    dispatcher = CrossPlatformDispatcher()
    results = await dispatcher.broadcast("user1", "msg")
    assert results == {}
