import socket
import json
import asyncio
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from magda_agent.integration.a2a_discovery import AgentCard
from magda_agent.integration.a2a_mdns import A2AMDNSDiscovery, A2AMDNSListener


@pytest.fixture
def local_card() -> AgentCard:
    """
    Fixture returning a sample local AgentCard.
    """
    return AgentCard(
        agent_id="magda-local",
        name="Magda Local",
        description="Local agent",
        capabilities=["code_generation", "web_search"],
        endpoints={"rpc": "http://127.0.0.1:9090/rpc"},
    )


@pytest.fixture
def peer_card() -> AgentCard:
    """
    Fixture returning a sample peer AgentCard.
    """
    return AgentCard(
        agent_id="peer-1",
        name="Peer One",
        description="A cool peer",
        capabilities=["translation"],
        endpoints={"rpc": "http://127.0.0.1:8081/rpc"},
    )


@pytest.mark.asyncio
async def test_mdns_discovery_init(local_card: AgentCard) -> None:
    """
    Verifies initialization and service_type normalization.
    """
    discovery = A2AMDNSDiscovery(local_card=local_card, service_type="_a2a._tcp.local")
    assert discovery.service_type == "_a2a._tcp.local."
    assert discovery.local_card == local_card
    assert discovery.aio_zc is None


@pytest.mark.asyncio
async def test_extract_port_from_endpoints(local_card: AgentCard) -> None:
    """
    Verifies port extraction logic with different URL formats.
    """
    # Test valid port
    discovery = A2AMDNSDiscovery(local_card=local_card)
    assert discovery._extract_port_from_endpoints() == 9090

    # Test fallback when endpoints is empty
    empty_card = AgentCard(
        agent_id="empty",
        name="Empty",
        description="",
        capabilities=[],
        endpoints={},
    )
    discovery_empty = A2AMDNSDiscovery(local_card=empty_card)
    assert discovery_empty._extract_port_from_endpoints() == 8000


@pytest.mark.asyncio
async def test_mdns_discovery_start_stop(local_card: AgentCard) -> None:
    """
    Verifies that start() registers service and stop() unregisters and cleans up.
    """
    discovery = A2AMDNSDiscovery(local_card=local_card)

    mock_async_zc = MagicMock()
    mock_async_zc.register_service = AsyncMock()
    mock_async_zc.unregister_service = AsyncMock()
    mock_async_zc.close = AsyncMock()
    mock_async_zc.zeroconf = MagicMock()

    with patch("magda_agent.integration.a2a_mdns.AsyncZeroconf", return_value=mock_async_zc), \
         patch("magda_agent.integration.a2a_mdns.AsyncServiceBrowser") as mock_browser_cls:

        await discovery.start()

        assert discovery.aio_zc == mock_async_zc
        assert discovery.service_info is not None
        assert discovery.listener is not None
        assert discovery.browser is not None

        mock_async_zc.register_service.assert_called_once_with(discovery.service_info)
        mock_browser_cls.assert_called_once_with(
            mock_async_zc.zeroconf,
            discovery.service_type,
            discovery.listener,
        )

        # Let's stop
        await discovery.stop()

        assert discovery.aio_zc is None
        assert discovery.service_info is None
        assert discovery.listener is None
        assert discovery.browser is None
        mock_async_zc.unregister_service.assert_called_once()
        mock_async_zc.close.assert_called_once()


@pytest.mark.asyncio
async def test_listener_add_remove_service(local_card: AgentCard, peer_card: AgentCard) -> None:
    """
    Verifies that A2AMDNSListener correctly handles addition, update, and removal of services.
    """
    discovery = A2AMDNSDiscovery(local_card=local_card)

    mock_async_zc = MagicMock()
    mock_async_zc.register_service = AsyncMock()
    mock_async_zc.unregister_service = AsyncMock()
    mock_async_zc.close = AsyncMock()
    mock_async_zc.zeroconf = MagicMock()

    # Create mock AsyncServiceInfo for peer
    mock_info = MagicMock()
    mock_info.name = "peer-1._a2a._tcp.local."
    mock_info.properties = {
        b"card": peer_card.to_json().encode("utf-8"),
        b"agent_id": b"peer-1",
        b"name": b"Peer One",
        b"capabilities": b"translation",
    }
    mock_info.port = 8081
    mock_info.addresses = [socket.inet_aton("127.0.0.1")]

    mock_async_zc.get_service_info = AsyncMock(return_value=mock_info)

    with patch("magda_agent.integration.a2a_mdns.AsyncZeroconf", return_value=mock_async_zc), \
         patch("magda_agent.integration.a2a_mdns.AsyncServiceBrowser"):

        await discovery.start()
        listener = discovery.listener
        assert listener is not None

        # Simulate add_service
        listener.add_service(None, discovery.service_type, "peer-1._a2a._tcp.local.")

        # Give small sleep for async task to run
        await asyncio.sleep(0.01)

        # Verify peer was registered
        discovered_agents = discovery.get_discovered_agents()
        assert len(discovered_agents) == 1
        assert discovered_agents[0].agent_id == "peer-1"
        assert discovered_agents[0].name == "Peer One"
        assert "translation" in discovered_agents[0].capabilities

        # Verify retrieval by capability
        matching = discovery.find_agents_by_capability("translation")
        assert len(matching) == 1
        assert matching[0].agent_id == "peer-1"

        # Verify retrieval by ID
        found_card = discovery.get_agent_by_id("peer-1")
        assert found_card is not None
        assert found_card.agent_id == "peer-1"

        # Simulate update_service
        listener.update_service(None, discovery.service_type, "peer-1._a2a._tcp.local.")
        await asyncio.sleep(0.01)
        assert len(discovery.get_discovered_agents()) == 1

        # Simulate remove_service
        listener.remove_service(None, discovery.service_type, "peer-1._a2a._tcp.local.")
        await asyncio.sleep(0.01)

        # Verify peer was removed
        assert len(discovery.get_discovered_agents()) == 0
        await discovery.stop()


@pytest.mark.asyncio
async def test_fallback_parsing(local_card: AgentCard) -> None:
    """
    Verifies fallback parsing works when "card" property is not present in properties dict.
    """
    discovery = A2AMDNSDiscovery(local_card=local_card)

    mock_info = MagicMock()
    mock_info.name = "peer-fallback._a2a._tcp.local."
    # No "card" property, only individual fields
    mock_info.properties = {
        b"agent_id": b"peer-fallback",
        b"name": b"Peer Fallback",
        b"capabilities": b"web_search,translation",
        b"description": b"A fallback mDNS peer",
    }
    mock_info.port = 8085
    mock_info.addresses = [socket.inet_aton("127.0.0.1")]

    await discovery.process_discovered_service(mock_info)

    discovered = discovery.get_discovered_agents()
    assert len(discovered) == 1
    card = discovered[0]
    assert card.agent_id == "peer-fallback"
    assert card.name == "Peer Fallback"
    assert card.description == "A fallback mDNS peer"
    assert set(card.capabilities) == {"web_search", "translation"}
    assert card.endpoints == {"rpc": "http://127.0.0.1:8085"}
