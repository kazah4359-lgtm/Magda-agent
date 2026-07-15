import socket
import logging
from typing import Dict, List, Optional, Any
from zeroconf.asyncio import AsyncZeroconf, AsyncServiceBrowser, AsyncServiceInfo
from magda_agent.integration.a2a_discovery import AgentCard

logger = logging.getLogger(__name__)


class A2AMDNSListener:
    """
    Listener to receive mDNS service events and delegate them to A2AMDNSDiscovery.
    """

    def __init__(self, discovery_parent: "A2AMDNSDiscovery") -> None:
        """
        Initializes the listener with a reference to the parent discovery instance.

        Args:
            discovery_parent: The parent A2AMDNSDiscovery instance.
        """
        self.discovery_parent = discovery_parent

    def add_service(self, zc: Any, type_: str, name: str) -> None:
        """
        Callback when a new service is added. Schedules asynchronous resolution.

        Args:
            zc: The Zeroconf instance.
            type_: The service type.
            name: The service name.
        """
        import asyncio
        asyncio.create_task(self.async_add_service(zc, type_, name))

    def remove_service(self, zc: Any, type_: str, name: str) -> None:
        """
        Callback when a service is removed. Schedules asynchronous removal.

        Args:
            zc: The Zeroconf instance.
            type_: The service type.
            name: The service name.
        """
        import asyncio
        asyncio.create_task(self.async_remove_service(zc, type_, name))

    def update_service(self, zc: Any, type_: str, name: str) -> None:
        """
        Callback when a service is updated. Schedules asynchronous update/resolution.

        Args:
            zc: The Zeroconf instance.
            type_: The service type.
            name: The service name.
        """
        import asyncio
        asyncio.create_task(self.async_add_service(zc, type_, name))

    async def async_add_service(self, zc: Any, type_: str, name: str) -> None:
        """
        Asynchronously fetches details of the added/updated service and processes it.

        Args:
            zc: The Zeroconf instance.
            type_: The service type.
            name: The service name.
        """
        try:
            aio_zc = self.discovery_parent.aio_zc
            if aio_zc is None:
                return
            info = await aio_zc.get_service_info(type_, name)
            if info:
                await self.discovery_parent.process_discovered_service(info)
        except Exception as e:
            logger.error(f"Error fetching service info for {name}: {e}")

    async def async_remove_service(self, zc: Any, type_: str, name: str) -> None:
        """
        Asynchronously removes the service from discovered peers list.

        Args:
            zc: The Zeroconf instance.
            type_: The service type.
            name: The service name.
        """
        await self.discovery_parent.process_removed_service(name)


class A2AMDNSDiscovery:
    """
    Handles mDNS-based discovery of other agents in the local network and
    broadcasting the local agent's card without requiring a central registry.
    """

    def __init__(self, local_card: AgentCard, service_type: str = "_a2a._tcp.local.") -> None:
        """
        Initializes the mDNS discovery with the local agent's card.

        Args:
            local_card: The local agent's capabilities card.
            service_type: The mDNS service type, defaults to "_a2a._tcp.local.".
        """
        self.local_card = local_card
        self.service_type = service_type
        if not self.service_type.endswith("."):
            self.service_type += "."

        self.aio_zc: Optional[AsyncZeroconf] = None
        self.service_info: Optional[AsyncServiceInfo] = None
        self.browser: Optional[AsyncServiceBrowser] = None
        self.listener: Optional[A2AMDNSListener] = None

        self._discovered_agents: Dict[str, AgentCard] = {}
        self._service_name_to_agent_id: Dict[str, str] = {}

    def _extract_port_from_endpoints(self) -> int:
        """
        Extracts a port number from the local card's endpoints, falling back to 8000.

        Returns:
            The parsed port number or 8000.
        """
        for url in self.local_card.endpoints.values():
            if ":" in url:
                try:
                    parts = url.split(":")
                    port_str = parts[-1].split("/")[0]
                    return int(port_str)
                except Exception:
                    pass
        return 8000

    async def start(self) -> None:
        """
        Starts the mDNS discovery, registers the local service, and begins browsing.
        """
        logger.info(f"Starting A2A mDNS Discovery for agent: {self.local_card.name}")
        self.aio_zc = AsyncZeroconf()

        port = self._extract_port_from_endpoints()
        properties = {
            "card": self.local_card.to_json(),
            "agent_id": self.local_card.agent_id,
            "name": self.local_card.name,
            "capabilities": ",".join(self.local_card.capabilities),
        }

        # Instance name must be unique: e.g., agent-id._a2a._tcp.local.
        self.service_info = AsyncServiceInfo(
            type_=self.service_type,
            name=f"{self.local_card.agent_id}.{self.service_type}",
            addresses=[socket.inet_aton("127.0.0.1")],
            port=port,
            properties=properties,
        )

        await self.aio_zc.register_service(self.service_info)

        self.listener = A2AMDNSListener(self)
        self.browser = AsyncServiceBrowser(
            self.aio_zc.zeroconf,
            self.service_type,
            self.listener,
        )

    async def stop(self) -> None:
        """
        Stops browsing, unregisters the local service, and closes AsyncZeroconf.
        """
        logger.info(f"Stopping A2A mDNS Discovery for agent: {self.local_card.name}")
        if self.browser:
            # ServiceBrowser itself is closed synchronously or cleaned up by closing Zeroconf
            self.browser = None

        if self.aio_zc:
            if self.service_info:
                try:
                    await self.aio_zc.unregister_service(self.service_info)
                except Exception as e:
                    logger.error(f"Error unregistering service: {e}")
                self.service_info = None

            try:
                await self.aio_zc.close()
            except Exception as e:
                logger.error(f"Error closing AsyncZeroconf: {e}")
            self.aio_zc = None

        self.listener = None
        self._discovered_agents.clear()
        self._service_name_to_agent_id.clear()

    async def process_discovered_service(self, info: AsyncServiceInfo) -> None:
        """
        Processes resolved service information and registers the peer.

        Args:
            info: The resolved ServiceInfo object.
        """
        properties: Dict[str, str] = {}
        for k, v in info.properties.items():
            key_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            val_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            properties[key_str] = val_str

        card: Optional[AgentCard] = None
        card_json = properties.get("card")
        if card_json:
            try:
                card = AgentCard.from_json(card_json)
            except Exception as e:
                logger.warning(f"Failed to parse AgentCard JSON from card property: {e}")

        if not card:
            agent_id = properties.get("agent_id")
            name = properties.get("name")
            if agent_id and name:
                capabilities_str = properties.get("capabilities", "")
                capabilities = [c.strip() for c in capabilities_str.split(",") if c.strip()]
                endpoints: Dict[str, str] = {}
                if info.addresses:
                    try:
                        ip = socket.inet_ntoa(info.addresses[0])
                        endpoints["rpc"] = f"http://{ip}:{info.port}"
                    except Exception:
                        pass
                card = AgentCard(
                    agent_id=agent_id,
                    name=name,
                    description=properties.get("description", f"mDNS discovered Agent {name}"),
                    capabilities=capabilities,
                    endpoints=endpoints,
                )

        if card:
            if card.agent_id == self.local_card.agent_id:
                return
            self._discovered_agents[card.agent_id] = card
            self._service_name_to_agent_id[info.name] = card.agent_id
            logger.info(f"mDNS Discovered peer agent: {card.name} ({card.agent_id}) with capabilities {card.capabilities}")

    async def process_removed_service(self, name: str) -> None:
        """
        Handles service removal.

        Args:
            name: Fully-qualified name of the service being removed.
        """
        agent_id = self._service_name_to_agent_id.pop(name, None)
        if agent_id:
            card = self._discovered_agents.pop(agent_id, None)
            if card:
                logger.info(f"mDNS Removed peer agent: {card.name} ({agent_id})")

    def get_discovered_agents(self) -> List[AgentCard]:
        """
        Retrieves the list of currently discovered peer agents.

        Returns:
            A list of discovered AgentCards.
        """
        return list(self._discovered_agents.values())

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentCard]:
        """
        Retrieves a discovered agent's card by its ID.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            The discovered AgentCard or None.
        """
        return self._discovered_agents.get(agent_id)

    def find_agents_by_capability(self, capability: str) -> List[AgentCard]:
        """
        Returns a list of discovered Agent Cards that support the given capability.

        Args:
            capability: The capability to filter by.

        Returns:
            A list of matching AgentCards.
        """
        return [card for card in self._discovered_agents.values() if capability in card.capabilities]
