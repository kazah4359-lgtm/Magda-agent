"""
Integration module for Magda agent.
"""

from magda_agent.integration.a2a_discovery import AgentCard, A2ADiscovery
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.integration.a2a_delegator_v4 import A2ADelegatorV4
from magda_agent.integration.a2a_delegation_v5 import A2ADelegatorV5
from magda_agent.integration.a2a import A2AManager
from magda_agent.integration.cross_platform import CrossPlatformDispatcher
from magda_agent.integration.discord_bridge import DiscordBridge
from magda_agent.integration.a2a_status_broadcaster import A2AStatusBroadcaster

__all__ = ["AgentCard", "A2ADiscovery", "A2ADelegator", "A2ADelegatorV4", "A2ADelegatorV5", "A2AManager", "CrossPlatformDispatcher", "DiscordBridge", "A2AStatusBroadcaster"]
