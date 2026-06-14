"""
Integration module for Magda agent.
"""

from magda_agent.integration.a2a_discovery import AgentCard, A2ADiscovery
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.integration.a2a import A2AManager
from magda_agent.integration.cross_platform import CrossPlatformDispatcher
from magda_agent.integration.discord_bridge import DiscordBridge

__all__ = ["AgentCard", "A2ADiscovery", "A2ADelegator", "A2AManager", "CrossPlatformDispatcher", "DiscordBridge"]
