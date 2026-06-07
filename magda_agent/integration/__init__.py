"""
Integration module for Magda agent.
"""

from magda_agent.integration.a2a_discovery import AgentCard, A2ADiscovery
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.integration.a2a import A2AManager

__all__ = ["AgentCard", "A2ADiscovery", "A2ADelegator", "A2AManager"]
