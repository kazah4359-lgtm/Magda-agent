"""
Security module for Magda Agent.
Contains components for sandboxing, taint tracking, and execution safety.
"""
from magda_agent.security.mcp_kernel import MCPKernel, SecurityError

__all__ = ["MCPKernel", "SecurityError"]
