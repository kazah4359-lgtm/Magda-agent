"""
OpenAI Realtime Guardrail Fallback module.
Implements dynamic fallback strategies when runtime safety guardrails block a tool call.
"""
from typing import Any, Callable, Dict, Optional, Tuple
from magda_agent.safety.policy import PolicyLayer

class RealtimeGuardrailFallback:
    def __init__(self, policy_layer: Optional[PolicyLayer] = None) -> None:
        """
        Initializes the fallback guardrail.
        """
        self.policy_layer = policy_layer or PolicyLayer()
        self.fallback_handlers: Dict[str, Callable] = {}

    def register_fallback(self, tool_name: str, handler: Callable) -> None:
        """
        Registers a dynamic fallback handler for a specific tool.
        """
        self.fallback_handlers[tool_name] = handler

    def execute_with_fallback(self, tool_func: Callable, tool_name: str, **kwargs: Any) -> Any:
        """
        Executes the tool with a fallback mechanism on violation.
        """
        allow, explanation = self.policy_layer.evaluate(tool_name, **kwargs)
        if allow:
            return tool_func(**kwargs)

        if tool_name in self.fallback_handlers:
            return self.fallback_handlers[tool_name](explanation, **kwargs)

        raise ValueError(f"Action blocked by policy and no fallback available: {explanation}")
