"""
OpenClaw MCP Subagent Router Plugin V1.

Inspired by OpenClaw gateway patterns: Build a subagent routing plugin using MCP
to delegate user prompts efficiently to specialized subagents or remote MCP endpoints.
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import httpx

from magda_agent.gateway.router import UnifiedMessage

logger = logging.getLogger(__name__)


@dataclass
class SubagentRouteTarget:
    """Represents a registered subagent route target."""
    subagent_id: str
    name: str
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    endpoint: Optional[str] = None
    handler: Optional[Callable[..., Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0


@dataclass
class RoutingDecision:
    """Summary of a routing decision made for a user prompt."""
    prompt: str
    target: Optional[SubagentRouteTarget]
    matched_reason: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPSubagentRouterPluginV1:
    """
    OpenClaw gateway-inspired subagent routing plugin that evaluates user prompts
    and delegates execution via MCP JSON-RPC endpoints or registered handlers.
    """

    def __init__(
        self,
        default_subagent_id: Optional[str] = None,
        timeout: float = 15.0,
        enable_audit_log: bool = True
    ) -> None:
        """
        Initialize the MCP Subagent Router Plugin.

        Args:
            default_subagent_id: Optional ID of the fallback subagent when no specific route matches.
            timeout: HTTP request timeout in seconds when querying MCP endpoints.
            enable_audit_log: Whether to retain an in-memory audit log of routed prompts.
        """
        self.subagents: Dict[str, SubagentRouteTarget] = {}
        self.default_subagent_id = default_subagent_id
        self.timeout = timeout
        self.enable_audit_log = enable_audit_log
        self.audit_log: List[Dict[str, Any]] = []

    def register_subagent(
        self,
        subagent_id: str,
        name: Optional[str] = None,
        description: str = "",
        capabilities: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        endpoint: Optional[str] = None,
        handler: Optional[Callable[..., Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> SubagentRouteTarget:
        """
        Register a subagent target for prompt delegation.

        Args:
            subagent_id: Unique identifier for the subagent.
            name: Display name (defaults to subagent_id).
            description: Description of subagent capabilities and purpose.
            capabilities: List of capability identifiers (e.g. ['code', 'sql']).
            keywords: Keywords that trigger routing to this subagent.
            endpoint: Remote MCP JSON-RPC endpoint URL (e.g. 'http://localhost:8000/mcp').
            handler: Callable handler for local execution fallback or mock.
            metadata: Additional metadata dictionary.
            priority: Priority weighting when multiple matches occur (higher takes precedence).

        Returns:
            The created SubagentRouteTarget.
        """
        target = SubagentRouteTarget(
            subagent_id=subagent_id,
            name=name or subagent_id,
            description=description,
            capabilities=capabilities or [],
            keywords=[k.lower() for k in (keywords or [])],
            endpoint=endpoint,
            handler=handler,
            metadata=metadata or {},
            priority=priority
        )
        self.subagents[subagent_id] = target
        logger.info(f"Registered subagent '{subagent_id}' with capabilities: {target.capabilities}")
        return target

    def unregister_subagent(self, subagent_id: str) -> bool:
        """Unregister a subagent by ID."""
        if subagent_id in self.subagents:
            del self.subagents[subagent_id]
            logger.info(f"Unregistered subagent '{subagent_id}'")
            return True
        return False

    def select_subagent(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        required_capability: Optional[str] = None
    ) -> RoutingDecision:
        """
        Select the best subagent to handle a user prompt based on explicit requirements,
        keywords, description, and capability matching.

        Args:
            prompt: The user prompt to route.
            context: Optional conversation context.
            required_capability: Optional explicit capability required.

        Returns:
            RoutingDecision containing selected target, reason, and confidence.
        """
        clean_prompt = prompt.lower().strip()

        # Check explicit required capability
        if required_capability:
            matching = [
                target for target in self.subagents.values()
                if required_capability in target.capabilities
            ]
            if matching:
                best = max(matching, key=lambda t: t.priority)
                return RoutingDecision(
                    prompt=prompt,
                    target=best,
                    matched_reason=f"explicit_capability:{required_capability}",
                    confidence=1.0
                )

        # Score registered subagents based on keyword matches, capability matches, and description
        scored_candidates: List[tuple[float, str, SubagentRouteTarget]] = []

        for target in self.subagents.values():
            score = 0.0
            reasons = []

            # Tokenize words for accurate matching
            prompt_tokens = set(re.findall(r"\b\w+\b", clean_prompt))

            # Exact subagent ID or name mention (highest confidence)
            if (
                target.subagent_id.lower() in prompt_tokens
                or target.name.lower() in clean_prompt
                or target.subagent_id.lower() in clean_prompt
            ):
                score += 5.0
                reasons.append("name_match")

            # Keyword matches (use word boundary matching)
            matched_kw = [
                kw for kw in target.keywords
                if kw in prompt_tokens or re.search(r"\b" + re.escape(kw) + r"\b", clean_prompt)
            ]
            if matched_kw:
                score += len(matched_kw) * 2.0
                reasons.append(f"keywords:{','.join(matched_kw)}")

            # Capability matches
            matched_caps = [
                cap for cap in target.capabilities
                if cap.lower() in prompt_tokens or re.search(r"\b" + re.escape(cap.lower()) + r"\b", clean_prompt)
            ]
            if matched_caps:
                score += len(matched_caps) * 2.5
                reasons.append(f"capabilities:{','.join(matched_caps)}")

            # Description keyword overlap (filter common stop words)
            stop_words = {"handles", "a", "an", "the", "and", "or", "in", "on", "to", "for", "with", "of", "is", "it"}
            if target.description:
                desc_tokens = set(re.findall(r"\b\w+\b", target.description.lower())) - stop_words
                meaningful_prompt_tokens = prompt_tokens - stop_words
                overlap = desc_tokens.intersection(meaningful_prompt_tokens)
                if overlap:
                    score += min(len(overlap) * 0.5, 2.0)
                    reasons.append(f"description_overlap:{len(overlap)}")

            # Priority bias
            score += target.priority * 0.1

            if score > 0.0:
                confidence = min(score / 5.0, 1.0)
                scored_candidates.append((score, "; ".join(reasons), target))

        if scored_candidates:
            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_reason, best_target = scored_candidates[0]
            confidence = min(best_score / 5.0, 1.0)
            return RoutingDecision(
                prompt=prompt,
                target=best_target,
                matched_reason=best_reason,
                confidence=confidence
            )

        # Fallback to default subagent if configured
        if self.default_subagent_id and self.default_subagent_id in self.subagents:
            return RoutingDecision(
                prompt=prompt,
                target=self.subagents[self.default_subagent_id],
                matched_reason="default_fallback",
                confidence=0.2
            )

        return RoutingDecision(
            prompt=prompt,
            target=None,
            matched_reason="no_route_matched",
            confidence=0.0
        )

    async def _delegate_to_mcp_endpoint(
        self,
        endpoint: str,
        subagent: SubagentRouteTarget,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dispatch prompt to remote MCP endpoint using standard JSON-RPC 2.0 protocol.

        Args:
            endpoint: URL of the MCP JSON-RPC endpoint.
            subagent: The selected target subagent.
            prompt: User prompt to delegate.
            context: Context list.
            extra_params: Additional parameters to include.

        Returns:
            Dict containing execution response from the MCP endpoint.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "execute_prompt",
            "params": {
                "subagent_id": subagent.subagent_id,
                "prompt": prompt,
                "context": context or [],
                **(extra_params or {})
            },
            "id": str(uuid.uuid4())
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                return {
                    "status": "mcp_error",
                    "subagent_id": subagent.subagent_id,
                    "error": data["error"]
                }

            return {
                "status": "success",
                "subagent_id": subagent.subagent_id,
                "result": data.get("result", {})
            }

    async def _delegate_to_handler(
        self,
        handler: Callable[..., Any],
        subagent: SubagentRouteTarget,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a local or mock subagent handler function."""
        kwargs = {
            "prompt": prompt,
            "context": context or [],
            "subagent": subagent,
            **(extra_params or {})
        }

        if asyncio.iscoroutinefunction(handler):
            res = await handler(**kwargs)
        else:
            res = await asyncio.to_thread(handler, **kwargs)

        return {
            "status": "success",
            "subagent_id": subagent.subagent_id,
            "result": res
        }

    async def route_prompt(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        required_capability: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate and route prompt to the matching subagent via MCP endpoint or callable handler.

        Args:
            prompt: The user prompt to route.
            context: Optional conversation context.
            required_capability: Optional explicit capability constraint.
            extra_params: Additional parameters passed to the target.

        Returns:
            Dict with execution status, subagent_id, result or error, and routing metadata.
        """
        decision = self.select_subagent(prompt, context=context, required_capability=required_capability)

        if not decision.target:
            result = {
                "status": "unrouted",
                "prompt": prompt,
                "reason": decision.matched_reason,
                "message": "No matching subagent found for prompt."
            }
            if self.enable_audit_log:
                self.audit_log.append({
                    "prompt": prompt,
                    "decision": decision,
                    "result": result
                })
            return result

        subagent = decision.target
        target_res: Dict[str, Any]

        try:
            if subagent.endpoint:
                logger.info(f"Routing prompt to MCP endpoint {subagent.endpoint} for subagent {subagent.subagent_id}")
                target_res = await self._delegate_to_mcp_endpoint(
                    endpoint=subagent.endpoint,
                    subagent=subagent,
                    prompt=prompt,
                    context=context,
                    extra_params=extra_params
                )
            elif subagent.handler:
                logger.info(f"Routing prompt to handler for subagent {subagent.subagent_id}")
                target_res = await self._delegate_to_handler(
                    handler=subagent.handler,
                    subagent=subagent,
                    prompt=prompt,
                    context=context,
                    extra_params=extra_params
                )
            else:
                target_res = {
                    "status": "configured_without_executor",
                    "subagent_id": subagent.subagent_id,
                    "message": f"Subagent '{subagent.subagent_id}' has neither endpoint nor handler configured."
                }
        except Exception as e:
            logger.error(f"Failed to delegate to subagent {subagent.subagent_id}: {e}")
            target_res = {
                "status": "error",
                "subagent_id": subagent.subagent_id,
                "error": str(e)
            }

        response = {
            **target_res,
            "routing_decision": {
                "subagent_id": subagent.subagent_id,
                "reason": decision.matched_reason,
                "confidence": decision.confidence
            }
        }

        if self.enable_audit_log:
            self.audit_log.append({
                "prompt": prompt,
                "decision": decision,
                "result": response
            })

        return response

    async def middleware(
        self,
        message: UnifiedMessage,
        next_call: Optional[Callable[[UnifiedMessage], Any]] = None
    ) -> Any:
        """
        OpenClaw gateway middleware compatible with LocalFirstGatewayV4.
        Inspects incoming UnifiedMessage, evaluates whether prompt delegation to a subagent
        is needed, and delegates or forwards to the next middleware in the chain.

        Args:
            message: Incoming UnifiedMessage.
            next_call: Next middleware or message handler function.

        Returns:
            Result from subagent delegation or next handler.
        """
        prompt = message.text
        # Check if message metadata requests subagent routing or if prompt has matching subagent
        requested_cap = message.metadata.get("required_capability")
        decision = self.select_subagent(prompt, required_capability=requested_cap)

        if decision.target and decision.confidence >= 0.3:
            # Delegate directly via MCP router
            return await self.route_prompt(
                prompt=prompt,
                context=message.metadata.get("context"),
                required_capability=requested_cap,
                extra_params={"user_id": message.user_id, "channel": message.channel}
            )

        if next_call:
            if asyncio.iscoroutinefunction(next_call):
                return await next_call(message)
            return next_call(message)

        return None


# Canonical and OpenClaw naming alias
OpenClawMCPSubagentRouterPluginV1 = MCPSubagentRouterPluginV1
