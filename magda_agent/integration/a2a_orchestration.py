import asyncio
import logging
from typing import Dict, Any, List, AsyncGenerator, Optional

from magda_agent.integration.a2a_cards import AgentCardV3
from magda_agent.integration.a2a_security import A2ASecurityContext
from magda_agent.integration.a2a_streaming import A2AStreamingDelegatorV2

class A2AOrchestratorStream:
    """
    Coordinates peer-to-peer delegation streams to multiple agents asynchronously,
    inspired by OpenClaw trends for streaming A2A orchestration.
    """
    def __init__(self, security_context: Optional[A2ASecurityContext] = None, timeout: float = 60.0) -> None:
        """
        Initializes the A2AOrchestratorStream.

        Args:
            security_context: Optional security context to pass to the delegator.
            timeout: Configurable timeout for each delegation stream.
        """
        self.security_context = security_context or A2ASecurityContext()
        self.delegator = A2AStreamingDelegatorV2(security_context=self.security_context, timeout=timeout)

    async def _stream_from_agent(self, agent: AgentCardV3, plan_context: Dict[str, Any], queue: asyncio.Queue) -> None:
        """
        Streams updates from a single agent and places them in an asyncio queue.

        Args:
            agent: The target AgentCardV3 representing the peer.
            plan_context: The task context to delegate.
            queue: The asyncio Queue to put updates into.
        """
        try:
            async for chunk in self.delegator.stream_delegation_v2(agent, plan_context):
                await queue.put({"agent_name": agent.name, "chunk": chunk})
        except Exception as e:
            logging.error(f"Error streaming from agent {agent.name}: {e}")
            await queue.put({"agent_name": agent.name, "chunk": {"error": str(e)}})

    async def broadcast_plan_stream(self, target_agents: List[AgentCardV3], plan_context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Broadcasts a plan to multiple agents and yields their chunked updates concurrently.

        Args:
            target_agents: A list of target AgentCardV3 objects.
            plan_context: The context to delegate.

        Yields:
            A dictionary containing the agent_name and the chunk received.
        """
        if not target_agents:
            logging.warning("No target agents provided for broadcast.")
            return

        queue: asyncio.Queue = asyncio.Queue()
        tasks = []

        for agent in target_agents:
            task = asyncio.create_task(self._stream_from_agent(agent, plan_context, queue))
            tasks.append(task)

        active_tasks = len(tasks)

        while active_tasks > 0 or not queue.empty():
            try:
                # Wait for the next item or a small timeout to check task completion
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield item
                queue.task_done()
            except asyncio.TimeoutError:
                # Re-evaluate active tasks
                active_tasks = sum(1 for t in tasks if not t.done())

        # Ensure all tasks are awaited to propagate any unhandled exceptions (though they should be caught in _stream_from_agent)
        await asyncio.gather(*tasks, return_exceptions=True)
