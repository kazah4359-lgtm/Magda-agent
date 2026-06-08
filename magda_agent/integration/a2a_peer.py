from typing import Dict, Any
from magda_agent.integration.a2a import A2AManager

class A2APeerDelegatorV5:
    """
    Wrapper for A2A Peer Delegation support (v5).
    """
    def __init__(self, manager: A2AManager):
        """
        Initializes the delegator.
        """
        self.manager = manager

    async def delegate_to_peer(self, capability: str, task_context: Dict[str, Any]) -> str:
        """
        Delegates a task to an A2A compliant peer.

        Args:
            capability: The capability to search for.
            task_context: The context for the task.

        Returns:
            A status message.
        """
        return await self.manager.delegate_task(capability, task_context)
