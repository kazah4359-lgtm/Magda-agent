import logging
from typing import Dict, Any, List

from magda_agent.integration.a2a_cards import AgentCardV3
from magda_agent.integration.a2a_security import A2ASecurityContext
from magda_agent.integration.a2a_delegation import A2ADelegator


class A2AWorkflowManager:
    """
    Manages peer-to-peer chaining of agent tasks natively without a central coordinator.
    It orchestrates a workflow by passing context sequentially through a list of agents.
    """
    def __init__(self, delegator: A2ADelegator, security_context: A2ASecurityContext = None) -> None:
        """
        Initializes the A2AWorkflowManager.

        Args:
            delegator: An A2ADelegator instance to delegate tasks to peers.
            security_context: Optional security context.
        """
        self.delegator = delegator
        self.security_context = security_context or getattr(delegator, 'security_context', None) or A2ASecurityContext()

    async def execute_chain(self, workflow_steps: List[Dict[str, Any]], target_agents: List[AgentCardV3], initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a chain of tasks by delegating each step to a corresponding agent sequentially,
        passing the output context of one step as the input context for the next.

        Args:
            workflow_steps: A list of step definitions, where each step corresponds to an agent in target_agents.
            target_agents: A list of AgentCardV3 objects representing the peers to execute the steps.
            initial_context: The initial context for the first step.

        Returns:
            The final context after all steps have been executed.
        """
        if len(workflow_steps) != len(target_agents):
            raise ValueError("The number of workflow steps must match the number of target agents.")

        current_context = initial_context.copy()

        for step, agent in zip(workflow_steps, target_agents):
            logging.info(f"Executing workflow step '{step.get('name', 'unnamed')}' on agent {agent.name}")

            # Merge the step configuration into the current context
            delegation_payload = {
                "step_config": step,
                "context": current_context
            }

            try:
                # Delegate to peer
                result_str = await self.delegator.delegate_to_peer(agent, delegation_payload)
                logging.info(f"Step '{step.get('name', 'unnamed')}' completed with result: {result_str}")

                # We expect the result_str to be handled or we update the context based on it.
                # Since delegate_to_peer returns a status string in the current implementation,
                # we'll record the result in the context. In a real scenario, delegate_to_peer
                # might return structured data. For now, we embed the result string into the context.
                current_context[f"result_{step.get('name', 'unnamed')}"] = result_str

            except Exception as e:
                logging.error(f"Workflow failed at step '{step.get('name', 'unnamed')}' on agent {agent.name}: {e}")
                current_context["error"] = str(e)
                current_context["failed_step"] = step.get('name', 'unnamed')
                break

        return current_context
