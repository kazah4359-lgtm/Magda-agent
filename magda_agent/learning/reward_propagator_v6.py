"""
OpenClaw-RL Online Reward Propagator V6.

Inspired by OpenClaw-RL implicit feedback trends: Dynamically traces sub-agent
execution paths and assigns positive or negative PAD feedback signals recursively up and down
the hierarchy of delegated actions based on user replies.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from magda_agent.emotions.mirror_neurons import MirrorNeurons

logger = logging.getLogger(__name__)


class ExecutionNode:
    """
    Represents an execution step within a sub-agent task hierarchy.
    """

    def __init__(
        self,
        node_id: str,
        agent_id: str,
        skill_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
        depth: int = 0
    ) -> None:
        """
        Initialize an ExecutionNode.

        Args:
            node_id (str): Unique node identifier.
            agent_id (str): Identifier of the agent executing the step.
            skill_id (Optional[str]): Identifier of the skill used (if applicable).
            parent_id (Optional[str]): Unique identifier of parent node in hierarchy.
            context_metadata (Optional[Dict[str, Any]]): Arbitrary context metadata.
            depth (int): Depth level within execution graph.
        """
        self.node_id = node_id
        self.agent_id = agent_id
        self.skill_id = skill_id
        self.parent_id = parent_id
        self.children_ids: List[str] = []
        self.context_metadata = context_metadata or {}
        self.depth = depth
        self.weight: float = 1.0


class OpenClawRLRewardPropagatorV6:
    """
    Online RL Reward Propagator V6.
    Traces execution hierarchies using context metadata, extracts PAD feedback signals
    from user replies via MirrorNeurons, and recursively propagates feedback up parent
    delegation chains and down to parallel subagents.
    """

    def __init__(
        self,
        mirror_neurons: Optional[MirrorNeurons] = None,
        learning_rate: float = 0.1,
        decay_factor: float = 0.8,
        min_weight: float = 0.1,
        max_weight: float = 2.0
    ) -> None:
        """
        Initialize the OpenClawRLRewardPropagatorV6.

        Args:
            mirror_neurons (Optional[MirrorNeurons]): Empathy module for PAD shift extraction.
            learning_rate (float): Step size for weight adjustments.
            decay_factor (float): Multiplier for propagating reward across hierarchy levels.
            min_weight (float): Minimum bound for agent/skill weights.
            max_weight (float): Maximum bound for agent/skill weights.
        """
        self.mirror_neurons = mirror_neurons or MirrorNeurons()
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.nodes: Dict[str, ExecutionNode] = {}
        self.agent_weights: Dict[str, float] = {}
        self.skill_weights: Dict[str, float] = {}

    def register_execution(
        self,
        node_id: str,
        agent_id: str,
        skill_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        context_metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionNode:
        """
        Register a new node in the execution hierarchy.

        Args:
            node_id (str): Unique identifier for this execution node.
            agent_id (str): Sub-agent identifier.
            skill_id (Optional[str]): Skill identifier used in execution.
            parent_id (Optional[str]): Parent execution node identifier.
            context_metadata (Optional[Dict[str, Any]]): Context metadata.

        Returns:
            ExecutionNode: The created execution node instance.
        """
        depth = 0
        if parent_id and parent_id in self.nodes:
            parent_node = self.nodes[parent_id]
            depth = parent_node.depth + 1
            if node_id not in parent_node.children_ids:
                parent_node.children_ids.append(node_id)

        node = ExecutionNode(
            node_id=node_id,
            agent_id=agent_id,
            skill_id=skill_id,
            parent_id=parent_id,
            context_metadata=context_metadata,
            depth=depth
        )

        self.nodes[node_id] = node

        if agent_id not in self.agent_weights:
            self.agent_weights[agent_id] = 1.0
        if skill_id and skill_id not in self.skill_weights:
            self.skill_weights[skill_id] = 1.0

        logger.debug(f"Registered execution node '{node_id}' for agent '{agent_id}' at depth {depth}")
        return node

    def get_node(self, node_id: str) -> Optional[ExecutionNode]:
        """
        Retrieve execution node by ID.
        """
        return self.nodes.get(node_id)

    def find_nodes_by_metadata(self, key: str, value: Any) -> List[ExecutionNode]:
        """
        Find execution nodes matching context metadata key-value pairs.
        """
        return [
            n for n in self.nodes.values()
            if n.context_metadata.get(key) == value
        ]

    def extract_pad_signal(self, text: str) -> Tuple[float, float, float]:
        """
        Extract Pleasure, Arousal, Dominance (PAD) shift tuple from text using MirrorNeurons.

        Args:
            text (str): Input text signal (user reply + optional tool output).

        Returns:
            Tuple[float, float, float]: PAD shift tuple (pleasure, arousal, dominance).
        """
        if not text:
            return (0.0, 0.0, 0.0)
        return self.mirror_neurons.empathize(text)

    def compute_reward(self, pad_shift: Tuple[float, float, float], tool_output: Optional[str] = None) -> float:
        """
        Compute a scalar reward signal (-1.0 to 1.0) from a PAD shift tuple.

        Args:
            pad_shift (Tuple[float, float, float]): PAD emotional shift (p, a, d).
            tool_output (Optional[str]): Optional output from tool execution.

        Returns:
            float: Scalar reward value bounded between -1.0 and 1.0.
        """
        p_shift, a_shift, d_shift = pad_shift

        # Base scalar reward primarily driven by Pleasure shift (p_shift)
        reward = p_shift * 2.0  # Scale p_shift (-0.5..0.5) to (-1.0..1.0)

        # Small bonus for positive tool output
        if tool_output and reward > 0.0:
            reward = min(1.0, reward + 0.1)

        return max(-1.0, min(1.0, reward))

    def propagate_feedback(
        self,
        node_id: str,
        user_reply: str,
        tool_output: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract implicit feedback from user reply and propagate reward signals
        upwards through parent nodes and trickling down to parallel subagents.

        Args:
            node_id (str): Target execution node ID where feedback originated.
            user_reply (str): The user's reply message.
            tool_output (Optional[str]): Optional tool execution result string.

        Returns:
            Dict[str, float]: Dictionary mapping updated entity IDs (agents/skills) to their new weights.
        """
        if node_id not in self.nodes:
            logger.warning(f"Node ID '{node_id}' not found in execution hierarchy.")
            return {}

        signal_text = user_reply or ""
        if tool_output:
            signal_text += f" {tool_output}"

        pad_shift = self.extract_pad_signal(signal_text)
        base_reward = self.compute_reward(pad_shift, tool_output)

        updated_weights: Dict[str, float] = {}
        visited: Set[str] = set()

        # 1. Direct update on target node
        self._apply_weight_update(node_id, base_reward, updated_weights)
        visited.add(node_id)

        target_node = self.nodes[node_id]

        # 2. Propagate upwards to parents
        parent_id = target_node.parent_id
        current_decay = self.decay_factor
        while parent_id and parent_id in self.nodes:
            if parent_id not in visited:
                decayed_reward = base_reward * current_decay
                self._apply_weight_update(parent_id, decayed_reward, updated_weights)
                visited.add(parent_id)

                # Trickle down to sibling parallel subagents under parent
                parent_node = self.nodes[parent_id]
                for child_id in parent_node.children_ids:
                    if child_id not in visited:
                        sibling_decayed_reward = decayed_reward * self.decay_factor
                        self._apply_weight_update(child_id, sibling_decayed_reward, updated_weights)
                        visited.add(child_id)

            parent_node = self.nodes[parent_id]
            parent_id = parent_node.parent_id
            current_decay *= self.decay_factor

        # 3. Trickle down to children of target node (if any parallel sub-tasks spawned below it)
        self._trickle_down_children(target_node, base_reward * self.decay_factor, visited, updated_weights)

        return updated_weights

    def _trickle_down_children(
        self,
        node: ExecutionNode,
        reward: float,
        visited: Set[str],
        updated_weights: Dict[str, float]
    ) -> None:
        """
        Recursively trickle reward down to child nodes with decay.
        """
        for child_id in node.children_ids:
            if child_id in self.nodes and child_id not in visited:
                self._apply_weight_update(child_id, reward, updated_weights)
                visited.add(child_id)
                child_node = self.nodes[child_id]
                self._trickle_down_children(child_node, reward * self.decay_factor, visited, updated_weights)

    def _apply_weight_update(
        self,
        node_id: str,
        reward: float,
        updated_weights: Dict[str, float]
    ) -> None:
        """
        Apply reward update to node, agent, and skill weights.
        """
        node = self.nodes[node_id]

        # Update node weight
        node.weight = max(self.min_weight, min(self.max_weight, node.weight + self.learning_rate * reward))

        # Update agent weight
        agent_id = node.agent_id
        curr_agent_w = self.agent_weights.get(agent_id, 1.0)
        new_agent_w = max(self.min_weight, min(self.max_weight, curr_agent_w + self.learning_rate * reward))
        self.agent_weights[agent_id] = new_agent_w
        updated_weights[f"agent:{agent_id}"] = new_agent_w

        # Update skill weight if present
        if node.skill_id:
            skill_id = node.skill_id
            curr_skill_w = self.skill_weights.get(skill_id, 1.0)
            new_skill_w = max(self.min_weight, min(self.max_weight, curr_skill_w + self.learning_rate * reward))
            self.skill_weights[skill_id] = new_skill_w
            updated_weights[f"skill:{skill_id}"] = new_skill_w

    def get_agent_weight(self, agent_id: str) -> float:
        """
        Get current weight of an agent.
        """
        return self.agent_weights.get(agent_id, 1.0)

    def get_skill_weight(self, skill_id: str) -> float:
        """
        Get current weight of a skill.
        """
        return self.skill_weights.get(skill_id, 1.0)

    def get_all_weights(self) -> Dict[str, Any]:
        """
        Get dictionary of all agent and skill weights.
        """
        return {
            "agents": self.agent_weights.copy(),
            "skills": self.skill_weights.copy()
        }


# Alias for backwards compatibility / shorthand imports
RewardPropagatorV6 = OpenClawRLRewardPropagatorV6
