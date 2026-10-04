import logging
from typing import Any, Dict, List, Optional
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.memory.context_engine import ContextPlugin

logger = logging.getLogger(__name__)


class OpenClawContextOnlineRLV2(ContextPlugin):
    """
    OpenClaw-RL Online Reinforcement Learning Context Engine Module V2.
    Dynamically updates context retrieval weights based on user feedback signals,
    optimizing context ranking and filtering across episodic, semantic,
    procedural, and working memory layers.
    """

    DEFAULT_CATEGORIES = ("episodic", "semantic", "procedural", "working")

    def __init__(
        self,
        mirror_neurons: Optional[MirrorNeurons] = None,
        min_weight: float = 0.1,
        max_weight: float = 3.0,
        learning_rate: float = 0.1,
    ) -> None:
        """
        Initializes OpenClawContextOnlineRLV2.

        Args:
            mirror_neurons: MirrorNeurons instance to calculate pleasure shift rewards from text feedback.
            min_weight: Minimum lower bound for context category weights.
            max_weight: Maximum upper bound for context category weights.
            learning_rate: Scaling factor for reward-driven weight updates.
        """
        self.mirror_neurons = mirror_neurons or MirrorNeurons()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.learning_rate = learning_rate

        # Mapping of user_id -> context_category -> weight
        self._user_weights: Dict[str, Dict[str, float]] = {}
        self._default_weights: Dict[str, float] = {
            cat: 1.0 for cat in self.DEFAULT_CATEGORIES
        }
        logger.info("OpenClawContextOnlineRLV2 initialized.")

    def _get_user_weights_dict(self, user_id: Optional[str] = None) -> Dict[str, float]:
        key = str(user_id) if user_id is not None else "default"
        if key not in self._user_weights:
            self._user_weights[key] = dict(self._default_weights)
        return self._user_weights[key]

    def get_retrieval_weights(self, user_id: Optional[str] = None) -> Dict[str, float]:
        """
        Returns the current context retrieval weights for a specific user or default context.

        Args:
            user_id: Optional user identifier.

        Returns:
            Dict[str, float]: Copy of category weights.
        """
        return dict(self._get_user_weights_dict(user_id))

    def adjust_weights_for_context(
        self, category: str, delta: float, user_id: Optional[str] = None
    ) -> float:
        """
        Adjusts the weight for a specific context category by a delta.

        Args:
            category: The memory category (e.g. 'episodic', 'semantic').
            delta: Weight increment or decrement.
            user_id: Optional user identifier.

        Returns:
            float: The updated weight value.
        """
        weights = self._get_user_weights_dict(user_id)
        current = weights.get(category, 1.0)
        updated = max(self.min_weight, min(self.max_weight, current + delta))
        weights[category] = updated
        logger.info(
            f"Adjusted context weight for category '{category}' (User: {user_id}): {current:.3f} -> {updated:.3f}"
        )
        return updated

    async def process_feedback(
        self,
        user_feedback: str,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        reward: Optional[float] = None,
    ) -> float:
        """
        Processes explicit user feedback text or explicit reward scalar to dynamically update
        context retrieval weights.

        Args:
            user_feedback: Feedback text provided by user.
            retrieved_context: Optional list of context items used in the preceding turn.
            user_id: Optional user identifier.
            reward: Optional explicit reward score overriding sentiment analysis.

        Returns:
            float: The computed or applied reward value.
        """
        computed_reward: float
        if reward is not None:
            computed_reward = reward
        elif user_feedback:
            p_shift, _, _ = self.mirror_neurons.empathize(user_feedback)
            computed_reward = p_shift
        else:
            computed_reward = 0.0

        if computed_reward == 0.0:
            return 0.0

        delta = computed_reward * self.learning_rate

        # Determine which context categories were involved
        categories_to_update = set()
        if retrieved_context:
            for item in retrieved_context:
                if isinstance(item, dict):
                    cat = item.get("category") or item.get("type") or item.get("layer")
                    if cat and isinstance(cat, str):
                        categories_to_update.add(cat.lower())

        if not categories_to_update:
            categories_to_update = set(self.DEFAULT_CATEGORIES)

        for cat in categories_to_update:
            self.adjust_weights_for_context(cat, delta, user_id=user_id)

        logger.info(
            f"Processed RL context feedback (Reward: {computed_reward:.2f}, User: {user_id}) for categories: {categories_to_update}"
        )
        return computed_reward

    def rank_and_filter_context(
        self, context_items: List[Dict[str, Any]], user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Ranks and scores context items using the learned retrieval weights for each category.

        Args:
            context_items: List of context dictionary items.
            user_id: Optional user identifier.

        Returns:
            List[Dict[str, Any]]: Context items sorted by weighted relevance score.
        """
        if not context_items:
            return []

        weights = self._get_user_weights_dict(user_id)

        scored_items = []
        for item in context_items:
            if not isinstance(item, dict):
                scored_items.append((1.0, item))
                continue

            category = (
                item.get("category") or item.get("type") or item.get("layer") or "working"
            )
            cat_str = str(category).lower()
            cat_weight = weights.get(cat_str, 1.0)

            base_score = float(
                item.get("score") or item.get("relevance") or item.get("confidence") or 1.0
            )
            final_score = base_score * cat_weight

            item_copy = dict(item)
            item_copy["_weighted_score"] = final_score
            item_copy["_category_weight"] = cat_weight
            scored_items.append((final_score, item_copy))

        # Sort descending by weighted score
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items]

    # --- ContextPlugin Interface Lifecycle Methods ---

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """
        ContextPlugin hook called after context items are retrieved.
        Re-ranks context items using current retrieval weights.
        """
        dict_items = []
        non_dict_items = []
        for item in context:
            if isinstance(item, dict):
                dict_items.append(item)
            else:
                non_dict_items.append(item)

        ranked_dicts = self.rank_and_filter_context(dict_items, user_id=str(user_id))
        return ranked_dicts + non_dict_items

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """
        ContextPlugin hook called when context is updated.
        Inspects payload for user feedback / reward signals.
        """
        feedback_text = ""
        reward_val = None
        retrieved_context = None

        if isinstance(new_context, dict):
            feedback_text = str(new_context.get("feedback") or new_context.get("user_reply") or "")
            if "reward" in new_context:
                try:
                    reward_val = float(new_context["reward"])
                except (ValueError, TypeError):
                    reward_val = None
            if "retrieved_context" in new_context and isinstance(new_context["retrieved_context"], list):
                retrieved_context = new_context["retrieved_context"]

        if feedback_text or reward_val is not None:
            # Synchronous trigger of feedback processing
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.process_feedback(
                        user_feedback=feedback_text,
                        retrieved_context=retrieved_context,
                        user_id=str(user_id),
                        reward=reward_val,
                    )
                )
            except RuntimeError:
                # If no running loop, run directly
                asyncio.run(
                    self.process_feedback(
                        user_feedback=feedback_text,
                        retrieved_context=retrieved_context,
                        user_id=str(user_id),
                        reward=reward_val,
                    )
                )
