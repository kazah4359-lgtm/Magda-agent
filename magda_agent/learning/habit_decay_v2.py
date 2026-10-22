import math
import time
import logging
import uuid
import chromadb
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class HabitDecayConfig:
    """
    Configurable parameters for OpenClaw-RL Habit Decay.
    """
    decay_rate: float = 0.05       # Daily decay rate for linear/exponential decay
    half_life_days: float = 14.0   # Half-life in days for exponential decay
    min_weight: float = 0.2        # Weight threshold below which habit is deleted
    decay_mode: str = "exponential" # "exponential" or "linear"
    feedback_sensitivity: float = 0.1 # Sensitivity factor when feedback modifies decay rate or weight


class OpenClawHabitDecayManagerV2:
    """
    OpenClaw-RL Habit Decay Parameterization & Feedback Manager V2.
    Manages habit decay parameters and applies dynamic decay to ChromaDB-backed
    or in-memory habit records based on explicit feedback loops.
    """

    def __init__(
        self,
        persist_directory: str = ":memory:",
        config: Optional[HabitDecayConfig] = None,
        collection_name: Optional[str] = None
    ) -> None:
        """
        Initializes OpenClawHabitDecayManagerV2.

        Args:
            persist_directory (str): ChromaDB persistence path or ":memory:".
            config (Optional[HabitDecayConfig]): Initial decay configuration.
            collection_name (Optional[str]): Custom collection name.
        """
        self.config = config or HabitDecayConfig()
        col_name = collection_name or f"habits_v2_{uuid.uuid4().hex[:8]}" if persist_directory == ":memory:" else (collection_name or "habits_v2")

        if persist_directory == ":memory:":
            self.client = chromadb.EphemeralClient()
            logger.info("Initialized OpenClawHabitDecayManagerV2 with EphemeralClient")
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Initialized OpenClawHabitDecayManagerV2 at {persist_directory}")

        self.collection = self.client.get_or_create_collection(name=col_name)

    def set_config(
        self,
        decay_rate: Optional[float] = None,
        half_life_days: Optional[float] = None,
        min_weight: Optional[float] = None,
        decay_mode: Optional[str] = None,
        feedback_sensitivity: Optional[float] = None
    ) -> None:
        """
        Updates decay parameters dynamically.
        """
        if decay_rate is not None:
            self.config.decay_rate = max(0.0, decay_rate)
        if half_life_days is not None:
            self.config.half_life_days = max(0.1, half_life_days)
        if min_weight is not None:
            self.config.min_weight = max(0.0, min_weight)
        if decay_mode is not None:
            if decay_mode in ("exponential", "linear"):
                self.config.decay_mode = decay_mode
            else:
                logger.warning(f"Unknown decay mode '{decay_mode}'. Keeping '{self.config.decay_mode}'.")
        if feedback_sensitivity is not None:
            self.config.feedback_sensitivity = max(0.0, feedback_sensitivity)

        logger.info(f"Updated HabitDecayConfig: {self.config}")

    def record_habit(
        self,
        input_text: str,
        skill_used: str,
        initial_weight: float = 1.0,
        user_id: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> str:
        """
        Records a habit into the ChromaDB collection.
        """
        habit_id = str(uuid.uuid4())
        ts = timestamp if timestamp is not None else time.time()
        metadata = {
            "skill_used": skill_used,
            "timestamp": ts,
            "weight": float(initial_weight)
        }
        if user_id is not None:
            metadata["user_id"] = user_id

        self.collection.add(
            documents=[input_text],
            metadatas=[metadata],
            ids=[habit_id]
        )
        logger.info(f"Recorded habit {habit_id} for skill '{skill_used}' with weight {initial_weight}")
        return habit_id

    def calculate_decayed_weight(
        self,
        current_weight: float,
        elapsed_days: float,
        config_override: Optional[HabitDecayConfig] = None
    ) -> float:
        """
        Calculates decayed weight based on elapsed days and decay configuration.
        """
        cfg = config_override or self.config
        if elapsed_days <= 0.0:
            return current_weight

        if cfg.decay_mode == "exponential":
            # half-life based decay: weight * (0.5 ** (elapsed_days / half_life_days))
            decay_factor = 0.5 ** (elapsed_days / cfg.half_life_days)
            return current_weight * decay_factor
        elif cfg.decay_mode == "linear":
            return current_weight - (cfg.decay_rate * elapsed_days)
        else:
            return current_weight

    def apply_decay(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Applies parameterized time decay to all habit records in the collection.
        Deletes habits whose decayed weight falls below min_weight.

        Returns:
            Dict[str, Any]: Summary containing deleted_count, updated_count, and total_remaining.
        """
        now = current_time if current_time is not None else time.time()
        results = self.collection.get(include=["metadatas"])

        if not results or not results.get("ids"):
            return {"deleted_count": 0, "updated_count": 0, "total_remaining": 0}

        ids_to_delete: List[str] = []
        ids_to_update: List[str] = []
        metadatas_to_update: List[Dict[str, Any]] = []

        for i, meta in enumerate(results["metadatas"]):
            if not meta or "timestamp" not in meta:
                continue

            record_time = meta["timestamp"]
            current_weight = meta.get("weight", 1.0)
            elapsed_seconds = max(0.0, now - record_time)
            elapsed_days = elapsed_seconds / (24.0 * 3600.0)

            new_weight = self.calculate_decayed_weight(current_weight, elapsed_days)

            if new_weight < self.config.min_weight:
                ids_to_delete.append(results["ids"][i])
            else:
                if new_weight != current_weight:
                    updated_meta = dict(meta)
                    updated_meta["weight"] = float(new_weight)
                    updated_meta["timestamp"] = float(now)
                    ids_to_update.append(results["ids"][i])
                    metadatas_to_update.append(updated_meta)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Decayed and deleted {len(ids_to_delete)} habit records below min_weight ({self.config.min_weight}).")

        if ids_to_update:
            self.collection.update(ids=ids_to_update, metadatas=metadatas_to_update)
            logger.info(f"Updated weights for {len(ids_to_update)} habit records.")

        total_remaining = self.collection.count()
        return {
            "deleted_count": len(ids_to_delete),
            "updated_count": len(ids_to_update),
            "total_remaining": total_remaining
        }

    def process_explicit_feedback(
        self,
        skill_used: str,
        feedback_score: float,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Adjusts habit weights and decay parameters based on explicit user feedback (-1.0 to +1.0 or score 0-10).

        - Positive feedback increases habit weights and can slow down decay rate.
        - Negative feedback decreases habit weights and can speed up decay rate.

        Args:
            skill_used (str): Skill associated with feedback.
            feedback_score (float): Feedback score (e.g., -1.0 to 1.0 or 0.0 to 10.0).
            user_id (Optional[int]): User ID filter if provided.

        Returns:
            Dict[str, Any]: Results of feedback application.
        """
        # Normalize feedback_score to range [-1.0, 1.0] if provided on scale 0.0 .. 10.0 or -1.0 .. 1.0
        if feedback_score > 1.0 or feedback_score == 0.0:
            norm_feedback = (feedback_score - 5.0) / 5.0
        else:
            norm_feedback = max(-1.0, min(1.0, feedback_score))

        query_kwargs: Dict[str, Any] = {"include": ["metadatas"]}
        results = self.collection.get(**query_kwargs)

        if not results or not results.get("ids"):
            return {"affected_habits": 0, "normalized_feedback": norm_feedback}

        ids_to_update: List[str] = []
        ids_to_delete: List[str] = []
        metadatas_to_update: List[Dict[str, Any]] = []

        for i, meta in enumerate(results["metadatas"]):
            if not meta or meta.get("skill_used") != skill_used:
                continue
            if user_id is not None and meta.get("user_id") != user_id:
                continue

            current_weight = meta.get("weight", 1.0)
            weight_delta = norm_feedback * self.config.feedback_sensitivity
            new_weight = current_weight + weight_delta

            if new_weight < self.config.min_weight:
                ids_to_delete.append(results["ids"][i])
            else:
                updated_meta = dict(meta)
                updated_meta["weight"] = float(new_weight)
                ids_to_update.append(results["ids"][i])
                metadatas_to_update.append(updated_meta)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        if ids_to_update:
            self.collection.update(ids=ids_to_update, metadatas=metadatas_to_update)

        # Dynamic global or skill decay parameter tuning based on overall feedback
        if norm_feedback < -0.5:
            # Negative feedback: accelerate decay slightly
            self.config.decay_rate *= (1.0 + self.config.feedback_sensitivity)
            self.config.half_life_days *= (1.0 - self.config.feedback_sensitivity * 0.5)
            self.config.half_life_days = max(1.0, self.config.half_life_days)
        elif norm_feedback > 0.5:
            # Positive feedback: slow down decay slightly
            self.config.decay_rate *= (1.0 - self.config.feedback_sensitivity * 0.5)
            self.config.decay_rate = max(0.001, self.config.decay_rate)
            self.config.half_life_days *= (1.0 + self.config.feedback_sensitivity * 0.5)

        logger.info(
            f"Processed feedback for skill '{skill_used}': norm_score={norm_feedback:.2f}, "
            f"updated={len(ids_to_update)}, deleted={len(ids_to_delete)}"
        )

        return {
            "affected_habits": len(ids_to_update) + len(ids_to_delete),
            "updated_count": len(ids_to_update),
            "deleted_count": len(ids_to_delete),
            "normalized_feedback": norm_feedback,
            "new_decay_rate": self.config.decay_rate,
            "new_half_life_days": self.config.half_life_days
        }
