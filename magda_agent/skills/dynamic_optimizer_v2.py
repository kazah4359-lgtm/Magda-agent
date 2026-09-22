import logging
import time
import functools
from typing import Dict, Any, Optional, Callable, Tuple, List


class DynamicSkillOptimizerV2:
    """
    Hermes-inspired Dynamic Skill Optimizer V2.
    Observes executions of standard skills, tracks consecutive successful runs,
    and upon reaching a threshold, generates and registers an optimized, cached
    version of the skill to reduce token usage and latency.
    """

    def __init__(
        self,
        registry: Optional[Any] = None,
        consecutive_threshold: int = 3,
        llm_client: Optional[Any] = None,
        cache_ttl: float = 300.0,
    ) -> None:
        """
        Initializes the DynamicSkillOptimizerV2.

        Args:
            registry: Optional SkillRegistry instance or dict-like object to register skills.
            consecutive_threshold: Number of consecutive successes required to optimize a skill.
            llm_client: Optional LLM client instance for advanced skill code/prompt optimization.
            cache_ttl: Time-to-live in seconds for cached skill outputs (default 300s).
        """
        self.registry = registry
        self.consecutive_threshold = consecutive_threshold
        self.llm_client = llm_client
        self.cache_ttl = cache_ttl

        self.consecutive_successes: Dict[str, int] = {}
        self.optimized_skills: Dict[str, Dict[str, Any]] = {}
        self.result_caches: Dict[str, Dict[Tuple, Dict[str, Any]]] = {}

    def record_execution(
        self,
        skill_name: str,
        success: bool,
        result: Any = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Records the outcome of a skill execution. Increments or resets the consecutive
        success counter. Triggers optimization if threshold is met.

        Args:
            skill_name: Name of the executed skill.
            success: Whether the execution succeeded.
            result: The output result of the skill execution.
            kwargs: Arguments supplied during execution.

        Returns:
            The name of the generated optimized skill if optimization occurred, else None.
        """
        if success:
            current = self.consecutive_successes.get(skill_name, 0) + 1
            self.consecutive_successes[skill_name] = current

            if current >= self.consecutive_threshold and not self.is_optimized(skill_name):
                return self.optimize_skill(skill_name)
        else:
            self.consecutive_successes[skill_name] = 0

        return None

    def observe_execution(
        self,
        skill_name: str,
        success: bool,
        result: Any = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Alias for record_execution."""
        return self.record_execution(skill_name, success, result, kwargs)

    def get_consecutive_successes(self, skill_name: str) -> int:
        """Returns the current count of consecutive successful executions for a skill."""
        return self.consecutive_successes.get(skill_name, 0)

    def is_optimized(self, skill_name: str) -> bool:
        """Checks if an optimized version of the skill has been generated and registered."""
        return skill_name in self.optimized_skills

    def get_optimized_skill_name(self, skill_name: str) -> str:
        """Returns the naming convention for the optimized version of a skill."""
        return f"opt_{skill_name}"

    def optimize_skill(
        self,
        skill_name: str,
        custom_generator: Optional[Callable[..., Any]] = None,
    ) -> str:
        """
        Generates and registers an optimized cached version of the skill.

        Args:
            skill_name: Name of the skill to optimize.
            custom_generator: Optional custom callable to generate optimized skill implementation.

        Returns:
            The registered optimized skill name.
        """
        opt_name = self.get_optimized_skill_name(skill_name)
        original_func: Optional[Callable] = None

        if self.registry is not None:
            if hasattr(self.registry, "skills") and skill_name in self.registry.skills:
                original_func = self.registry.skills[skill_name]
            elif hasattr(self.registry, "get_skill"):
                original_func = self.registry.get_skill(skill_name)
            elif isinstance(self.registry, dict) and skill_name in self.registry:
                original_func = self.registry[skill_name]

        self.result_caches[skill_name] = {}

        def _make_hashable(val: Any) -> Any:
            if isinstance(val, dict):
                return tuple(sorted((k, _make_hashable(v)) for k, v in val.items()))
            elif isinstance(val, (list, tuple)):
                return tuple(_make_hashable(x) for x in val)
            return val

        def optimized_wrapper(**kwargs) -> Any:
            """Cached and latency/token-optimized wrapper for the skill."""
            cache = self.result_caches.get(skill_name, {})
            key = _make_hashable(kwargs)
            now = time.time()

            # Check cache
            if key in cache:
                cached_entry = cache[key]
                if now - cached_entry["timestamp"] < self.cache_ttl:
                    logging.info(f"Cache hit for optimized skill '{opt_name}'")
                    return cached_entry["result"]

            if custom_generator is not None:
                res = custom_generator(**kwargs)
            elif original_func is not None:
                res = original_func(**kwargs)
            else:
                res = f"Optimized execution for '{skill_name}' (cached)."

            cache[key] = {"result": res, "timestamp": now}
            self.result_caches[skill_name] = cache
            return res

        description = f"Optimized and cached version of '{skill_name}' to reduce latency and token usage."

        if self.registry is not None:
            if hasattr(self.registry, "register_skill"):
                self.registry.register_skill(opt_name, optimized_wrapper, description)
            elif isinstance(self.registry, dict):
                self.registry[opt_name] = optimized_wrapper

        self.optimized_skills[skill_name] = {
            "optimized_name": opt_name,
            "original_name": skill_name,
            "created_at": time.time(),
            "description": description,
            "wrapper": optimized_wrapper,
        }

        logging.info(f"Generated and registered optimized skill: {opt_name}")
        return opt_name

    def execute_optimized(self, skill_name: str, **kwargs) -> Any:
        """
        Executes the optimized version of a skill directly if available.

        Args:
            skill_name: Name of the original skill.
            **kwargs: Arguments to pass to the optimized skill function.

        Returns:
            The execution result.
        """
        if skill_name in self.optimized_skills:
            wrapper = self.optimized_skills[skill_name]["wrapper"]
            return wrapper(**kwargs)
        elif self.registry is not None and hasattr(self.registry, "has_skill"):
            opt_name = self.get_optimized_skill_name(skill_name)
            if self.registry.has_skill(opt_name):
                return self.registry.execute_skill(opt_name, **kwargs)

        raise ValueError(f"No optimized version found for skill '{skill_name}'")

    def clear_cache(self, skill_name: Optional[str] = None) -> None:
        """
        Clears the result cache for a specific skill, or for all skills if skill_name is None.
        """
        if skill_name is not None:
            if skill_name in self.result_caches:
                self.result_caches[skill_name].clear()
        else:
            self.result_caches.clear()


# Aliases for compatibility across modules
DynamicOptimizerV2 = DynamicSkillOptimizerV2
HermesDynamicOptimizerV2 = DynamicSkillOptimizerV2
HermesDynamicSkillOptimizerV2 = DynamicSkillOptimizerV2
