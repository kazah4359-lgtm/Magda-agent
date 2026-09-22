import pytest
import time
from unittest.mock import MagicMock
from magda_agent.skills.dynamic_optimizer_v2 import (
    DynamicSkillOptimizerV2,
    DynamicOptimizerV2,
    HermesDynamicOptimizerV2,
)
from magda_agent.skills.registry import SkillRegistry


def test_consecutive_success_detection():
    optimizer = DynamicSkillOptimizerV2(consecutive_threshold=3)

    assert optimizer.get_consecutive_successes("test_skill") == 0

    optimizer.record_execution("test_skill", success=True)
    assert optimizer.get_consecutive_successes("test_skill") == 1

    optimizer.record_execution("test_skill", success=True)
    assert optimizer.get_consecutive_successes("test_skill") == 2

    # Failure resets counter
    optimizer.record_execution("test_skill", success=False)
    assert optimizer.get_consecutive_successes("test_skill") == 0


def test_skill_optimization_and_registration():
    registry = SkillRegistry()
    mock_func = MagicMock(return_value="original output")
    registry.register_skill("calc_sum", mock_func, "Calculates sum")

    optimizer = DynamicSkillOptimizerV2(registry=registry, consecutive_threshold=3)

    opt_res1 = optimizer.record_execution("calc_sum", success=True)
    assert opt_res1 is None
    assert not optimizer.is_optimized("calc_sum")

    opt_res2 = optimizer.record_execution("calc_sum", success=True)
    assert opt_res2 is None

    # 3rd consecutive success triggers optimization
    opt_res3 = optimizer.record_execution("calc_sum", success=True)
    assert opt_res3 == "opt_calc_sum"
    assert optimizer.is_optimized("calc_sum")
    assert registry.has_skill("opt_calc_sum")


def test_optimized_skill_caching_and_execution():
    registry = SkillRegistry()
    call_count = 0

    def sample_skill(x: int) -> str:
        nonlocal call_count
        call_count += 1
        return f"result_{x}_{call_count}"

    registry.register_skill("sample_skill", sample_skill, "Sample skill")
    optimizer = DynamicSkillOptimizerV2(registry=registry, consecutive_threshold=2, cache_ttl=10.0)

    optimizer.record_execution("sample_skill", success=True)
    opt_name = optimizer.record_execution("sample_skill", success=True)
    assert opt_name == "opt_sample_skill"

    # Execute optimized skill first time with x=5
    res1 = registry.execute_skill("opt_sample_skill", x=5)
    assert res1 == "result_5_1"
    assert call_count == 1

    # Second time with same arguments should hit cache
    res2 = registry.execute_skill("opt_sample_skill", x=5)
    assert res2 == "result_5_1"
    assert call_count == 1  # Not incremented due to cache hit

    # Different argument x=10 triggers original function call
    res3 = registry.execute_skill("opt_sample_skill", x=10)
    assert res3 == "result_10_2"
    assert call_count == 2

    # Direct execution via optimizer
    res4 = optimizer.execute_optimized("sample_skill", x=5)
    assert res4 == "result_5_1"


def test_clear_cache():
    registry = SkillRegistry()
    mock_func = MagicMock(side_effect=["v1", "v2", "v3"])
    registry.register_skill("data_fetch", mock_func, "Fetches data")

    optimizer = DynamicSkillOptimizerV2(registry=registry, consecutive_threshold=1)
    optimizer.record_execution("data_fetch", success=True)

    res1 = optimizer.execute_optimized("data_fetch", query="abc")
    assert res1 == "v1"

    res2 = optimizer.execute_optimized("data_fetch", query="abc")
    assert res2 == "v1"

    # Clear cache for data_fetch
    optimizer.clear_cache("data_fetch")

    res3 = optimizer.execute_optimized("data_fetch", query="abc")
    assert res3 == "v2"


def test_custom_generator_and_aliases():
    optimizer1 = DynamicOptimizerV2()
    optimizer2 = HermesDynamicOptimizerV2()
    assert isinstance(optimizer1, DynamicSkillOptimizerV2)
    assert isinstance(optimizer2, DynamicSkillOptimizerV2)

    registry = {}
    optimizer = DynamicSkillOptimizerV2(registry=registry, consecutive_threshold=1)

    custom_gen = lambda **kwargs: f"custom_{kwargs.get('a', '')}"
    opt_name = optimizer.optimize_skill("custom_skill", custom_generator=custom_gen)

    assert opt_name == "opt_custom_skill"
    assert "opt_custom_skill" in registry

    res = registry["opt_custom_skill"](a="hello")
    assert res == "custom_hello"


def test_execute_optimized_error():
    optimizer = DynamicSkillOptimizerV2()
    with pytest.raises(ValueError, match="No optimized version found"):
        optimizer.execute_optimized("non_existent_skill")
