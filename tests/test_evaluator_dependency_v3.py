import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from magda_agent.evaluation.evaluator_dependency_v3 import EvaluatorDependencyV3

@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
def evaluator(mock_llm):
    return EvaluatorDependencyV3(llm=mock_llm)

@pytest.mark.asyncio
async def test_evaluate_valid_plan(evaluator, mock_llm):
    plan = {
        "goal": "Write and test a function",
        "steps": [
            {"id": "step1", "description": "Write code", "dependencies": []},
            {"id": "step2", "description": "Write tests", "dependencies": ["step1"]},
            {"id": "step3", "description": "Run tests", "dependencies": ["step2"]}
        ]
    }

    mock_llm.chat_completion = AsyncMock(return_value=json.dumps({
        "score": 9,
        "approved": True,
        "feedback": "Logical order is correct."
    }))

    result = await evaluator.evaluate_plan_dependencies(plan)

    assert result["approved"] is True
    assert result["score"] == 9
    assert result["parallelism_ratio"] == 1/3
    assert "No cycles detected" in result["feedback"]

@pytest.mark.asyncio
async def test_evaluate_plan_with_cycle(evaluator, mock_llm):
    plan = {
        "goal": "Cyclic task",
        "steps": [
            {"id": "step1", "description": "Step 1", "dependencies": ["step2"]},
            {"id": "step2", "description": "Step 2", "dependencies": ["step1"]}
        ]
    }

    mock_llm.chat_completion = AsyncMock(return_value=json.dumps({
        "score": 10,
        "approved": True,
        "feedback": "LLM thinks it is okay but structure is not."
    }))

    result = await evaluator.evaluate_plan_dependencies(plan)

    assert result["approved"] is False
    assert result["score"] <= 5
    assert "Dependency cycle detected" in result["feedback"]

@pytest.mark.asyncio
async def test_evaluate_plan_missing_deps(evaluator, mock_llm):
    plan = {
        "goal": "Missing dep task",
        "steps": [
            {"id": "step1", "description": "Step 1", "dependencies": ["non_existent"]}
        ]
    }

    mock_llm.chat_completion = AsyncMock(return_value=json.dumps({
        "score": 8,
        "approved": True,
        "feedback": "Logic seems fine."
    }))

    result = await evaluator.evaluate_plan_dependencies(plan)

    assert result["approved"] is False
    assert result["score"] <= 5
    assert "Missing dependencies" in result["feedback"]

@pytest.mark.asyncio
async def test_evaluate_empty_plan(evaluator):
    plan = {"steps": []}
    result = await evaluator.evaluate_plan_dependencies(plan)
    assert result["approved"] is False
    assert "no steps" in result["feedback"].lower()

@pytest.mark.asyncio
async def test_parallelism_ratio(evaluator, mock_llm):
    plan = {
        "goal": "Parallel task",
        "steps": [
            {"id": "s1", "description": "s1", "dependencies": []},
            {"id": "s2", "description": "s2", "dependencies": []},
            {"id": "s3", "description": "s3", "dependencies": ["s1", "s2"]}
        ]
    }
    mock_llm.chat_completion = AsyncMock(return_value=json.dumps({
        "score": 10, "approved": True, "feedback": "OK"
    }))

    result = await evaluator.evaluate_plan_dependencies(plan)
    assert result["parallelism_ratio"] == 2/3
