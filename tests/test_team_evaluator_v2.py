import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.evaluation.team_evaluator_v2 import TeamEvaluatorV2

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock()
    return llm

@pytest.mark.asyncio
async def test_team_evaluator_success(mock_llm):
    """Tests successful evaluation of a multi-agent plan execution."""
    # Mock for EvaluatorDependencyV3 logical evaluation (called inside dependency_evaluator)
    # AND Holistic review.
    # EvaluatorDependencyV3._evaluate_logical_soundness is called via llm.chat_completion
    # Holistic review is also called via llm.chat_completion

    mock_llm.chat_completion.side_effect = [
        '{"score": 9, "approved": true, "feedback": "Dependency graph makes sense."}', # For EvaluatorDependencyV3
        '{"score": 10, "approved": true, "feedback": "All tasks completed successfully and goal achieved."}' # For Holistic Review
    ]

    evaluator = TeamEvaluatorV2(llm=mock_llm)

    plan = {
        "goal": "Build a website",
        "steps": [
            {"id": "step1", "description": "Write HTML", "dependencies": []},
            {"id": "step2", "description": "Write CSS", "dependencies": ["step1"]}
        ]
    }
    results = {
        "step1": "HTML code written.",
        "step2": "CSS code written."
    }

    result = await evaluator.evaluate_team_execution(plan, results)

    assert result["approved"] is True
    assert result["score"] == 10
    assert "metadata" in result
    assert result["metadata"]["holistic_eval"]["score"] == 10

@pytest.mark.asyncio
async def test_team_evaluator_structural_failure(mock_llm):
    """Tests rejection when a dependency cycle is detected."""
    evaluator = TeamEvaluatorV2(llm=mock_llm)

    plan = {
        "goal": "Build a website",
        "steps": [
            {"id": "step1", "description": "Write HTML", "dependencies": ["step2"]},
            {"id": "step2", "description": "Write CSS", "dependencies": ["step1"]}
        ]
    }
    results = {
        "step1": "HTML code written.",
        "step2": "CSS code written."
    }

    result = await evaluator.evaluate_team_execution(plan, results)

    assert result["approved"] is False
    assert "Structural validation failed" in result["feedback"]
    assert "cycle detected" in result["feedback"].lower()

@pytest.mark.asyncio
async def test_team_evaluator_missing_results(mock_llm):
    """Tests rejection when some step results are missing."""
    # Mock for EvaluatorDependencyV3
    mock_llm.chat_completion.return_value = '{"score": 9, "approved": true, "feedback": "Logic OK"}'

    evaluator = TeamEvaluatorV2(llm=mock_llm)

    plan = {
        "goal": "Build a website",
        "steps": [
            {"id": "step1", "description": "Write HTML", "dependencies": []},
            {"id": "step2", "description": "Write CSS", "dependencies": ["step1"]}
        ]
    }
    results = {
        "step1": "HTML code written."
    }

    result = await evaluator.evaluate_team_execution(plan, results)

    assert result["approved"] is False
    assert "Execution incomplete" in result["feedback"]
    assert "step2" in result["feedback"]

@pytest.mark.asyncio
async def test_team_evaluator_holistic_rejection(mock_llm):
    """Tests rejection when holistic LLM review fails."""
    mock_llm.chat_completion.side_effect = [
        '{"score": 9, "approved": true, "feedback": "Logic OK"}', # Dependency review
        '{"score": 3, "approved": false, "feedback": "CSS is incomplete and does not achieve goal."}' # Holistic review
    ]

    evaluator = TeamEvaluatorV2(llm=mock_llm)

    plan = {
        "goal": "Build a website",
        "steps": [
            {"id": "step1", "description": "Write HTML", "dependencies": []},
            {"id": "step2", "description": "Write CSS", "dependencies": ["step1"]}
        ]
    }
    results = {
        "step1": "HTML code written.",
        "step2": "Empty CSS."
    }

    result = await evaluator.evaluate_team_execution(plan, results)

    assert result["approved"] is False
    assert result["score"] <= 5
    assert "CSS is incomplete" in result["feedback"]
