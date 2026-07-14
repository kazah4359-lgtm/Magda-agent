import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Generator
from magda_agent.llm_client import LLMClient
from magda_agent.evaluation.evaluator_reflection_v3 import EvaluatorReflectionV3


@pytest.fixture
def mock_llm() -> Generator[MagicMock, None, None]:
    """
    Creates a mocked LLMClient instance.

    Yields:
        MagicMock: A mock LLM client with AsyncMock chat_completion.
    """
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock()
    yield llm


@pytest.mark.asyncio
async def test_evaluator_reflection_v3_metrics_only() -> None:
    """
    Tests that the metrics calculation correctly parses subagent traces and outputs
    accurate values for success rate, execution time, and error density.
    """
    evaluator = EvaluatorReflectionV3(llm=MagicMock())

    simulated_logs = [
        {
            "subagent_id": "sa-1",
            "task": "Develop HTML page layout",
            "status": "success",
            "execution_time_seconds": 15.5,
            "system_logs": ["Init workspace", "Write layout"],
            "errors": []
        },
        {
            "subagent_id": "sa-2",
            "task": "Develop CSS page styles",
            "status": "failure",
            "execution_time_seconds": 10.0,
            "system_logs": ["Compile CSS", "Fatal parsing error"],
            "errors": ["CompilationError: line 42", "SyntaxError: unmatched bracket"]
        }
    ]

    metrics = evaluator._parse_and_calculate_metrics(simulated_logs)

    assert metrics["total_subagents"] == 2
    assert metrics["successful_subagents"] == 1
    assert metrics["failed_subagents"] == 1
    assert metrics["success_rate"] == 0.5
    assert metrics["total_execution_time"] == 25.5
    assert metrics["total_errors"] == 2
    assert metrics["error_density"] == 1.0


@pytest.mark.asyncio
async def test_evaluator_reflection_v3_metrics_empty() -> None:
    """
    Tests that metrics calculation gracefully handles empty subagent logs list.
    """
    evaluator = EvaluatorReflectionV3(llm=MagicMock())
    metrics = evaluator._parse_and_calculate_metrics([])

    assert metrics["total_subagents"] == 0
    assert metrics["successful_subagents"] == 0
    assert metrics["failed_subagents"] == 0
    assert metrics["success_rate"] == 0.0
    assert metrics["total_execution_time"] == 0.0
    assert metrics["total_errors"] == 0
    assert metrics["error_density"] == 0.0


@pytest.mark.asyncio
async def test_evaluator_reflection_v3_full_success(mock_llm: MagicMock) -> None:
    """
    Tests successful subagent reflection execution when the LLM returns a valid JSON response.
    """
    mock_json = (
        "{"
        '  "approved": true,'
        '  "quality_score": 9,'
        '  "critique": "The subagents executed effectively, HTML and CSS integrate perfectly.",'
        '  "bottlenecks": [],'
        '  "actionable_feedback_for_planner": "None, the execution was flawless."'
        "}"
    )
    mock_llm.chat_completion.return_value = mock_json

    evaluator = EvaluatorReflectionV3(llm=mock_llm)

    simulated_logs = [
        {
            "subagent_id": "sa-1",
            "task": "Develop HTML page layout",
            "status": "success",
            "execution_time_seconds": 15.5,
            "system_logs": ["Init workspace", "Write layout"],
            "errors": []
        }
    ]

    result = await evaluator.reflect_on_subagent_logs(simulated_logs, "Create a clean front-page")

    assert result["approved"] is True
    assert result["metrics"]["total_subagents"] == 1
    assert result["metrics"]["success_rate"] == 1.0
    assert result["reflection"]["quality_score"] == 9
    assert result["reflection"]["approved"] is True
    assert "flawless" in result["reflection"]["actionable_feedback_for_planner"]


@pytest.mark.asyncio
async def test_evaluator_reflection_v3_with_markdown_cleaning(mock_llm: MagicMock) -> None:
    """
    Tests that the LLM markdown blocks (e.g. ```json ... ```) are cleaned and parsed correctly.
    """
    mock_json_md = (
        "```json\n"
        "{\n"
        '  "approved": false,\n'
        '  "quality_score": 4,\n'
        '  "critique": "CSS is entirely empty.",\n'
        '  "bottlenecks": ["sa-2"],\n'
        '  "actionable_feedback_for_planner": "Planner should verify CSS structure before running packaging tasks."\n'
        "}\n"
        "```"
    )
    mock_llm.chat_completion.return_value = mock_json_md

    evaluator = EvaluatorReflectionV3(llm=mock_llm)

    simulated_logs = [
        {
            "subagent_id": "sa-1",
            "task": "Develop HTML page layout",
            "status": "success",
            "execution_time_seconds": 15.5,
            "system_logs": ["Init workspace", "Write layout"],
            "errors": []
        },
        {
            "subagent_id": "sa-2",
            "task": "Develop CSS page styles",
            "status": "failure",
            "execution_time_seconds": 10.0,
            "system_logs": ["Compile CSS", "Fatal parsing error"],
            "errors": ["CompilationError: line 42"]
        }
    ]

    result = await evaluator.reflect_on_subagent_logs(simulated_logs, "Create a polished webpage")

    assert result["approved"] is False
    assert result["metrics"]["success_rate"] == 0.5
    assert result["reflection"]["quality_score"] == 4
    assert "sa-2" in result["reflection"]["bottlenecks"]


@pytest.mark.asyncio
async def test_evaluator_reflection_v3_fallback_on_invalid_json(mock_llm: MagicMock) -> None:
    """
    Tests that a fallback reflection dictionary is generated if the LLM returns invalid JSON.
    """
    mock_llm.chat_completion.return_value = "This response is not JSON at all."

    evaluator = EvaluatorReflectionV3(llm=mock_llm)

    simulated_logs = [
        {
            "subagent_id": "sa-1",
            "task": "Develop HTML page layout",
            "status": "success",
            "execution_time_seconds": 15.5,
            "system_logs": ["Init workspace", "Write layout"],
            "errors": []
        }
    ]

    result = await evaluator.reflect_on_subagent_logs(simulated_logs, "Task Goal")

    assert result["approved"] is True
    assert result["metrics"]["success_rate"] == 1.0
    assert "Failed to perform LLM analysis" in result["reflection"]["critique"]
    assert "Please inspect subagent error" in result["reflection"]["actionable_feedback_for_planner"]


@pytest.mark.asyncio
async def test_evaluator_reflection_v3_fallback_on_missing_keys(mock_llm: MagicMock) -> None:
    """
    Tests that a fallback reflection dictionary is generated if the LLM returns JSON missing required keys.
    """
    mock_llm.chat_completion.return_value = '{"approved": true, "quality_score": 10}' # missing critique, etc.

    evaluator = EvaluatorReflectionV3(llm=mock_llm)

    simulated_logs = [
        {
            "subagent_id": "sa-1",
            "task": "Develop HTML page layout",
            "status": "success",
            "execution_time_seconds": 15.5,
            "system_logs": ["Init workspace", "Write layout"],
            "errors": []
        }
    ]

    result = await evaluator.reflect_on_subagent_logs(simulated_logs, "Task Goal")

    assert result["approved"] is True
    assert result["metrics"]["success_rate"] == 1.0
    assert "lacks required JSON keys" in result["reflection"]["critique"]
