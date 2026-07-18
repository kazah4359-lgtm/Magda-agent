import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from magda_agent.agents.evaluator_subagent_v2 import EvaluatorSubagentV2
from magda_agent.llm_client import LLMClient
from magda_agent.memory.storage import MemorySystem

@pytest.fixture
def mock_llm():
    return AsyncMock(spec=LLMClient)

@pytest.fixture
def mock_memory():
    return AsyncMock(spec=MemorySystem)

@pytest.mark.asyncio
async def test_evaluate_response_success(mock_llm, mock_memory):
    evaluator = EvaluatorSubagentV2(llm=mock_llm, memory=mock_memory)

    # Mock the sub_agent execute to return a valid JSON evaluation
    evaluator.sub_agent.execute = AsyncMock(return_value="""```json
{
  "usefulness": 8,
  "accuracy": 9,
  "completeness": 7,
  "emotional_adequacy": 8,
  "policy_evaluations": {},
  "average_score": 8.0,
  "feedback": "Good response"
}
```""")

    result = await evaluator.evaluate_response("hello", "hi there")

    assert result is not None
    assert result["average_score"] == 8.0
    assert result["feedback"] == "Good response"

    # Check that memory was updated
    mock_memory.add_memory.assert_called_once()

@pytest.mark.asyncio
async def test_evaluate_response_retry_and_fail(mock_llm, mock_memory):
    evaluator = EvaluatorSubagentV2(llm=mock_llm, memory=mock_memory)

    # Mock sub_agent execute to always return invalid JSON
    evaluator.sub_agent.execute = AsyncMock(return_value="invalid json")

    result = await evaluator.evaluate_response("hello", "hi there")

    assert result is None
    # Should retry 3 times
    assert evaluator.sub_agent.execute.call_count == 3
    # Memory should not be updated
    mock_memory.add_memory.assert_not_called()

def test_get_feedback_for_prompt(mock_llm, mock_memory):
    evaluator = EvaluatorSubagentV2(llm=mock_llm, memory=mock_memory)

    # No evaluation yet
    assert evaluator.get_feedback_for_prompt() == ""

    # High score
    evaluator.last_evaluation = {"average_score": 9.0, "feedback": "Great"}
    assert evaluator.get_feedback_for_prompt() == ""

    # Low score
    evaluator.last_evaluation = {"average_score": 5.0, "feedback": "Needs work"}
    feedback = evaluator.get_feedback_for_prompt()
    assert "low evaluation score (5.0/10)" in feedback
    assert "Needs work" in feedback

@pytest.mark.asyncio
async def test_evaluate_planner_graph(mock_llm, mock_memory):
    evaluator = EvaluatorSubagentV2(llm=mock_llm, memory=mock_memory)

    # Mock worktree manager
    evaluator.worktree_manager = MagicMock()
    evaluator.worktree_manager.execute_concurrently = AsyncMock(return_value=['{"status": "graph evaluated"}'])

    plan = {
        "steps": [
            {"id": "step1", "dependencies": []},
            {"id": "step2", "dependencies": ["step1"]}
        ]
    }

    result = await evaluator.evaluate_planner_graph(plan)

    assert result == ['{"status": "graph evaluated"}']
    evaluator.worktree_manager.execute_concurrently.assert_called_once()
