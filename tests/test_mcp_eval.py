import pytest
from magda_agent.evaluation.mcp_eval import MCPEvaluatorPlugin

@pytest.mark.asyncio
async def test_mcp_evaluator_valid_skill():
    evaluator = MCPEvaluatorPlugin()
    result = await evaluator.evaluate_skill("safe_skill", {"param1": "value1"})

    assert result["score"] == 1.0
    assert result["approved"] is True
    assert "Passed" in result["feedback"]

@pytest.mark.asyncio
async def test_mcp_evaluator_invalid_skill():
    evaluator = MCPEvaluatorPlugin()
    result = await evaluator.evaluate_skill("unsafe_skill", {"malicious_intent": "true"})

    assert result["score"] == 0.0
    assert result["approved"] is False
    assert "Failed" in result["feedback"]

@pytest.mark.asyncio
async def test_mcp_evaluator_empty_skill_name():
    evaluator = MCPEvaluatorPlugin()
    result = await evaluator.evaluate_skill("", {"param1": "value1"})

    assert result["score"] == 0.0
    assert result["approved"] is False
    assert "Failed" in result["feedback"]
