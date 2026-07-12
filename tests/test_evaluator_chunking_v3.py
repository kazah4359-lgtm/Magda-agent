import pytest
from unittest.mock import AsyncMock, patch
import json
from magda_agent.agents.evaluator_chunking_v3 import EvaluatorAgentChunkingV3
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_evaluator_chunking_v3_success():
    """
    Test the map-reduce pattern for a successful evaluation where all chunks are approved.
    """
    mock_llm = AsyncMock(spec=LLMClient)
    # Use a small chunk size to force chunking
    evaluator = EvaluatorAgentChunkingV3(llm=mock_llm, chunk_size=20)

    # 44 characters -> 3 chunks (20, 20, 4)
    output_text = "This is a very long text to test the chunker"

    with patch("magda_agent.agents.sub_agent.SubAgent.execute") as mock_execute:
        # Mock LLM returning passing evaluations for all chunks
        mock_execute.side_effect = [
            '{"score": 8, "approved": true, "feedback": "Chunk 1 is good."}',
            '{"score": 9, "approved": true, "feedback": "Chunk 2 is great."}',
            '{"score": 7, "approved": true, "feedback": "Chunk 3 is okay."}'
        ]

        result = await evaluator.evaluate_output(output_text, "Test request", {"acceptance": []})

        assert result["approved"] is True
        assert result["score"] == 8.0 # (8+9+7)/3
        assert result["metadata"]["chunks_processed"] == 3
        assert "Chunk 1 is good." in result["feedback"]
        assert "Chunk 2 is great." in result["feedback"]
        assert "Chunk 3 is okay." in result["feedback"]

@pytest.mark.asyncio
async def test_evaluator_chunking_v3_failure():
    """
    Test the map-reduce pattern where one chunk fails, causing the whole evaluation to fail.
    """
    mock_llm = AsyncMock(spec=LLMClient)
    evaluator = EvaluatorAgentChunkingV3(llm=mock_llm, chunk_size=20)

    output_text = "This text will fail in one chunk."

    with patch("magda_agent.agents.sub_agent.SubAgent.execute") as mock_execute:
        # One chunk fails
        mock_execute.side_effect = [
            '{"score": 8, "approved": true, "feedback": "Good chunk"}',
            '{"score": 3, "approved": false, "feedback": "Bad chunk"}'
        ]

        result = await evaluator.evaluate_output(output_text, "Test request", {})

        assert result["approved"] is False
        assert result["score"] == 5.5 # (8+3)/2
        assert "Bad chunk" in result["feedback"]

@pytest.mark.asyncio
async def test_evaluator_chunking_v3_parsing_error():
    """
    Test the map-reduce pattern when the LLM returns invalid JSON.
    """
    mock_llm = AsyncMock(spec=LLMClient)
    evaluator = EvaluatorAgentChunkingV3(llm=mock_llm, chunk_size=50)

    output_text = "Single chunk."

    with patch("magda_agent.agents.sub_agent.SubAgent.execute") as mock_execute:
        # Invalid JSON
        mock_execute.return_value = 'This is not JSON.'

        result = await evaluator.evaluate_output(output_text, "Test request", {})

        assert result["approved"] is False
        assert result["score"] == 0.0
        assert "Evaluation error" in result["feedback"]
