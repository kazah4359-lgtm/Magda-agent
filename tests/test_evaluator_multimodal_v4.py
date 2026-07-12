import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.evaluator_multimodal_v4 import EvaluatorMultimodalV4

@pytest.mark.asyncio
async def test_evaluator_v4_agent_approve_text_only():
    """Tests the EvaluatorMultimodalV4 correctly parses a passing score from the LLM (text only)."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='{"score": 9, "approved": true, "feedback": "Great response!"}')

    agent = EvaluatorMultimodalV4(llm=mock_llm)
    result = await agent.evaluate_generator_output("The sky is blue.", "What color is the sky?")

    assert result["score"] == 9
    assert result["approved"] is True
    assert result["feedback"] == "Great response!"

    # Check messages
    args, kwargs = mock_llm.chat_completion.call_args
    messages = args[0]
    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "text"


@pytest.mark.asyncio
async def test_evaluator_v4_agent_with_images():
    """Tests the EvaluatorMultimodalV4 includes images in the LLM request."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='{"score": 8, "approved": true, "feedback": "Looks like a dog."}')

    agent = EvaluatorMultimodalV4(llm=mock_llm)
    images = ["base64_encoded_dog_image_1", "base64_encoded_dog_image_2"]
    result = await agent.evaluate_generator_output("It is a dog.", "What is in the image?", images=images)

    assert result["score"] == 8
    assert result["approved"] is True
    assert result["feedback"] == "Looks like a dog."

    # Check messages
    args, kwargs = mock_llm.chat_completion.call_args
    messages = args[0]
    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert isinstance(content, list)
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,base64_encoded_dog_image_1"
    assert content[2]["type"] == "image_url"
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,base64_encoded_dog_image_2"


@pytest.mark.asyncio
async def test_evaluator_v4_agent_reject():
    """Tests the EvaluatorMultimodalV4 correctly parses a failing score from the LLM with markdown."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='```json\n{"score": 4, "approved": false, "feedback": "Incorrect color."}\n```')

    agent = EvaluatorMultimodalV4(llm=mock_llm)
    result = await agent.evaluate_generator_output("The sky is green.", "What color is the sky?")

    assert result["score"] == 4
    assert result["approved"] is False
    assert result["feedback"] == "Incorrect color."
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_evaluator_v4_agent_error_handling_invalid_json():
    """Tests the EvaluatorMultimodalV4 correctly handles JSON parsing errors."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='This is not valid JSON')

    agent = EvaluatorMultimodalV4(llm=mock_llm)
    result = await agent.evaluate_generator_output("Some output", "Some request")

    assert result["score"] == 0
    assert result["approved"] is False
    assert "Evaluation error" in result["feedback"] or "Expecting value" in result["feedback"]
    mock_llm.chat_completion.assert_called_once()
