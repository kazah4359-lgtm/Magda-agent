import pytest
from unittest.mock import AsyncMock, patch
from magda_agent.agents.code_review import CodeReviewWorker
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    client = LLMClient(api_key="fake-key", model="fake-model")
    client.chat_completion = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_review_pr_valid_json(mock_llm_client):
    """
    Tests that review_pr correctly parses a valid JSON response from the LLM.
    """
    worker = CodeReviewWorker(llm=mock_llm_client)

    # Mocking SubAgent.execute instead of chat_completion directly because CodeReviewWorker calls self.execute
    # Let's mock execute directly on the worker or patch it.
    with patch.object(worker, "execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = '''
        ```json
        [
            {"comment_id": "main.py:42", "reply": "This variable `x` is unused."},
            {"comment_id": "tests/test_main.py:10", "reply": "Missing assertions here."}
        ]
        ```
        '''

        diff = "diff --git a/main.py b/main.py\n+x = 10\n"
        comments = await worker.review_pr(diff)

        mock_execute.assert_called_once_with(task=diff, context="Please review the following PR diff:\n\n")
        assert len(comments) == 2
        assert comments[0]["comment_id"] == "main.py:42"
        assert comments[0]["reply"] == "This variable `x` is unused."
        assert comments[1]["comment_id"] == "tests/test_main.py:10"
        assert comments[1]["reply"] == "Missing assertions here."

@pytest.mark.asyncio
async def test_review_pr_invalid_json(mock_llm_client):
    """
    Tests that review_pr handles invalid JSON gracefully.
    """
    worker = CodeReviewWorker(llm=mock_llm_client)

    with patch.object(worker, "execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "I think the code looks good!"

        diff = "diff --git a/main.py b/main.py\n+y = 20\n"
        comments = await worker.review_pr(diff)

        mock_execute.assert_called_once()
        assert len(comments) == 0

@pytest.mark.asyncio
async def test_review_pr_missing_fields(mock_llm_client):
    """
    Tests that review_pr filters out comments missing required fields.
    """
    worker = CodeReviewWorker(llm=mock_llm_client)

    with patch.object(worker, "execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = '''
        [
            {"comment_id": "file.py:1", "reply": "Valid."},
            {"reply": "Missing comment_id."},
            {"comment_id": "file.py:2"}
        ]
        '''

        diff = "diff --git a/main.py b/main.py\n+z = 30\n"
        comments = await worker.review_pr(diff)

        mock_execute.assert_called_once()
        assert len(comments) == 1
        assert comments[0]["comment_id"] == "file.py:1"
        assert comments[0]["reply"] == "Valid."
