import pytest
from unittest.mock import patch, MagicMock
from magda_agent.evaluation.smoke import SmokeEvaluator

def test_smoke_evaluator_success():
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "tests passed"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        evaluator = SmokeEvaluator()
        result = evaluator.evaluate()

        assert result["success"] is True
        assert result["returncode"] == 0
        assert result["stdout"] == "tests passed"
        mock_run.assert_called_once()

def test_smoke_evaluator_failure():
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "tests failed"
        mock_result.stderr = "error details"
        mock_run.return_value = mock_result

        evaluator = SmokeEvaluator()
        result = evaluator.evaluate()

        assert result["success"] is False
        assert result["returncode"] == 1
        assert result["stdout"] == "tests failed"
        assert result["stderr"] == "error details"

def test_smoke_evaluator_exception():
    with patch("subprocess.run", side_effect=Exception("Command not found")):
        evaluator = SmokeEvaluator(command="invalid_cmd")
        result = evaluator.evaluate()

        assert result["success"] is False
        assert result["returncode"] == -1
        assert "Command not found" in result["stderr"]

from magda_agent.evaluation.smoke import PostMergeSmokeWorkflow

def test_post_merge_smoke_workflow_success():
    evaluator = SmokeEvaluator()
    evaluator.evaluate = MagicMock(return_value={"success": True, "returncode": 0, "stdout": "", "stderr": "", "command": ""})
    workflow = PostMergeSmokeWorkflow(evaluator=evaluator)
    assert workflow.run_workflow() is True

def test_post_merge_smoke_workflow_failure():
    evaluator = SmokeEvaluator()
    evaluator.evaluate = MagicMock(return_value={"success": False, "returncode": 1, "stdout": "", "stderr": "error", "command": ""})
    workflow = PostMergeSmokeWorkflow(evaluator=evaluator)
    assert workflow.run_workflow() is False
