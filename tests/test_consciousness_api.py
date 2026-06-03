import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from magda_agent.consciousness.api import app

client = TestClient(app)

def test_get_state():
    with patch("magda_agent.consciousness.api.consciousness") as mock_consciousness:
        mock_consciousness.get_internal_state.return_value = "Mocked State"

        response = client.get("/state")
        assert response.status_code == 200
        assert response.json() == {"state": "Mocked State"}

def test_process_input():
    with patch("magda_agent.consciousness.api.consciousness") as mock_consciousness:
        mock_consciousness.process_input = AsyncMock(return_value="Mocked Response")

        response = client.post("/process", json={"text": "Hello Magda"})
        assert response.status_code == 200
        assert response.json() == {"response": "Mocked Response"}
        mock_consciousness.process_input.assert_called_once_with("Hello Magda")

def test_process_input_empty_text():
    response = client.post("/process", json={"text": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Text cannot be empty"
