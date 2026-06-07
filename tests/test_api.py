import pytest
from fastapi.testclient import TestClient
from magda_agent.api import app

client = TestClient(app)
AUTH_HEADERS = {"Authorization": "Bearer test-token"}

class MockLLMClient:
    def get_system_prompt(self, *args, **kwargs):
        return "Mock prompt"
    async def chat_completion(self, *args, **kwargs):
        return "Mocked response from LLM"

@pytest.fixture(autouse=True)
def mock_llm_client(monkeypatch):
    monkeypatch.setenv("MAGDA_API_TOKEN", "test-token")
    mock_instance = MockLLMClient()
    monkeypatch.setattr("magda_agent.api.consciousness.llm", mock_instance)
    monkeypatch.setattr("magda_agent.api.subconsciousness.llm", mock_instance)

def test_state_endpoint():
    response = client.get("/state", headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert "state" in response.json()

def test_healthcheck_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_process_endpoint():
    # Start up the app tasks
    with TestClient(app) as client_with_startup:
        response = client_with_startup.post("/process", json={"text": "Hello"}, headers=AUTH_HEADERS)
        assert response.status_code == 200
        assert "response" in response.json()


def test_process_endpoint_requires_auth():
    response = client.post("/process", json={"text": "Hello"})
    assert response.status_code == 401
