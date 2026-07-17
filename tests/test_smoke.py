import pytest
from fastapi.testclient import TestClient
import pytest
from pytest import MonkeyPatch
from magda_agent.api import app

def test_agent_startup_teardown(monkeypatch: MonkeyPatch) -> None:
    """
    Test basic agent startup and teardown via FastAPI's lifespan events.
    """
    monkeypatch.setenv("MAGDA_API_TOKEN", "test-token")

    # Using TestClient as a context manager triggers the lifespan events
    # (startup and shutdown)
    with TestClient(app) as client:
        # Check that the application is running by calling the health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
