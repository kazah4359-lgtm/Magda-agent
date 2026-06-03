import pytest
from fastapi.testclient import TestClient
from magda_agent.consciousness.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_process_consciousness():
    payload = {"input_text": "Hello, world!"}
    response = client.post("/process", json=payload)
    assert response.status_code == 200
    assert response.json() == {"thoughts": "Thinking about: Hello, world!"}
