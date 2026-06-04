import pytest
from magda_agent.safety.risk_system import RiskSystem

def test_risk_system_initialization():
    """Test that the RiskSystem initializes correctly."""
    rs = RiskSystem()
    assert rs is not None

def test_classify_file_change_workflows():
    """Test that changes to workflows are high risk."""
    rs = RiskSystem()
    assert rs.classify_file_change(".github/workflows/main.yml") == "high"
    assert rs.classify_file_change(".github/workflows/deploy.yaml") == "high"

def test_classify_file_change_requirements():
    """Test that changes to requirements are high risk."""
    rs = RiskSystem()
    assert rs.classify_file_change("requirements.txt") == "high"

def test_classify_file_change_docs():
    """Test that docs-only changes are low risk."""
    rs = RiskSystem()
    assert rs.classify_file_change("docs/cognitive_architecture.md") == "low"
    assert rs.classify_file_change("docs/api.md") == "low"
    assert rs.classify_file_change("README.md") == "low"
    assert rs.classify_file_change("backlog.md") == "low"

def test_classify_file_change_sandbox_and_registry():
    """Test that sandbox and skill registry changes are high risk."""
    rs = RiskSystem()
    assert rs.classify_file_change("magda_agent/skills/registry.py") == "high"
    assert rs.classify_file_change("magda_agent/skills/system_execute_code.py") == "high"

def test_classify_file_change_critical():
    """Test that changes to secrets or deployment are critical risk."""
    rs = RiskSystem()
    assert rs.classify_file_change("secrets/api_keys.json") == "critical"
    assert rs.classify_file_change(".env") == "critical"
    assert rs.classify_file_change("deployment/docker-compose.prod.yml") == "critical"

def test_classify_file_change_medium():
    """Test that standard code changes default to medium risk."""
    rs = RiskSystem()
    assert rs.classify_file_change("magda_agent/consciousness/core.py") == "medium"
    assert rs.classify_file_change("magda_agent/api.py") == "medium"
    assert rs.classify_file_change("tests/test_api.py") == "medium"
    assert rs.classify_file_change("agent_tasks.json") == "medium"

def test_classify_changes_overall():
    """Test that classify_changes returns the highest risk level from a list of files."""
    rs = RiskSystem()

    # Empty list
    assert rs.classify_changes([]) == "low"

    # All low
    assert rs.classify_changes(["docs/index.md", "README.md"]) == "low"

    # Low and medium
    assert rs.classify_changes(["docs/index.md", "magda_agent/api.py"]) == "medium"

    # Low, medium, high
    assert rs.classify_changes(["docs/index.md", "magda_agent/api.py", "requirements.txt"]) == "high"

    # Low, medium, high, critical
    assert rs.classify_changes(["docs/index.md", "magda_agent/api.py", "requirements.txt", ".env"]) == "critical"
