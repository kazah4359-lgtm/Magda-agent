import pytest
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.learning.skill_versioning import SkillVersioning

@pytest.fixture
def procedural_memory(tmp_path):
    """Fixture to provide an ephemeral ProceduralMemory instance."""
    return ProceduralMemory(persist_directory=":memory:")

@pytest.fixture
def skill_versioning(procedural_memory):
    """Fixture to provide a SkillVersioning instance."""
    return SkillVersioning(procedural_memory)

def test_create_and_get_best_version(skill_versioning, procedural_memory):
    """Test creating versions and retrieving the best one based on usage outcomes."""
    # Create an initial version
    procedural_memory.store_procedure("test_skill", "Procedure v1", metadata={"version": 1, "usage_outcomes": "[]"})

    # Record some failures for v1
    skill_versioning.record_usage_outcome("test_skill", 1, success=False)
    skill_versioning.record_usage_outcome("test_skill", 1, success=False)

    # Create a new version
    v2 = skill_versioning.create_new_version("test_skill", "Procedure v2 (improved)", base_version=1)
    assert v2 == 2

    # Record some successes for v2
    skill_versioning.record_usage_outcome("test_skill", 2, success=True)
    skill_versioning.record_usage_outcome("test_skill", 2, success=True)

    # Best version should now be v2
    best = skill_versioning.get_best_version("test_skill")
    assert best is not None
    assert best["version"] == 2
    assert best["procedure"] == "Procedure v2 (improved)"

def test_get_best_version_fallback(skill_versioning, procedural_memory):
    """Test getting the highest version when there are no outcomes or equal scores."""
    procedural_memory.store_procedure("test_skill2", "Proc v1", metadata={"version": 1})
    procedural_memory.store_procedure("test_skill2", "Proc v2", metadata={"version": 2})

    best = skill_versioning.get_best_version("test_skill2")
    assert best is not None
    assert best["version"] == 2

def test_record_usage_outcome_creates_json(skill_versioning, procedural_memory):
    """Test that recording outcome serializes into json correctly."""
    procedural_memory.store_procedure("json_skill", "Do stuff", metadata={"version": 1})

    skill_versioning.record_usage_outcome("json_skill", 1, success=True, details="Worked great")

    versions = procedural_memory.get_procedure_versions("json_skill")
    meta = versions['metadatas'][0]
    import json
    outcomes = json.loads(meta["usage_outcomes"])
    assert len(outcomes) == 1
    assert outcomes[0]["success"] is True
    assert outcomes[0]["details"] == "Worked great"
