import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.learning.skill_improver_v1 import SkillImproverV1
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.memory.episodic import EpisodicMemory


@pytest.fixture
def procedural_memory():
    return ProceduralMemory(persist_directory=":memory:")


@pytest.fixture
def episodic_memory():
    return EpisodicMemory(persist_directory=":memory:")


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value='''```json
[
  {
    "name": "parse_json_logs",
    "procedure": "def parse_logs(data): return [json.loads(x) for x in data]",
    "metadata": {"source": "distilled_from_episodic_memory", "tags": ["json", "logs"]}
  }
]
```''')
    return llm


@pytest.mark.asyncio
async def test_distill_skills_from_experience_llm(procedural_memory, mock_llm):
    improver = SkillImproverV1(
        procedural_memory=procedural_memory,
        llm_client=mock_llm,
    )

    logs = [
        {"action": "Read log file", "outcome": "Success"},
        {"action": "Parse JSON records", "outcome": "Success"},
    ]

    skills = await improver.distill_skills_from_experience(
        user_id=1,
        experience_logs=logs,
    )

    assert len(skills) == 1
    assert skills[0]["name"] == "parse_json_logs"
    assert "parse_logs" in skills[0]["procedure"]

    stored = procedural_memory.recall_procedure("parse_json_logs", user_id=1)
    assert len(stored) > 0
    assert "parse_logs" in stored[0]

    stats = improver.get_improver_stats()
    assert stats["distilled_skills"] == 1


@pytest.mark.asyncio
async def test_distill_skills_fallback_no_llm(procedural_memory):
    improver = SkillImproverV1(
        procedural_memory=procedural_memory,
        llm_client=None,
    )

    logs = [{"action": "Execute pipeline", "outcome": "Success"}]

    skills = await improver.distill_skills_from_experience(
        user_id=2,
        experience_logs=logs,
    )

    assert len(skills) == 1
    assert skills[0]["name"].startswith("distilled_skill_heuristic_")
    assert "Distilled procedure from 1 episodes" in skills[0]["procedure"]

    stats = improver.get_improver_stats()
    assert stats["distilled_skills"] == 1


@pytest.mark.asyncio
async def test_distill_skills_from_episodic_memory(procedural_memory, episodic_memory, mock_llm):
    episodic_memory.store_event("User requested report generation. Steps: query SQL, generate PDF.", user_id=10)

    improver = SkillImproverV1(
        procedural_memory=procedural_memory,
        episodic_memory=episodic_memory,
        llm_client=mock_llm,
    )

    skills = await improver.distill_skills_from_experience(user_id=10)

    assert len(skills) == 1
    mock_llm.generate.assert_called_once()
    stats = improver.get_improver_stats()
    assert stats["distilled_skills"] == 1


@pytest.mark.asyncio
async def test_refine_skill_from_feedback(procedural_memory, mock_llm):
    procedural_memory.store_procedure(
        name="data_processing",
        procedure="def process(data): return data",
        user_id=5,
    )

    mock_llm.generate = AsyncMock(return_value="def process(data):\n    return [x.strip() for x in data if x]")

    improver = SkillImproverV1(
        procedural_memory=procedural_memory,
        llm_client=mock_llm,
    )

    refined = await improver.refine_skill_from_feedback(
        skill_name="data_processing",
        feedback="Handle empty elements and strip whitespaces",
        user_id=5,
    )

    assert refined is not None
    assert "strip()" in refined["procedure"]

    recalled = procedural_memory.recall_procedure("data_processing", user_id=5)
    assert len(recalled) > 0
    assert any("strip()" in doc for doc in recalled)

    stats = improver.get_improver_stats()
    assert stats["refined_skills"] == 1


@pytest.mark.asyncio
async def test_empty_experience_logs(procedural_memory):
    improver = SkillImproverV1(procedural_memory=procedural_memory)

    skills = await improver.distill_skills_from_experience(experience_logs=[])
    assert len(skills) == 0

    stats = improver.get_improver_stats()
    assert stats["distilled_skills"] == 0


@pytest.mark.asyncio
async def test_llm_exception_handling(procedural_memory):
    failing_llm = MagicMock()
    failing_llm.generate = AsyncMock(side_effect=RuntimeError("LLM quota exceeded"))

    improver = SkillImproverV1(
        procedural_memory=procedural_memory,
        llm_client=failing_llm,
    )

    logs = [{"action": "test", "outcome": "failed"}]
    skills = await improver.distill_skills_from_experience(experience_logs=logs)

    assert len(skills) == 0
    stats = improver.get_improver_stats()
    assert stats["failed_distillations"] == 1
