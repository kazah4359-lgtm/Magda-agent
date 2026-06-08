import pytest
import sys
from unittest.mock import MagicMock, AsyncMock

# Mock transformers to prevent CI issues
sys.modules['transformers'] = MagicMock()

from magda_agent.consciousness.core import Consciousness
from magda_agent.memory.storage import MemorySystem
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.llm_client import LLMClient
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.skills.registry import SkillRegistry

@pytest.fixture
def mock_llm() -> MagicMock:
    """Provides a mock LLMClient instance."""
    llm = MagicMock(spec=LLMClient)
    llm.get_system_prompt.return_value = "System prompt"
    llm.chat_completion = AsyncMock(return_value="LLM Response")
    return llm

@pytest.fixture
def setup_consciousness(mock_llm: MagicMock, tmp_path: str) -> tuple[Consciousness, MemorySystem]:
    """Sets up a Consciousness instance with a mocked LLM and fresh MemorySystem."""
    persist_dir = str(tmp_path / "test_episodic_db")
    memory_system = MemorySystem(persist_directory=persist_dir, llm=mock_llm, short_term_limit=5)
    emotions = EmotionalEngine()
    skills = SkillRegistry()

    consciousness = Consciousness(
        llm=mock_llm,
        emotions=emotions,
        memory=memory_system,
        skills=skills
    )
    return consciousness, memory_system

@pytest.mark.asyncio
async def test_cross_session_continuity(setup_consciousness: tuple[Consciousness, MemorySystem]) -> None:
    """Verifies that past episodic memories are retrieved when starting a new session."""
    consciousness, memory_system = setup_consciousness
    user_id = 999

    # 1. Store a past episodic memory directly (simulating a previous session)
    memory_system.episodic_memory.store_event("My favorite color is blue.", user_id=user_id)

    # 2. Process input in a "new session" (working memory is empty)
    user_input = "What is my favorite color?"

    # Run the core process_input
    response = await consciousness.process_input(user_input, user_id=user_id)

    # Verify that the llm was called with the episodic memory context
    system_prompt_calls = consciousness.llm.get_system_prompt.call_args_list
    assert len(system_prompt_calls) > 0

    # We look for the "Past Relevant Episodes" in the context parameter passed to get_system_prompt
    context_str = system_prompt_calls[0].kwargs.get('context', '')

    assert "Past Relevant Episodes:" in context_str
    assert "blue" in context_str

@pytest.mark.asyncio
async def test_continuity_ignored_when_active_session(setup_consciousness: tuple[Consciousness, MemorySystem]) -> None:
    """Verifies that episodic continuity is not triggered if the session is already active."""
    consciousness, memory_system = setup_consciousness
    user_id = 888

    # 1. Store a past episodic memory directly (simulating a previous session)
    memory_system.episodic_memory.store_event("My favorite color is red.", user_id=user_id)

    # 2. Simulate an active session by populating working memory
    from magda_agent.memory.working import MemoryEntry
    from magda_agent.emotions.engine import PADState

    # Add memory through the official async way to properly handle the list limit/summarization
    await memory_system.add_memory("Hello", 0.5, PADState(0,0,0), ["tag"], user_id=user_id)

    # 3. Process input
    user_input = "What is my favorite color?"
    response = await consciousness.process_input(user_input, user_id=user_id)

    # Verify that the episodic memory was NOT included because working memory wasn't empty
    system_prompt_calls = consciousness.llm.get_system_prompt.call_args_list
    context_str = system_prompt_calls[1].kwargs.get('context', '') if len(system_prompt_calls) > 1 else system_prompt_calls[0].kwargs.get('context', '')

    assert "Past Relevant Episodes:" not in context_str
    assert "red" not in context_str
