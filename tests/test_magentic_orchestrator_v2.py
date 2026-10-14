import pytest
import asyncio
import json
from unittest.mock import AsyncMock
from magda_agent.llm_client import LLMClient
from magda_agent.architecture.magentic_orchestrator_v2 import (
    MagenticBlackboardV2,
    MagenticWorkerV2,
    MagenticOrchestratorV2,
)


def test_blackboard_concurrent_updates_race_condition_free():
    async def run_test():
        blackboard = MagenticBlackboardV2(initial_facts=["Initial fact"])

        async def worker_add_fact(index: int):
            await asyncio.sleep(0.001)
            await blackboard.add_fact(f"Fact from worker {index}", author=f"Worker-{index}")
            await blackboard.set_value(f"key_{index}", index, author=f"Worker-{index}")

        # Launch 50 concurrent updates
        tasks = [worker_add_fact(i) for i in range(50)]
        await asyncio.gather(*tasks)

        facts = await blackboard.get_facts()
        snapshot = await blackboard.snapshot()

        # 1 initial fact + 50 facts added = 51 facts total
        assert len(facts) == 51
        assert len(snapshot["state"]) == 50
        assert len(snapshot["history"]) == 100  # 50 facts + 50 set_values

    asyncio.run(run_test())


def test_magentic_orchestrator_v2_success_flow():
    async def run_test():
        mock_llm = AsyncMock(spec=LLMClient)

        plan_json = json.dumps([
            {"id": "step1", "description": "Search info", "worker": "WebSurfer"},
            {"id": "step2", "description": "Write code", "worker": "Coder"}
        ])

        mock_llm.chat_completion.side_effect = [
            plan_json,               # Plan step
            "Web search result",     # WebSurfer exec
            "Code generated result", # Coder exec
            "YES Final task complete" # Review step
        ]

        orchestrator = MagenticOrchestratorV2(llm=mock_llm)
        result = await orchestrator.orchestrate("Build simple web search feature")

        assert result == "YES Final task complete"
        assert mock_llm.chat_completion.call_count == 4

        facts = await orchestrator.blackboard.get_facts()
        assert len(facts) == 2
        assert "[WebSurfer] Web search result" in facts[0]
        assert "[Coder] Code generated result" in facts[1]

    asyncio.run(run_test())


def test_magentic_orchestrator_v2_parallel_execution():
    async def run_test():
        mock_llm = AsyncMock(spec=LLMClient)

        plan_json = json.dumps([
            {"id": "step1", "description": "Gather dataset A"},
            {"id": "step2", "description": "Gather dataset B"},
            {"id": "step3", "description": "Gather dataset C"}
        ])

        mock_llm.chat_completion.side_effect = [
            plan_json,
            "Data A ready",
            "Data B ready",
            "Data C ready",
            "YES All data gathered"
        ]

        orchestrator = MagenticOrchestratorV2(llm=mock_llm)
        result = await orchestrator.orchestrate("Gather data in parallel", parallel=True)

        assert result == "YES All data gathered"
        assert mock_llm.chat_completion.call_count == 5

        facts = await orchestrator.blackboard.get_facts()
        assert len(facts) == 3

    asyncio.run(run_test())


def test_magentic_orchestrator_v2_max_iterations():
    async def run_test():
        mock_llm = AsyncMock(spec=LLMClient)

        mock_llm.chat_completion.side_effect = [
            '[{"id": "1", "description": "Attempt task"}]', "Result step", "NO Not done yet",
            '[{"id": "1", "description": "Attempt task"}]', "Result step", "NO Not done yet",
        ]

        orchestrator = MagenticOrchestratorV2(llm=mock_llm)
        result = await orchestrator.orchestrate("Unresolvable task", max_iterations=2)

        assert "Task incomplete after 2 iterations" in result
        assert mock_llm.chat_completion.call_count == 6

    asyncio.run(run_test())


def test_magentic_orchestrator_v2_invalid_plan_fallback():
    async def run_test():
        mock_llm = AsyncMock(spec=LLMClient)

        mock_llm.chat_completion.side_effect = [
            "THIS IS NOT VALID JSON",
            "Fallback execution complete",
            "YES Complete"
        ]

        orchestrator = MagenticOrchestratorV2(llm=mock_llm)
        result = await orchestrator.orchestrate("Do simple task")

        assert result == "YES Complete"
        assert mock_llm.chat_completion.call_count == 3

    asyncio.run(run_test())


def test_worker_error_handling():
    async def run_test():
        mock_llm = AsyncMock(spec=LLMClient)
        mock_llm.chat_completion.side_effect = Exception("LLM connection timeout")

        blackboard = MagenticBlackboardV2()
        worker = MagenticWorkerV2("Tester", "Tester role", mock_llm)

        res = await worker.execute_subtask("Run test suite", blackboard)
        assert "Worker Tester error: LLM connection timeout" in res

        facts = await blackboard.get_facts()
        assert len(facts) == 1
        assert "Worker Tester error" in facts[0]

    asyncio.run(run_test())
