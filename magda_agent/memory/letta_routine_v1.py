import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from types import TracebackType

if TYPE_CHECKING:
    from magda_agent.llm_client import LLMClient
    from magda_agent.memory.episodic import EpisodicMemory
    from magda_agent.memory.procedural import ProceduralMemory

logger = logging.getLogger(__name__)


class LettaVirtualContextRoutineBuilder:
    """
    LettaVirtualContextRoutineBuilder automatically extracts routine patterns from episodic memory records
    and populates procedural memory with reusable routines, inspired by MemGPT/Letta patterns.
    """

    def __init__(
        self,
        procedural_memory: Optional["ProceduralMemory"] = None,
        episodic_memory: Optional["EpisodicMemory"] = None,
        llm_client: Optional["LLMClient"] = None,
        user_id: Optional[int] = None,
    ) -> None:
        self.procedural_memory = procedural_memory
        self.episodic_memory = episodic_memory
        self.llm_client = llm_client
        self.user_id = user_id

    async def __aenter__(self) -> "LettaVirtualContextRoutineBuilder":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if exc_type is None:
            await self.extract_and_store_routines(user_id=self.user_id)

    def extract_patterns_heuristically(self, records: List[str]) -> List[Dict[str, str]]:
        """
        Fallback heuristic routine pattern extraction from episodic record text.
        Finds repeated action sequences or structured steps in episodic memory.
        """
        routines: List[Dict[str, str]] = []
        if not records:
            return routines

        # Identify explicit routine markers or repeated workflows
        current_routine_steps: List[str] = []
        current_routine_title = ""

        header_keywords = ["routine:", "workflow:", "procedure:", "pattern:"]

        for record in records:
            lines = [line.strip() for line in record.splitlines() if line.strip()]
            for line in lines:
                lower = line.lower()
                if any(kw in lower for kw in header_keywords):
                    if current_routine_steps and current_routine_title:
                        routines.append({
                            "name": current_routine_title,
                            "procedure": "\n".join(current_routine_steps),
                        })
                        current_routine_steps = []
                    current_routine_title = line.split(":", 1)[-1].strip() or "Extracted Routine"
                    current_routine_steps.append(line)
                elif current_routine_title and any(
                    line.startswith(prefix) for prefix in ["-", "*", "1.", "2.", "3.", "Step", "step"]
                ):
                    current_routine_steps.append(line)
                elif " then " in lower or " -> " in lower:
                    # Common sequential action pattern
                    title = "Sequential Action Routine"
                    routines.append({
                        "name": title,
                        "procedure": line,
                    })

        if current_routine_steps and current_routine_title:
            routines.append({
                "name": current_routine_title,
                "procedure": "\n".join(current_routine_steps),
            })

        # If no explicit markers found but records exist, aggregate into a default routine
        if not routines and records:
            routines.append({
                "name": "General Procedural Routine",
                "procedure": "\n".join(records),
            })

        return routines

    async def extract_and_store_routines(
        self,
        user_id: Optional[int] = None,
        query: str = "routine procedure workflow step action",
        top_k: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Queries episodic memory for potential routine records, extracts procedural pattern steps,
        and stores them into procedural memory.
        """
        target_user_id = user_id if user_id is not None else self.user_id
        records: List[str] = []

        if self.episodic_memory:
            records = self.episodic_memory.recall_events(query=query, top_k=top_k, user_id=target_user_id)

        if not records:
            return []

        routines: List[Dict[str, str]] = []

        if self.llm_client:
            try:
                combined_records = "\n---\n".join(records)
                prompt = [
                    {
                        "role": "system",
                        "content": "Extract reusable procedural routines from the episodic memory logs. Output routine names and step-by-step procedures.",
                    },
                    {
                        "role": "user",
                        "content": f"Episodic records:\n{combined_records}",
                    },
                ]
                llm_response = await self.llm_client.chat_completion(prompt)
                routines.append({
                    "name": "LLM Extracted Routine",
                    "procedure": llm_response,
                })
            except Exception as e:
                logger.warning(f"LLM routine extraction failed, falling back to heuristics: {e}")
                routines = self.extract_patterns_heuristically(records)
        else:
            routines = self.extract_patterns_heuristically(records)

        stored_routines: List[Dict[str, str]] = []
        if self.procedural_memory:
            for routine in routines:
                name = routine.get("name", "Extracted Routine")
                procedure = routine.get("procedure", "")
                if procedure:
                    self.procedural_memory.store_procedure(
                        name=name,
                        procedure=procedure,
                        metadata={"extracted_by": "LettaVirtualContextRoutineBuilder"},
                        user_id=target_user_id,
                    )
                    stored_routines.append({"name": name, "procedure": procedure})

        return stored_routines


# Alias for backward compatibility / versioning naming conventions
LettaRoutineBuilderV1 = LettaVirtualContextRoutineBuilder
