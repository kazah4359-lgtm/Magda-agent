import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from magda_agent.llm_client import LLMClient
    from magda_agent.memory.episodic import EpisodicMemory
    from magda_agent.memory.procedural import ProceduralMemory

logger = logging.getLogger(__name__)


class LettaVirtualContextRoutineBuilderV2:
    """
    Advanced routine generator hook V2 inspired by MemGPT/Letta patterns.
    Dynamically builds procedural memory from virtual context streams and episodic logs.
    """

    def __init__(
        self,
        procedural_memory: Optional["ProceduralMemory"] = None,
        episodic_memory: Optional["EpisodicMemory"] = None,
        llm_client: Optional["LLMClient"] = None,
        user_id: Optional[int] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.procedural_memory = procedural_memory
        self.episodic_memory = episodic_memory
        self.llm_client = llm_client
        self.user_id = user_id
        self.confidence_threshold = confidence_threshold

    async def __aenter__(self) -> "LettaVirtualContextRoutineBuilderV2":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if exc_type is None:
            await self.extract_and_store_routines(user_id=self.user_id)

    def _normalize_chunk(self, chunk: Union[str, Dict[str, Any]]) -> str:
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            content = chunk.get("content") or chunk.get("text") or chunk.get("record") or ""
            role = chunk.get("role", "")
            return f"[{role}] {content}" if role else str(content)
        return str(chunk)

    def extract_patterns_heuristically(
        self, records: List[Union[str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Extract routine patterns and step-by-step procedures from context streams heuristically.
        Calculates confidence scores, tags, and step sequences.
        """
        routines: List[Dict[str, Any]] = []
        if not records:
            return routines

        normalized_records = [self._normalize_chunk(r) for r in records]

        header_keywords = ["routine:", "workflow:", "procedure:", "pattern:", "routine v2:"]
        step_prefixes = ("-", "*", "1.", "2.", "3.", "4.", "5.", "step ", "step:", "action:")

        current_steps: List[str] = []
        current_title = ""
        current_tags: List[str] = []

        for record in normalized_records:
            lines = [line.strip() for line in record.splitlines() if line.strip()]
            for line in lines:
                lower = line.lower()
                if any(kw in lower for kw in header_keywords):
                    if current_steps and current_title:
                        confidence = min(0.95, 0.6 + 0.05 * len(current_steps))
                        routines.append({
                            "name": current_title,
                            "procedure": "\n".join(current_steps),
                            "confidence": confidence,
                            "tags": current_tags or ["routine", "heuristic"],
                        })
                        current_steps = []
                        current_tags = []

                    title_part = line.split(":", 1)[-1].strip()
                    current_title = title_part if title_part else "Extracted Routine V2"
                    current_steps.append(line)
                    if "#" in line:
                        current_tags.extend([tag.strip("#") for tag in line.split() if tag.startswith("#")])
                elif current_title and (
                    any(lower.startswith(prefix) for prefix in step_prefixes) or " step " in lower
                ):
                    current_steps.append(line)
                elif " then " in lower or " -> " in lower or "=>" in lower:
                    steps = [s.strip() for s in line.replace("=>", "->").split("->") if s.strip()]
                    procedure_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
                    routines.append({
                        "name": "Sequential Workflow Routine V2",
                        "procedure": procedure_text if len(steps) > 1 else line,
                        "confidence": 0.75,
                        "tags": ["sequential", "auto-extracted"],
                    })

        if current_steps and current_title:
            confidence = min(0.95, 0.6 + 0.05 * len(current_steps))
            routines.append({
                "name": current_title,
                "procedure": "\n".join(current_steps),
                "confidence": confidence,
                "tags": current_tags or ["routine", "heuristic"],
            })

        if not routines and normalized_records:
            combined = "\n".join(normalized_records)
            routines.append({
                "name": "Virtual Context Virtual Routine V2",
                "procedure": combined,
                "confidence": 0.5,
                "tags": ["virtual_context", "fallback"],
            })

        return [r for r in routines if r.get("confidence", 0.0) >= self.confidence_threshold]

    async def process_context_stream(
        self,
        stream: List[Union[str, Dict[str, Any]]],
        user_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a virtual context stream chunk-by-chunk to extract and store procedural routines.
        """
        target_user_id = user_id if user_id is not None else self.user_id
        routines = self.extract_patterns_heuristically(stream)

        stored_routines: List[Dict[str, Any]] = []
        if self.procedural_memory:
            for routine in routines:
                name = routine.get("name", "Extracted Routine V2")
                procedure = routine.get("procedure", "")
                confidence = routine.get("confidence", 0.7)
                tags = routine.get("tags", [])
                if procedure:
                    self.procedural_memory.store_procedure(
                        name=name,
                        procedure=procedure,
                        metadata={
                            "extracted_by": "LettaVirtualContextRoutineBuilderV2",
                            "version": "v2",
                            "confidence": confidence,
                            "tags": ",".join(tags),
                        },
                        user_id=target_user_id,
                    )
                    stored_routines.append(routine)

        return stored_routines

    async def extract_and_store_routines(
        self,
        user_id: Optional[int] = None,
        query: str = "routine procedure workflow step action v2",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Queries episodic memory or virtual context for routine patterns, extracts step-by-step
        procedures (via LLM or heuristic V2 fallback), and stores them into procedural memory.
        """
        target_user_id = user_id if user_id is not None else self.user_id
        records: List[str] = []

        if self.episodic_memory:
            records = self.episodic_memory.recall_events(query=query, top_k=top_k, user_id=target_user_id)

        if not records:
            return []

        routines: List[Dict[str, Any]] = []

        if self.llm_client:
            try:
                combined = "\n---\n".join(records)
                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "Extract reusable procedural routines from virtual context streams and episodic logs. "
                            "Output clean routine names, step-by-step procedures, and structured instructions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Virtual context records:\n{combined}",
                    },
                ]
                llm_response = await self.llm_client.chat_completion(prompt)
                routines.append({
                    "name": "LLM Extracted Routine V2",
                    "procedure": llm_response,
                    "confidence": 0.9,
                    "tags": ["llm", "v2"],
                })
            except Exception as e:
                logger.warning(f"LLM V2 routine extraction failed, falling back to heuristic V2: {e}")
                routines = self.extract_patterns_heuristically(records)
        else:
            routines = self.extract_patterns_heuristically(records)

        stored_routines: List[Dict[str, Any]] = []
        if self.procedural_memory:
            for routine in routines:
                name = routine.get("name", "Extracted Routine V2")
                procedure = routine.get("procedure", "")
                confidence = routine.get("confidence", 0.8)
                tags = routine.get("tags", [])
                if procedure:
                    self.procedural_memory.store_procedure(
                        name=name,
                        procedure=procedure,
                        metadata={
                            "extracted_by": "LettaVirtualContextRoutineBuilderV2",
                            "version": "v2",
                            "confidence": confidence,
                            "tags": ",".join(tags) if isinstance(tags, list) else str(tags),
                        },
                        user_id=target_user_id,
                    )
                    stored_routines.append(routine)

        return stored_routines


# Alias for versioning conventions
LettaRoutineBuilderV2 = LettaVirtualContextRoutineBuilderV2
