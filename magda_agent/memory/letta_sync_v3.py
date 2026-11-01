import json
import logging
import os
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from magda_agent.llm_client import LLMClient
    from magda_agent.memory.episodic import EpisodicMemory
    from magda_agent.memory.procedural import ProceduralMemory

logger = logging.getLogger(__name__)


class LettaVirtualContextAdvancedSyncV3:
    """
    Advanced synchronization hook V3 inspired by MemGPT/Letta patterns.
    Pushes procedural memory distilled from virtual context streams to disk in real-time.
    Supports asynchronous context management, heuristic & LLM extraction, and ContextPlugin hooks.
    """

    def __init__(
        self,
        storage_dir: str = "./data/letta_sync_v3",
        procedural_memory: Optional["ProceduralMemory"] = None,
        episodic_memory: Optional["EpisodicMemory"] = None,
        llm_client: Optional["LLMClient"] = None,
        user_id: Optional[int] = None,
        confidence_threshold: float = 0.5,
        auto_sync: bool = True,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.procedural_memory = procedural_memory
        self.episodic_memory = episodic_memory
        self.llm_client = llm_client
        self.user_id = user_id
        self.confidence_threshold = confidence_threshold
        self.auto_sync = auto_sync
        self.synced_procedures: List[Dict[str, Any]] = []

        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create storage directory {self.storage_dir}: {e}")

    async def __aenter__(self) -> "LettaVirtualContextAdvancedSyncV3":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if exc_type is None and self.auto_sync:
            await self.extract_and_sync_routines(user_id=self.user_id)

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
        Extract routine patterns and step-by-step procedures from virtual context streams.
        """
        routines: List[Dict[str, Any]] = []
        if not records:
            return routines

        normalized_records = [self._normalize_chunk(r) for r in records]

        header_keywords = [
            "routine:",
            "workflow:",
            "procedure:",
            "pattern:",
            "routine v3:",
            "pipeline:",
        ]
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
                        confidence = min(0.98, 0.65 + 0.05 * len(current_steps))
                        routines.append({
                            "name": current_title,
                            "procedure": "\n".join(current_steps),
                            "confidence": confidence,
                            "tags": current_tags or ["routine", "heuristic", "letta_v3"],
                        })
                        current_steps = []
                        current_tags = []

                    title_part = line.split(":", 1)[-1].strip()
                    current_title = title_part if title_part else "Extracted Routine V3"
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
                        "name": "Sequential Workflow Routine V3",
                        "procedure": procedure_text if len(steps) > 1 else line,
                        "confidence": 0.8,
                        "tags": ["sequential", "auto-extracted", "v3"],
                    })

        if current_steps and current_title:
            confidence = min(0.98, 0.65 + 0.05 * len(current_steps))
            routines.append({
                "name": current_title,
                "procedure": "\n".join(current_steps),
                "confidence": confidence,
                "tags": current_tags or ["routine", "heuristic", "letta_v3"],
            })

        if not routines and normalized_records:
            combined = "\n".join(normalized_records)
            routines.append({
                "name": "Virtual Context Real-Time Routine V3",
                "procedure": combined,
                "confidence": 0.55,
                "tags": ["virtual_context", "fallback", "v3"],
            })

        return [r for r in routines if r.get("confidence", 0.0) >= self.confidence_threshold]

    def _sanitize_filename(self, name: str) -> str:
        safe_chars = [c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip()]
        filename = "".join(safe_chars).strip("_")
        return filename[:64] if filename else "procedure"

    async def write_procedure_to_disk(
        self,
        procedure_record: Dict[str, Any],
        user_id: Optional[int] = None,
    ) -> Path:
        """
        Asynchronously writes distilled procedural memory to disk in real-time.
        """
        self._ensure_storage_dir()
        name = procedure_record.get("name", "procedure")
        target_user = user_id if user_id is not None else self.user_id
        user_prefix = f"user_{target_user}_" if target_user is not None else ""
        filename = f"{user_prefix}{self._sanitize_filename(name)}.json"
        target_path = self.storage_dir / filename

        payload = dict(procedure_record)
        payload["user_id"] = target_user
        payload["version"] = "v3"

        # Try async file I/O via aiofiles if available, otherwise synchronous fallback
        try:
            import aiofiles
            async with aiofiles.open(target_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(payload, indent=2, ensure_ascii=False))
        except ImportError:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed writing procedure to disk {target_path}: {e}")
            raise

        logger.info(f"Persisted procedural routine to disk: {target_path}")
        return target_path

    async def sync_procedures_to_disk(
        self,
        procedures: List[Dict[str, Any]],
        user_id: Optional[int] = None,
    ) -> List[Path]:
        """
        Persists a list of procedural records to disk in real-time.
        """
        written_paths: List[Path] = []
        for proc in procedures:
            path = await self.write_procedure_to_disk(proc, user_id=user_id)
            written_paths.append(path)
            self.synced_procedures.append(proc)
        return written_paths

    async def process_context_stream(
        self,
        stream: List[Union[str, Dict[str, Any]]],
        user_id: Optional[int] = None,
        sync_to_disk: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process a virtual context stream chunk-by-chunk, extract routines,
        optionally save into ProceduralMemory, and flush to disk in real-time.
        """
        target_user_id = user_id if user_id is not None else self.user_id
        routines = self.extract_patterns_heuristically(stream)

        stored_routines: List[Dict[str, Any]] = []
        for routine in routines:
            name = routine.get("name", "Extracted Routine V3")
            procedure = routine.get("procedure", "")
            confidence = routine.get("confidence", 0.75)
            tags = routine.get("tags", [])

            if procedure:
                if self.procedural_memory:
                    self.procedural_memory.store_procedure(
                        name=name,
                        procedure=procedure,
                        metadata={
                            "extracted_by": "LettaVirtualContextAdvancedSyncV3",
                            "version": "v3",
                            "confidence": confidence,
                            "tags": ",".join(tags) if isinstance(tags, list) else str(tags),
                        },
                        user_id=target_user_id,
                    )
                stored_routines.append(routine)

        if sync_to_disk and stored_routines:
            await self.sync_procedures_to_disk(stored_routines, user_id=target_user_id)

        return stored_routines

    async def extract_and_sync_routines(
        self,
        user_id: Optional[int] = None,
        query: str = "routine procedure workflow step action v3",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Queries episodic memory for routines, extracts procedures (LLM or heuristic),
        stores in procedural memory, and flushes to disk in real-time.
        """
        target_user_id = user_id if user_id is not None else self.user_id
        records: List[str] = []

        if self.episodic_memory:
            records = self.episodic_memory.recall_events(
                query=query, top_k=top_k, user_id=target_user_id
            )

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
                    "name": "LLM Extracted Routine V3",
                    "procedure": llm_response,
                    "confidence": 0.95,
                    "tags": ["llm", "v3", "real-time"],
                })
            except Exception as e:
                logger.warning(f"LLM routine extraction failed, falling back to heuristic V3: {e}")
                routines = self.extract_patterns_heuristically(records)
        else:
            routines = self.extract_patterns_heuristically(records)

        stored_routines: List[Dict[str, Any]] = []
        for routine in routines:
            name = routine.get("name", "Extracted Routine V3")
            procedure = routine.get("procedure", "")
            confidence = routine.get("confidence", 0.85)
            tags = routine.get("tags", [])

            if procedure:
                if self.procedural_memory:
                    self.procedural_memory.store_procedure(
                        name=name,
                        procedure=procedure,
                        metadata={
                            "extracted_by": "LettaVirtualContextAdvancedSyncV3",
                            "version": "v3",
                            "confidence": confidence,
                            "tags": ",".join(tags) if isinstance(tags, list) else str(tags),
                        },
                        user_id=target_user_id,
                    )
                stored_routines.append(routine)

        if stored_routines:
            await self.sync_procedures_to_disk(stored_routines, user_id=target_user_id)

        return stored_routines

    # --- ContextPlugin and lifecycle hooks ---
    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """
        Hook called on context updates to trigger real-time inspection.
        """
        if isinstance(new_context, list):
            # Extract heuristics and synchronously note or schedule
            routines = self.extract_patterns_heuristically(new_context)
            for r in routines:
                if r not in self.synced_procedures:
                    self.synced_procedures.append(r)
        logger.debug(f"LettaVirtualContextAdvancedSyncV3 context updated for user {user_id}")

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """
        Context compaction hook: automatically extracts routines from items about to be compacted/evicted,
        stores them in procedural memory, and writes them to disk before eviction.
        """
        user_id = metadata.get("user_id", self.user_id)
        if context_items:
            stream_entries = [
                getattr(item, "content", str(item)) for item in context_items
            ]
            await self.process_context_stream(stream_entries, user_id=user_id, sync_to_disk=True)

        return context_items

    def load_synced_procedures(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Reads and returns all procedures persisted to disk for a given user.
        """
        self._ensure_storage_dir()
        loaded: List[Dict[str, Any]] = []
        target_user = user_id if user_id is not None else self.user_id

        for filepath in self.storage_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if target_user is None or data.get("user_id") == target_user:
                        loaded.append(data)
            except Exception as e:
                logger.warning(f"Failed to read synced procedure file {filepath}: {e}")

        return loaded


# Aliases for naming conventions
LettaAdvancedSyncV3 = LettaVirtualContextAdvancedSyncV3
