import logging
import json
from typing import List, Dict, Optional, Any

from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


class SkillImproverV1:
    """
    Hermes Agent Experience Skill Improvement V1.
    Creates and refines skills dynamically from episodic memory/experience logs.
    """

    def __init__(
        self,
        procedural_memory: Optional[ProceduralMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.procedural_memory = procedural_memory
        self.episodic_memory = episodic_memory
        self.llm_client = llm_client
        self._stats = {
            "distilled_skills": 0,
            "refined_skills": 0,
            "failed_distillations": 0,
        }

    async def distill_skills_from_experience(
        self,
        user_id: Optional[int] = None,
        experience_logs: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Extracts and distills reusable procedural skills from episodic experience.
        """
        logs_to_process: List[Dict[str, Any]] = []

        if experience_logs:
            logs_to_process = experience_logs
        elif self.episodic_memory:
            if query:
                recalled = self.episodic_memory.recall_events(query=query, top_k=limit, user_id=user_id)
                logs_to_process = [{"text": text} for text in recalled]
            else:
                events = self.episodic_memory.get_all_events(user_id=user_id, limit=limit)
                logs_to_process = events

        if not logs_to_process:
            logger.info("No episodic logs found for skill distillation.")
            return []

        formatted_logs = []
        for idx, item in enumerate(logs_to_process):
            if isinstance(item, dict):
                text = item.get("text") or item.get("action") or str(item)
            else:
                text = str(item)
            formatted_logs.append(f"Episode {idx + 1}: {text}")

        logs_str = "\n".join(formatted_logs)

        distilled_results: List[Dict[str, Any]] = []

        if self.llm_client:
            prompt = f"""
Analyze the following episodic execution logs and distill one or more reusable procedural skills.

Logs:
{logs_str}

Return a valid JSON array of objects, where each object has keys:
- "name": short identifier for the skill
- "procedure": step-by-step procedure description or Python function definition
- "metadata": dict containing "source": "distilled_from_episodic_memory"
"""
            try:
                if hasattr(self.llm_client, "generate"):
                    response = await self.llm_client.generate(prompt)
                elif hasattr(self.llm_client, "chat_completion"):
                    response = await self.llm_client.chat_completion([{"role": "user", "content": prompt}])
                else:
                    response = str(self.llm_client)

                if response and isinstance(response, str):
                    clean_res = response.strip()
                    if clean_res.startswith("```json"):
                        clean_res = clean_res[7:].strip()
                    if clean_res.startswith("```"):
                        clean_res = clean_res[3:].strip()
                    if clean_res.endswith("```"):
                        clean_res = clean_res[:-3].strip()

                    try:
                        parsed = json.loads(clean_res)
                        if isinstance(parsed, dict):
                            parsed = [parsed]
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "name" in item and "procedure" in item:
                                    distilled_results.append(item)
                    except json.JSONDecodeError:
                        distilled_results.append({
                            "name": "distilled_skill_from_memory",
                            "procedure": clean_res,
                            "metadata": {"source": "distilled_from_episodic_memory"}
                        })
            except Exception as e:
                logger.error(f"Error during LLM skill distillation: {e}")
                self._stats["failed_distillations"] += 1
        else:
            distilled_results.append({
                "name": f"distilled_skill_heuristic_{len(logs_to_process)}",
                "procedure": f"Distilled procedure from {len(logs_to_process)} episodes:\n" + logs_str[:200],
                "metadata": {"source": "heuristic_distillation", "count": len(logs_to_process)}
            })

        for skill in distilled_results:
            name = skill.get("name", "unnamed_distilled_skill")
            procedure = skill.get("procedure", "")
            meta = skill.get("metadata", {})
            if self.procedural_memory:
                self.procedural_memory.store_procedure(
                    name=name,
                    procedure=procedure,
                    user_id=user_id,
                    metadata=meta
                )
            self._stats["distilled_skills"] += 1

        return distilled_results

    async def refine_skill_from_feedback(
        self,
        skill_name: str,
        feedback: str,
        user_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Refines an existing skill using feedback signals and stores the updated version in procedural memory.
        """
        current_procedure = ""
        if self.procedural_memory:
            procedures = self.procedural_memory.recall_procedure(query=skill_name, user_id=user_id)
            if procedures:
                current_procedure = procedures[0]

        refined_procedure = ""
        if self.llm_client:
            prompt = f"""
Refine the following skill procedure based on user feedback.

Skill Name: {skill_name}
Current Procedure: {current_procedure}
Feedback: {feedback}

Provide ONLY the updated procedure text or code.
"""
            try:
                if hasattr(self.llm_client, "generate"):
                    refined_procedure = await self.llm_client.generate(prompt)
                elif hasattr(self.llm_client, "chat_completion"):
                    refined_procedure = await self.llm_client.chat_completion([{"role": "user", "content": prompt}])
            except Exception as e:
                logger.error(f"Failed to refine skill with LLM: {e}")

        if not refined_procedure:
            refined_procedure = f"{current_procedure}\n\n# Refined based on feedback: {feedback}"

        refined_skill = {
            "name": skill_name,
            "procedure": refined_procedure,
            "metadata": {
                "refined": True,
                "feedback": feedback,
                "source": "skill_improver_v1"
            }
        }

        if self.procedural_memory:
            self.procedural_memory.store_procedure(
                name=skill_name,
                procedure=refined_procedure,
                user_id=user_id,
                metadata=refined_skill["metadata"]
            )

        self._stats["refined_skills"] += 1
        return refined_skill

    def get_improver_stats(self) -> Dict[str, int]:
        """Returns statistics on skill distillations and refinements."""
        return dict(self._stats)
