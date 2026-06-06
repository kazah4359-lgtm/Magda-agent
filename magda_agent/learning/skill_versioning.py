import json
import logging
from typing import Dict, Any, Optional

from magda_agent.memory.procedural import ProceduralMemory


class SkillVersioning:
    """
    Manages versioning and usage tracking for skills in ProceduralMemory.
    When a skill's outcome improves, a new version can be created.
    Tracks usage outcomes per version to select the best one.
    """

    def __init__(self, procedural_memory: ProceduralMemory) -> None:
        self.procedural_memory = procedural_memory

    def get_best_version(self, skill_name: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves the version of a skill with the best outcome score.
        If no outcomes are tracked, returns the highest version number.
        """
        versions = self.procedural_memory.get_procedure_versions(skill_name, user_id=user_id)
        if not versions:
            return None

        best_version = None
        best_score = -float('inf')
        highest_version_num = -1
        highest_version_doc = None

        for idx, meta in enumerate(versions['metadatas']):
            version_num = meta.get("version", 1)
            doc = {
                "name": meta.get("name", skill_name),
                "procedure": versions['documents'][idx].split("Procedure: ")[-1] if "Procedure: " in versions['documents'][idx] else versions['documents'][idx],
                "version": version_num,
                "metadata": meta
            }

            if version_num > highest_version_num:
                highest_version_num = version_num
                highest_version_doc = doc

            outcomes_str = meta.get("usage_outcomes", "[]")
            try:
                outcomes = json.loads(outcomes_str)
            except json.JSONDecodeError:
                outcomes = []

            # Calculate score: for simplicity, score = number of successful outcomes minus failures
            score = 0
            for outcome in outcomes:
                if outcome.get("success"):
                    score += 1
                else:
                    score -= 1

            if score > best_score:
                best_score = score
                best_version = doc

        # If all scores are equal (e.g. 0), fallback to highest version
        if best_version is None or best_score == 0:
            return highest_version_doc

        return best_version

    def record_usage_outcome(
        self,
        skill_name: str,
        version: int,
        success: bool,
        details: str = "",
        user_id: Optional[int] = None
    ) -> None:
        """
        Records the outcome of using a specific version of a skill.
        Updates the usage_outcomes metadata for that version.
        """
        versions = self.procedural_memory.get_procedure_versions(skill_name, user_id=user_id)
        if not versions or not versions.get('metadatas'):
            logging.warning(f"Skill '{skill_name}' version {version} not found for recording outcome.")
            return

        for idx, meta in enumerate(versions['metadatas']):
            if meta.get("version", 1) == version:
                doc_id = versions['ids'][idx]
                outcomes_str = meta.get("usage_outcomes", "[]")
                try:
                    outcomes = json.loads(outcomes_str)
                except json.JSONDecodeError:
                    outcomes = []

                outcomes.append({
                    "success": success,
                    "details": details
                })

                meta["usage_outcomes"] = json.dumps(outcomes)

                # Update the document in ChromaDB
                try:
                    self.procedural_memory.collection.update(
                        ids=[doc_id],
                        metadatas=[meta]
                    )
                    logging.info(f"Recorded outcome for skill '{skill_name}' version {version}: success={success}")
                except Exception as e:
                    logging.error(f"Failed to update skill outcome in ChromaDB: {e}")
                return

        logging.warning(f"Skill '{skill_name}' version {version} not found in versions list.")

    def create_new_version(
        self,
        skill_name: str,
        new_procedure: str,
        base_version: int,
        user_id: Optional[int] = None
    ) -> int:
        """
        Creates a new version of an existing skill.
        Returns the new version number.
        """
        versions = self.procedural_memory.get_procedure_versions(skill_name, user_id=user_id)

        highest_version = 0
        if versions and versions.get('metadatas'):
            for meta in versions['metadatas']:
                v = meta.get("version", 1)
                if v > highest_version:
                    highest_version = v

        new_version = highest_version + 1

        metadata = {
            "version": new_version,
            "base_version": base_version,
            "usage_outcomes": "[]"
        }

        self.procedural_memory.store_procedure(
            name=skill_name,
            procedure=new_procedure,
            metadata=metadata,
            user_id=user_id
        )

        logging.info(f"Created new version {new_version} for skill '{skill_name}' (based on v{base_version})")
        return new_version
