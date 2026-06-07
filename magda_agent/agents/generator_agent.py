import logging
import asyncio
from typing import Optional, List, Dict, Any

from magda_agent.llm_client import LLMClient
from magda_agent.skills.registry import SkillRegistry
from magda_agent.planning.planner import Planner
from magda_agent.learning.skill_versioning import SkillVersioning
from magda_agent.learning.skill_creator import SkillCreator
from magda_agent.safety.guardrails import RealtimeGuardrail, FallbackStrategy

class GeneratorAgent:
    """
    Agent responsible for executing plan steps and generating final text response.
    """
    def __init__(
        self,
        llm: LLMClient,
        skills: SkillRegistry,
        planner: Optional[Planner] = None,
        skill_versioning: Optional[SkillVersioning] = None,
        skill_creator: Optional[SkillCreator] = None,
        guardrail: Optional[RealtimeGuardrail] = None,
        tracer=None
    ):
        self.llm = llm
        self.skills = skills
        self.planner = planner
        self.skill_versioning = skill_versioning
        self.skill_creator = skill_creator
        self.guardrail = guardrail
        self.tracer = tracer

    async def execute_plan(self, user_input: str, user_id: Optional[str] = None) -> str:
        """
        Executes the plan step by step and returns the string representation of results.
        """
        plan_str = ""
        if not self.planner:
            return plan_str

        plan = self.planner.get_current_plan()
        if plan:
            MAX_STEPS = 5
            SKILL_TIMEOUT = 10.0
            steps_executed = 0
            plan_stopped_early = False

            while self.planner.get_current_plan() and steps_executed < MAX_STEPS:
                steps_executed += 1
                step = self.planner.get_current_plan()[0]
                skill_name = step.get('skill')
                kwargs = step.get('skill_kwargs') or {}

                if skill_name:
                    # Real-time guardrail check before execution
                    if self.guardrail:
                        allowed, explanation, strategy = self.guardrail.check_action(skill_name, **kwargs)
                        if not allowed:
                            if strategy == FallbackStrategy.STOP_EXECUTION:
                                result = f"Guardrail Fallback (STOP): {explanation}"
                                plan_stopped_early = True
                                logging.warning(f"Plan stopped by guardrail: {explanation}")
                            elif strategy == FallbackStrategy.REQUEST_REVIEW:
                                result = f"Guardrail Fallback (REVIEW REQUIRED): {explanation}"
                                plan_stopped_early = True # Also stop for now until human intervention
                                logging.warning(f"Plan paused for review by guardrail: {explanation}")
                            else:
                                result = f"Guardrail Denied: {explanation}"

                            self.planner.mark_step_completed(0, str(result))
                            break

                    try:
                        task = asyncio.to_thread(self.skills.execute_skill, skill_name, **kwargs)
                        result = await asyncio.wait_for(task, timeout=SKILL_TIMEOUT)
                    except asyncio.TimeoutError:
                        logging.error(f"Timeout executing skill {skill_name}")
                        result = f"Error: Skill {skill_name} timed out after {SKILL_TIMEOUT} seconds."
                        plan_stopped_early = True
                    except Exception as e:
                        logging.error(f"Error executing skill {skill_name}: {e}")
                        result = f"Error: {e}"
                else:
                    result = "No skill executed for this step."

                self.planner.mark_step_completed(0, str(result))
                if self.skill_versioning and skill_name:
                    success = 'Error:' not in str(result)
                    best = self.skill_versioning.get_best_version(skill_name, user_id=user_id)
                    if best:
                        self.skill_versioning.record_usage_outcome(skill_name, best['version'], success, str(result), user_id=user_id)

                if plan_stopped_early:
                    break

            if self.planner.get_current_plan() and steps_executed >= MAX_STEPS:
                plan_stopped_early = True
                logging.warning("Plan execution stopped due to MAX_STEPS limit.")

            if plan_stopped_early:
                self.planner.clear_pending_plan()
            else:
                if self.skill_creator and len(self.planner.completed_steps) > 1:
                    asyncio.create_task(
                        self.skill_creator.extract_and_store_skill(
                            user_input,
                            self.planner.completed_steps,
                            user_id=user_id
                        )
                    )

            plan_str = "Executed Plan Results:\n"
            for i, step in enumerate(self.planner.completed_steps):
                plan_str += f"- Step {i+1}: {step.get('description')} (Skill: {step.get('skill')})\n"
                plan_str += f"  Result: {step.get('result')}\n"

            if plan_stopped_early:
                plan_str += "\nNote: Plan execution was stopped early due to limits.\n"

        return plan_str

    async def generate_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generates the final response based on context and executed plan results.
        """
        response = await self.llm.chat_completion(messages)
        if self.tracer:
            self.tracer.add_step("response_generated", {"response": response})
        return response
