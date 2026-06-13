import logging
import asyncio
from typing import Optional, List, Dict, Any

from magda_agent.llm_client import LLMClient
from magda_agent.skills.registry import SkillRegistry
from magda_agent.planning.planner import Planner
from magda_agent.learning.skill_versioning import SkillVersioning
from magda_agent.learning.skill_creator import SkillCreator
from magda_agent.safety.guardrails import RealtimeGuardrail, FallbackStrategy
from magda_agent.skills.mcp_client import MCPClient


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
        mcp_client: Optional[MCPClient] = None,
        tracer=None
    ):
        self.llm = llm
        self.skills = skills
        self.planner = planner
        self.skill_versioning = skill_versioning
        self.skill_creator = skill_creator
        self.guardrail = guardrail
        self.mcp_client = mcp_client
        self.tracer = tracer

    async def execute_plan(self, user_input: str, user_id: Optional[str] = None) -> str:
        """
        Executes the plan step by step (supporting concurrency) and returns results.
        """
        plan_str = ""
        if not self.planner:
            return plan_str

        plan = self.planner.get_current_plan(user_id=user_id)
        if plan:
            MAX_STEPS = 10
            SKILL_TIMEOUT = 15.0
            steps_executed = 0
            plan_stopped_early = False

            while steps_executed < MAX_STEPS:
                executable_steps = self.planner.get_executable_steps(user_id=user_id)
                if not executable_steps or not isinstance(executable_steps, list):
                    break

                batch_tasks = []
                step_metadatas = []

                for step in executable_steps:
                    if steps_executed >= MAX_STEPS:
                        break

                    skill_name = step.get('skill')
                    kwargs = step.get('skill_kwargs') or {}
                    step_id = step.get('id')

                    if not step_id:
                        logging.warning(f"Step missing ID, skipping: {step.get('description')}")
                        continue

                    steps_executed += 1

                    if not skill_name:
                        self.planner.mark_step_id_completed(step_id, "No skill executed for this step.", user_id=user_id)
                        continue

                    # Guardrail check
                    if self.guardrail:
                        allowed, explanation, strategy = self.guardrail.check_action(skill_name, **kwargs)
                        if not allowed:
                            if strategy == FallbackStrategy.STOP_EXECUTION:
                                result = f"Guardrail Fallback (STOP): {explanation}"
                                plan_stopped_early = True
                            elif strategy == FallbackStrategy.REQUEST_REVIEW:
                                result = f"Guardrail Fallback (REVIEW REQUIRED): {explanation}"
                                plan_stopped_early = True
                            else:
                                result = f"Guardrail Denied: {explanation}"

                            self.planner.mark_step_id_completed(step_id, str(result), user_id=user_id)
                            if plan_stopped_early:
                                break
                            continue

                    # Execution Task
                    if hasattr(self, 'mcp_client') and self.mcp_client and self.mcp_client.has_tool(skill_name):
                        task = asyncio.create_task(self.mcp_client.execute_tool(skill_name, **kwargs))
                    else:
                        task = asyncio.create_task(asyncio.to_thread(self.skills.execute_skill, skill_name, **kwargs))

                    batch_tasks.append(asyncio.wait_for(task, timeout=SKILL_TIMEOUT))
                    step_metadatas.append({'id': step_id, 'skill': skill_name})

                    if plan_stopped_early:
                        break

                if not batch_tasks:
                    break

                # Run batch concurrently
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    meta = step_metadatas[i]
                    step_id = meta['id']
                    skill_name = meta['skill']

                    if isinstance(result, asyncio.TimeoutError):
                        result_str = f"Error: Skill {skill_name} timed out."
                        plan_stopped_early = True
                    elif isinstance(result, Exception):
                        result_str = f"Error: {result}"
                    else:
                        result_str = str(result)

                    self.planner.mark_step_id_completed(step_id, result_str, user_id=user_id)

                    # Skill Versioning
                    if self.skill_versioning:
                        success = 'Error:' not in result_str
                        best = self.skill_versioning.get_best_version(skill_name, user_id=user_id)
                        if best:
                            self.skill_versioning.record_usage_outcome(
                                skill_name, best['version'], success, result_str, user_id=user_id
                            )

                if plan_stopped_early:
                    break

            current_plan = self.planner.get_current_plan(user_id=user_id)
            if current_plan and steps_executed >= MAX_STEPS:
                plan_stopped_early = True
                logging.warning("Plan execution stopped due to MAX_STEPS limit.")
                plan_str += "\nPlan execution stopped due to MAX_STEPS limit."

            if plan_stopped_early:
                self.planner.clear_pending_plan(user_id=user_id)
            else:
                if self.skill_creator and len(self.planner.get_completed_steps(user_id=user_id)) > 1:
                    asyncio.create_task(
                        self.skill_creator.extract_and_store_skill(
                            user_input,
                            self.planner.get_completed_steps(user_id=user_id),
                            user_id=user_id
                        )
                    )

            plan_str_res = "Executed Plan Results:\n"
            for i, step in enumerate(self.planner.get_completed_steps(user_id=user_id)):
                plan_str_res += f"- Step {i+1}: {step.get('description')} (Skill: {step.get('skill')})\n"
                plan_str_res += f"  Result: {step.get('result')}\n"

            if plan_stopped_early:
                plan_str_res += "\nNote: Plan execution was stopped early due to limits.\n"

            plan_str = plan_str_res + plan_str

        return plan_str

    async def generate_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generates the final response based on context and executed plan results.
        """
        response = await self.llm.chat_completion(messages)
        if self.tracer:
            self.tracer.add_step("response_generated", {"response": response})
        return response
