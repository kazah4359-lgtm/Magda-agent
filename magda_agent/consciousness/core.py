import logging
from typing import List, Dict, Any, Optional
from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.skills.registry import SkillRegistry
from magda_agent.planning.planner import Planner
from magda_agent.memory.long_term import LongTermMemory
from magda_agent.metacognition.evaluator import Evaluator
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.attachment import AttachmentModel
from magda_agent.thalamus.router import Thalamus
from magda_agent.action.selector import BasalGanglia
from magda_agent.drives.hypothalamus import Hypothalamus
from magda_agent.emotions.insula import Insula
from magda_agent.reflexes.brainstem import Brainstem
from magda_agent.rhythms.pineal_gland import PinealGland
from magda_agent.emotions.mirror_neurons import MirrorNeurons

class Consciousness:
    """
    The main cognitive loop of the AGI agent.
    Responsible for perception, memory retrieval, emotional processing, and response generation.
    """
    def __init__(
        self,
        llm: LLMClient,
        emotions: EmotionalEngine,
        memory: MemorySystem,
        skills: SkillRegistry,
        planner: Optional[Planner] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        evaluator: Optional[Evaluator] = None,
        habit_tracker: Optional[HabitTracker] = None,
        attachment: Optional[AttachmentModel] = None,
        thalamus: Optional[Thalamus] = None,
        basal_ganglia: Optional[BasalGanglia] = None,
        hypothalamus: Optional[Hypothalamus] = None,
        insula: Optional[Insula] = None,
        brainstem: Optional[Brainstem] = None,
        pineal_gland: Optional[PinealGland] = None,
        mirror_neurons: Optional[MirrorNeurons] = None
    ):
        self.llm = llm
        self.emotions = emotions
        self.memory = memory
        self.skills = skills
        self.planner = planner
        self.long_term_memory = long_term_memory
        self.evaluator = evaluator
        self.habit_tracker = habit_tracker
        self.attachment = attachment
        self.thalamus = thalamus
        self.basal_ganglia = basal_ganglia
        self.hypothalamus = hypothalamus
        self.insula = insula
        self.brainstem = brainstem
        self.pineal_gland = pineal_gland
        self.mirror_neurons = mirror_neurons

    async def process_input(self, user_input: str, user_id: Optional[int] = None) -> str:
        logging.info(f"Consciousness processing: {user_input}")

        if self.thalamus and not self.thalamus.filter_input(user_input):
            return "Message ignored by Thalamus."

        # 0. Brainstem Autonomic Reflexes
        if self.brainstem:
            reflex_response = self.brainstem.process_reflex(user_input)
            if reflex_response:
                logging.info(f"Brainstem reflex triggered for: {user_input}")
                return reflex_response

        # 1. Perception & Emotion Update (Initial reaction)
        # For simplicity, we just slightly increase arousal when receiving input
        self.emotions.update(0.01, 0.05, 0.01)

        if self.mirror_neurons:
            p_shift, a_shift, d_shift = self.mirror_neurons.empathize(user_input)
            if p_shift != 0.0 or a_shift != 0.0 or d_shift != 0.0:
                self.emotions.update(p_shift, a_shift, d_shift)

        if self.hypothalamus:
            activity_level = 1.0
            if self.pineal_gland:
                # Modulate energy drain based on time of day (e.g. morning = higher modifier, less relative drain)
                modifier = self.pineal_gland.get_energy_modifier()
                activity_level = activity_level / modifier

            self.hypothalamus.update(activity_level) # Activity modulated by time of day


            if self.insula:
                v_shift, a_shift, d_shift = self.insula.process_interoception(
                    self.hypothalamus.energy,
                    self.hypothalamus.boredom
                )
                self.emotions.update(v_shift, a_shift, d_shift)

        # 2. Memory Retrieval
        relevant_memories = self.memory.retrieve_relevant(user_input, user_id=user_id)
        context_str = "\n".join([f"- {m.content}" for m in relevant_memories])

        if self.long_term_memory:
            long_term_memories = self.long_term_memory.recall(user_input, user_id=user_id)
            if long_term_memories:
                context_str += "\nLong Term Memories:\n" + "\n".join([f"- {m}" for m in long_term_memories])

        # 3. Planning (Prefrontal Cortex)
        plan_str = ""
        if self.planner:
            # Only generate a new plan if we don't have an active one
            if not self.planner.get_current_plan():
                await self.planner.generate_plan(user_input, user_id=user_id)

            plan = self.planner.get_current_plan()
            if plan:
                import asyncio

                MAX_STEPS = 5
                SKILL_TIMEOUT = 10.0
                steps_executed = 0
                plan_stopped_early = False

                # Execute the steps and collect results
                while self.planner.get_current_plan() and steps_executed < MAX_STEPS:
                    steps_executed += 1
                    step = self.planner.get_current_plan()[0]
                    skill_name = step.get('skill')
                    kwargs = step.get('skill_kwargs') or {}

                    if skill_name:
                        # Execute heavy skills in a separate thread to avoid blocking the event loop
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

                    if plan_stopped_early:
                        break

                if self.planner.get_current_plan() and steps_executed >= MAX_STEPS:
                    plan_stopped_early = True
                    logging.warning("Plan execution stopped due to MAX_STEPS limit.")

                if plan_stopped_early:
                    self.planner.clear_pending_plan()

                plan_str = "Executed Plan Results:\n"
                for i, step in enumerate(self.planner.completed_steps):
                    plan_str += f"- Step {i+1}: {step.get('description')} (Skill: {step.get('skill')})\n"
                    plan_str += f"  Result: {step.get('result')}\n"

                if plan_stopped_early:
                    plan_str += "\nNote: Plan execution was stopped early due to limits.\n"

        # 4. LLM Reasoning
        emotion_summary = self.emotions.get_summary()
        if self.hypothalamus:
            emotion_summary += f" | {self.hypothalamus.get_drives_summary()}"

        if self.pineal_gland:
            emotion_summary += f" | Time of day: {self.pineal_gland.get_time_context()}"

        if self.attachment and user_id is not None:
            self.attachment.record_interaction(user_id)
            attachment_prompt = self.attachment.get_attachment_prompt(user_id)
            if attachment_prompt:
                emotion_summary += f"\n{attachment_prompt}"

        system_prompt = self.llm.get_system_prompt(
            context=context_str,
            emotions=emotion_summary
        )
        if plan_str:
            system_prompt += f"\n\n{plan_str}\nUse the plan results to generate the final response."

        if self.evaluator:
            eval_feedback = self.evaluator.get_feedback_for_prompt()
            if eval_feedback:
                system_prompt += f"\n\n{eval_feedback}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # Determine if a skill should be used (Simplified logic for PoC)
        # In a real system, the LLM would decide which tool to call.

        # Action Selection using Basal Ganglia if available
        if self.basal_ganglia:
            possible_actions = [
                {"action": "chat", "priority": 10},
                {"action": "ignore", "priority": 1}
            ]
            selected_action = self.basal_ganglia.select_action(possible_actions)
            if selected_action and selected_action["action"] == "ignore":
                return "Message ignored by Basal Ganglia."

        response = await self.llm.chat_completion(messages)

        # 5. Post-processing & Memory Storage
        memory_content = f"User said: {user_input} | I replied: {response}"
        self.memory.add_memory(
            content=memory_content,
            importance=0.5,
            emotional_state=self.emotions.state,
            tags=["conversation"],
            user_id=user_id
        )

        if self.long_term_memory:
            self.long_term_memory.store(text=memory_content, metadata={"type": "conversation"}, user_id=user_id)

        # Gradual emotional decay after processing
        self.emotions.decay()

        # 6. Metacognition (Self-Evaluation)
        if self.evaluator:
            await self.evaluator.evaluate_response(user_input, response)

            # Record habit if we have an evaluation and a tracker
            if self.habit_tracker and self.evaluator.last_evaluation:
                avg_score = self.evaluator.last_evaluation.get("average_score", 0.0)
                if self.planner and self.planner.completed_steps:
                    for step in self.planner.completed_steps:
                        skill = step.get("skill")
                        if skill:
                            self.habit_tracker.record_usage(user_input, skill, float(avg_score), user_id=user_id)

        return response

    def get_internal_state(self) -> str:
        planner_state = self.planner.get_state_summary() if self.planner else "Planner: Not available"
        return f"""
{self.emotions.get_summary()}
{self.memory.get_summary()}
{self.skills.get_skills_summary()}
{planner_state}
"""
