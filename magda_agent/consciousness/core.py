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
from magda_agent.exploration.curiosity import CuriosityExplorer
from magda_agent.drives.hypothalamus import Hypothalamus
from magda_agent.emotions.insula import Insula
from magda_agent.reflexes.brainstem import Brainstem
from magda_agent.rhythms.pineal_gland import PinealGland
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.emotions.style_adapter import StyleAdapter
from magda_agent.user_model.model import UserModel
from magda_agent.learning.online import OnlineLearner
from magda_agent.attention.salience import SalienceNetwork
from magda_agent.attention.workspace import GlobalWorkspace
from magda_agent.context.engine import ContextEngine
from magda_agent.learning.skill_creator import SkillCreator
from magda_agent.tracing.tracer import ThoughtChainTracer

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
        mirror_neurons: Optional[MirrorNeurons] = None,
        salience: Optional[SalienceNetwork] = None,
        global_workspace: Optional[GlobalWorkspace] = None,
        context_engine: Optional[ContextEngine] = None,
        skill_creator: Optional[SkillCreator] = None,
        online_learner: Optional[OnlineLearner] = None,
        tracer: Optional[ThoughtChainTracer] = None,
        style_adapter: Optional[StyleAdapter] = None,
        user_model: Optional[UserModel] = None
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
        self.curiosity_explorer = CuriosityExplorer()
        self.hypothalamus = hypothalamus
        self.insula = insula
        self.brainstem = brainstem
        self.pineal_gland = pineal_gland
        self.mirror_neurons = mirror_neurons
        self.salience = salience
        self.global_workspace = global_workspace
        self.context_engine = context_engine
        self.skill_creator = skill_creator
        self.online_learner = online_learner
        self.tracer = tracer
        self.style_adapter = style_adapter
        self.user_model = user_model

        if self.global_workspace:
            self.global_workspace.register_listener(self._broadcast_event)


    def _broadcast_event(self, event: Dict[str, Any]) -> None:
        """
        Receives broadcasted focus events from the Global Workspace
        and distributes the context to relevant subsystems.
        """
        logging.info(f"Consciousness broadcasting event: {event.get('type')}")
        # Simulate emotional arousal on focus shift as part of context processing
        if self.emotions:
            score = event.get('_salience_score', 0.0)
            # Arousal slightly increases based on the salience of the focused event
            a_delta = min(0.1, score * 0.1)
            self.emotions.update(p_delta=0.0, a_delta=a_delta, d_delta=0.0, user_id=None)

    async def process_input(self, user_input: str, user_id: Optional[int] = None) -> str:
        logging.info(f"Consciousness processing: {user_input}")
        if self.tracer:
            self.tracer.add_step("input_received", {"user_input": user_input, "user_id": user_id})

        # Use ContextEngine ingest hook if available
        if self.context_engine:
            user_input = await self.context_engine.ingest(user_input, {"user_id": user_id})

        if self.online_learner:
            # For simplicity, we use the planner's last state or a generic string as the action context
            last_context = self.planner.get_state_summary() if getattr(self, 'planner', None) else "Recent action context"
            await self.online_learner.process_feedback(user_input, last_context, user_id)

        if self.thalamus and not self.thalamus.filter_input(user_input):
            return "Message ignored by Thalamus."

        focus_content = user_input
        if self.global_workspace:
            # 0. Global Workspace selection
            # Clear previous candidates
            self.global_workspace.clear()

            # Add user input as candidate
            main_event = {"type": "user_input", "content": user_input, "urgency": 0.5}
            self.global_workspace.add_candidate(main_event)

            # If we had other sub-systems adding events, they would do so here.
            # E.g. self.global_workspace.add_candidate(boredom_event)

            focused_event = self.global_workspace.select_focus()

            if focused_event:
                focus_content = str(focused_event.get("content", focus_content))
                score = focused_event.get("_salience_score", 0.0)
                explanation = focused_event.get("_salience_explanation", "")
                logging.info(f"Workspace focused on event '{focused_event.get('type')}' with Salience: {score:.2f} ({explanation})")
                if self.tracer:
                    self.tracer.add_step("global_workspace_focus", {"event_type": focused_event.get('type'), "salience": score, "explanation": explanation})

            # Only user_input is supported for full processing in the current API,
            # but using focus_content ensures workspace output is piped in.
            user_input = focus_content
        elif self.salience:
            # Fallback if no workspace but salience exists
            event = {"content": user_input}
            score, explanation = self.salience.score_event(event)
            logging.info(f"Salience score: {score:.2f} ({explanation})")
            if self.tracer:
                self.tracer.add_step("salience_scoring", {"salience": score, "explanation": explanation})

        # 0. Brainstem Autonomic Reflexes
        if self.brainstem:
            reflex_response = self.brainstem.process_reflex(user_input)
            if reflex_response:
                logging.info(f"Brainstem reflex triggered for: {user_input}")
                return reflex_response

        # 1. Perception & Emotion Update (Initial reaction)
        # For simplicity, we just slightly increase arousal when receiving input
        self.emotions.update(0.01, 0.05, 0.01, user_id=user_id)

        if self.mirror_neurons:
            p_shift, a_shift, d_shift = self.mirror_neurons.empathize(user_input)
            if p_shift != 0.0 or a_shift != 0.0 or d_shift != 0.0:
                self.emotions.update(p_shift, a_shift, d_shift, user_id=user_id)

                # Curiosity-driven exploration
        if self.hypothalamus and self.curiosity_explorer:
            if self.curiosity_explorer.should_explore(self.hypothalamus.boredom):
                exploration_tasks = self.curiosity_explorer.explore()
                # If we have a global workspace, we could post these tasks there.
                # For now, we'll log them and potentially add them to the system prompt if needed.
                logging.info(f"Curiosity triggered. Proposed tasks: {exploration_tasks}")

                # We decrease boredom slightly just to show we acted on it
                if len(exploration_tasks) > 0:
                     self.hypothalamus.update(1.0)

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
                self.emotions.update(v_shift, a_shift, d_shift, user_id=user_id)

        # 2. Memory Retrieval
        relevant_memories = self.memory.retrieve_relevant(user_input, user_id=user_id)
        if self.tracer:
            self.tracer.add_step("memory_retrieval", {"retrieved_count": len(relevant_memories)})

        # Use ContextEngine assemble hook if available
        if self.context_engine:
            context_str = await self.context_engine.assemble(relevant_memories, {"user_id": user_id})
        else:
            context_str = "\n".join([f"- {m.content}" for m in relevant_memories])

        # Cross-session continuity: Check if this is a new session (no active working memory)
        working_memory_entries = self.memory.working_memory.get_entries(user_id=user_id)
        if len(working_memory_entries) == 0:
            past_episodes = self.memory.episodic_memory.recall_events(user_input, top_k=3, user_id=user_id)
            if past_episodes:
                context_str += "\n\nPast Relevant Episodes:\n" + "\n".join([f"- {ep}" for ep in past_episodes])

        if self.long_term_memory:
            long_term_memories = self.long_term_memory.recall(user_input, user_id=user_id)
            if long_term_memories:
                context_str += "\nLong Term Memories:\n" + "\n".join([f"- {m}" for m in long_term_memories])

        # 3. Planning (Prefrontal Cortex)
        plan_str = ""
        if self.tracer:
            self.tracer.add_step("planning_start", {})
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
                else:
                    if self.skill_creator and len(self.planner.completed_steps) > 1:
                        # Extract skill candidate from successful multi-step plan
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

        # 4. LLM Reasoning
        if self.tracer:
            self.tracer.add_step("llm_reasoning_start", {"plan_str": plan_str})
        emotion_summary = self.emotions.get_summary(user_id=user_id)
        if self.hypothalamus:
            emotion_summary += f" | {self.hypothalamus.get_drives_summary()}"

        if self.pineal_gland:
            emotion_summary += f" | Time of day: {self.pineal_gland.get_time_context()}"

        if self.attachment:
            self.attachment.record_interaction(user_id)
            attachment_prompt = self.attachment.get_attachment_prompt(user_id)
            if attachment_prompt:
                emotion_summary += f"\n{attachment_prompt}"

        system_prompt = self.llm.get_system_prompt(
            context=context_str,
            emotions=emotion_summary
        )

        if self.style_adapter:
            um = None
            if self.user_model and user_id is not None:
                um = self.user_model.get_model(user_id)
            pad_state = self.emotions.get_state_history(user_id)[0]
            style_modifier = self.style_adapter.get_style_prompt(pad_state, um)
            if style_modifier:
                system_prompt += f"\n\n{style_modifier}"

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
                if self.tracer:
                    self.tracer.add_step("action_selection", {"selected_action": "ignore"})
                return "Message ignored by Basal Ganglia."
            elif selected_action and self.tracer:
                self.tracer.add_step("action_selection", {"selected_action": selected_action["action"]})

        response = await self.llm.chat_completion(messages)
        if self.tracer:
            self.tracer.add_step("response_generated", {"response": response})

        # 5. Post-processing & Memory Storage
        memory_content = f"User said: {user_input} | I replied: {response}"
        await self.memory.add_memory(
            content=memory_content,
            importance=0.5,
            emotional_state=self.emotions.get_state_history(user_id)[0],
            tags=["conversation"],
            user_id=user_id
        )

        if self.long_term_memory:
            self.long_term_memory.store(text=memory_content, metadata={"type": "conversation"}, user_id=user_id)

        # Gradual emotional decay after processing
        self.emotions.decay(user_id=user_id)

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
