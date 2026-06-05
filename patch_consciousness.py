import re

with open("magda_agent/consciousness/core.py", "r") as f:
    content = f.read()

# Add import
content = content.replace(
    "from magda_agent.context.engine import ContextEngine\nfrom magda_agent.learning.skill_creator import SkillCreator",
    "from magda_agent.context.engine import ContextEngine\nfrom magda_agent.learning.skill_creator import SkillCreator\nfrom magda_agent.tracing.tracer import ThoughtChainTracer"
)

# Add __init__ argument
content = content.replace(
    "        skill_creator: Optional[SkillCreator] = None,\n        online_learner: Optional[OnlineLearner] = None\n    ):",
    "        skill_creator: Optional[SkillCreator] = None,\n        online_learner: Optional[OnlineLearner] = None,\n        tracer: Optional[ThoughtChainTracer] = None\n    ):"
)
content = content.replace(
    "        self.online_learner = online_learner\n",
    "        self.online_learner = online_learner\n        self.tracer = tracer\n"
)

# Add tracing to process_input
# Start of process_input
content = content.replace(
    "        logging.info(f\"Consciousness processing: {user_input}\")",
    "        logging.info(f\"Consciousness processing: {user_input}\")\n        if self.tracer:\n            self.tracer.add_step(\"input_received\", {\"user_input\": user_input, \"user_id\": user_id})"
)

# After Global Workspace focus
content = content.replace(
    "                logging.info(f\"Workspace focused on event '{focused_event.get('type')}' with Salience: {score:.2f} ({explanation})\")",
    "                logging.info(f\"Workspace focused on event '{focused_event.get('type')}' with Salience: {score:.2f} ({explanation})\")\n                if self.tracer:\n                    self.tracer.add_step(\"global_workspace_focus\", {\"event_type\": focused_event.get('type'), \"salience\": score, \"explanation\": explanation})"
)

# After Salience fallback focus
content = content.replace(
    "            logging.info(f\"Salience score: {score:.2f} ({explanation})\")",
    "            logging.info(f\"Salience score: {score:.2f} ({explanation})\")\n            if self.tracer:\n                self.tracer.add_step(\"salience_scoring\", {\"salience\": score, \"explanation\": explanation})"
)

# Before Memory Retrieval, after emotion update
content = content.replace(
    "        # 2. Memory Retrieval\n        relevant_memories = self.memory.retrieve_relevant(user_input, user_id=user_id)",
    "        # 2. Memory Retrieval\n        relevant_memories = self.memory.retrieve_relevant(user_input, user_id=user_id)\n        if self.tracer:\n            self.tracer.add_step(\"memory_retrieval\", {\"retrieved_count\": len(relevant_memories)})"
)

# Planning Step
content = content.replace(
    "        # 3. Planning (Prefrontal Cortex)\n        plan_str = \"\"",
    "        # 3. Planning (Prefrontal Cortex)\n        plan_str = \"\"\n        if self.tracer:\n            self.tracer.add_step(\"planning_start\", {})"
)

# LLM Reasoning start
content = content.replace(
    "        # 4. LLM Reasoning\n        emotion_summary = self.emotions.get_summary(user_id=user_id)",
    "        # 4. LLM Reasoning\n        if self.tracer:\n            self.tracer.add_step(\"llm_reasoning_start\", {\"plan_str\": plan_str})\n        emotion_summary = self.emotions.get_summary(user_id=user_id)"
)

# Before basal ganglia return (ignore)
content = content.replace(
    "            if selected_action and selected_action[\"action\"] == \"ignore\":\n                return \"Message ignored by Basal Ganglia.\"",
    "            if selected_action and selected_action[\"action\"] == \"ignore\":\n                if self.tracer:\n                    self.tracer.add_step(\"action_selection\", {\"selected_action\": \"ignore\"})\n                return \"Message ignored by Basal Ganglia.\"\n            elif selected_action and self.tracer:\n                self.tracer.add_step(\"action_selection\", {\"selected_action\": selected_action[\"action\"]})"
)

# After response is generated
content = content.replace(
    "        response = await self.llm.chat_completion(messages)",
    "        response = await self.llm.chat_completion(messages)\n        if self.tracer:\n            self.tracer.add_step(\"response_generated\", {\"response\": response})"
)


with open("magda_agent/consciousness/core.py", "w") as f:
    f.write(content)
