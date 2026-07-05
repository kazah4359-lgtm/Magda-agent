from typing import Dict, Any, Optional
from magda_agent.emotions.engine import PADState

class StyleAdapter:
    """
    Adapts the system response style based on the agent's emotional state
    and the user's preferences.
    """

    def get_style_prompt(self, pad_state: PADState, user_model: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a system prompt modifier based on PAD state and user model.

        Args:
            pad_state (PADState): The current emotional state (Pleasure, Arousal, Dominance).
            user_model (Optional[Dict[str, Any]]): The user model containing preferences and style.

        Returns:
            str: A string to be appended to the system prompt to guide the LLM's style.
        """
        style_instructions = ["Response Style Instructions:"]

        # Emotional State Modifications
        p, a, d = pad_state.pleasure, pad_state.arousal, pad_state.dominance

        if a > 0.4:
            style_instructions.append("- State is high arousal: Be concise, direct, and slightly urgent. Keep responses shorter than usual.")
        elif a < -0.4:
            style_instructions.append("- State is low arousal: Be relaxed, elaborate, and thoughtful. Take your time explaining.")

        if p > 0.4:
            style_instructions.append("- State is high pleasure: Be warm, friendly, and optimistic. Use engaging language.")
        elif p < -0.4:
            style_instructions.append("- State is low pleasure: Be serious, formal, and objective. Avoid overly cheerful language.")

        if d > 0.4:
            style_instructions.append("- State is high dominance: Be confident, assertive, and decisive in your statements.")
        elif d < -0.4:
            style_instructions.append("- State is low dominance: Be cautious, polite, and cooperative. Soften your statements.")

        # User Model Modifications
        if user_model:
            preferences = user_model.get("preferences", {})
            comm_style = user_model.get("communication_style", "default")
            expertise = user_model.get("expertise_level", "unknown")

            if comm_style and comm_style != "default":
                style_instructions.append(f"- User prefers communication style: {comm_style}.")

            if expertise == "technical" or preferences.get("technical_details", False):
                style_instructions.append("- User is highly technical: Prioritize code examples, precise terminology, and avoid over-explaining basics.")
            elif expertise == "beginner":
                style_instructions.append("- User is a beginner: Explain concepts simply, step-by-step, avoiding heavy jargon.")

            if "short_answers" in preferences and preferences["short_answers"]:
                style_instructions.append("- User strictly prefers very short, bulleted answers.")

        # Behavior Weight Modifications
        if user_model and "behavior_weights" in user_model:
            bw = user_model["behavior_weights"]
            if bw.get("verbosity", 1.0) > 1.3:
                style_instructions.append("- High verbosity mode: Provide detailed, thorough explanations.")
            elif bw.get("verbosity", 1.0) < 0.7:
                style_instructions.append("- Low verbosity mode: Be as brief and concise as possible.")

            if bw.get("directness", 1.0) > 1.3:
                style_instructions.append("- High directness mode: Get straight to the point, avoid preamble.")
            elif bw.get("directness", 1.0) < 0.7:
                style_instructions.append("- Low directness mode: Use polite softeners and structured introductions.")

        if len(style_instructions) > 1:
            return "\n".join(style_instructions)
        return ""
