# Magdalina AGI Architecture

Magdalina is an autonomous, self-improving AGI agent designed with a cognitive architecture inspired by the human mind. It operates through three main cognitive layers and a complex memory system.

## 1. Cognitive Layers

### 1.1 Consciousness (Main Loop)
- **Role**: The current focus and decision-making center.
- **Function**: Processes incoming stimuli (user messages, environment changes), retrieves relevant memories, selects appropriate skills, and executes actions.
- **Working Memory**: A limited-capacity buffer for immediate context.

### 1.2 Subconsciousness (Background Processing)
- **Role**: Asynchronous reflection and optimization.
- **Function**: Operates in the background to analyze past actions, consolidate memories (moving from short-term to long-term), "dream" (simulate scenarios for learning), and adjust emotional baselines.
- **Self-Improvement**: Identifies mistakes and creates "lessons learned" to be stored in the Unconscious or Long-Term Memory.

### 1.3 Unconscious (Foundation)
- **Role**: Hardcoded rules and base instincts.
- **Function**: Contains the system prompt, safety guidelines, and fundamental goals that the agent cannot easily modify. It provides the "biological" constraints of the agent's behavior.

## 2. Memory System

Magdalina utilizes a multi-tiered memory system that mimics human recall:

- **Short-Term Memory**: Highly detailed, recent interactions.
- **Long-Term Memory**: Summarized or highly important past experiences.
- **Importance Scoring**: Each memory is assigned a weight based on its impact. High-importance memories are preserved longer.
- **Emotional Coloring**: Memories are tagged with the emotional state of the agent at the time of the event, influencing how they are recalled.
- **Decay & Consolidation**: Over time, less important memories fade (summarized or deleted) while important ones are reinforced.

## 3. Emotional Engine

Emotions in Magdalina are calculated mathematically:
- **Pleasure/Pain**: Response to success or failure in achieving goals.
- **Arousal**: Intensity of the current situation.
- **Dominance**: Perception of control over the environment.
- **Impact**: Emotions influence the "energy" available for tasks and the way memories are stored and retrieved.

## 4. Skills & Expertise

Instead of being a generalist, Magdalina uses a dynamic skill system:
- **Expert Modules**: Specialized sets of tools and prompts (e.g., Programmer, Analyst, Creative).
- **Skill Acquisition**: The Subconsciousness can identify the need for a new skill and "train" it by creating new expert modules.
- **Contextual Loading**: The Consciousness pulls in relevant skills based on the task at hand.

## 5. Technical Stack
- **Engine**: Python
- **Interface**: Telegram Bot (Dashboard)
- **Environment**: Docker & Virtual Machine (Sandbox)
- **Models**: Cloud-based LLMs (e.g., GPT-4o, Claude 3.5 Sonnet) via API.
