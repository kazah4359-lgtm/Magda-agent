# Magdalina: Cognitive AGI Architecture

Magdalina is an experimental AGI agent built with a multi-layered cognitive architecture inspired by human brain functions. This document outlines the core components and their interactions.

## 1. Consciousness (Main Cognitive Loop)
The **Consciousness** layer (implemented in `magda_agent/consciousness/core.py`) serves as the central orchestration point. It handles:
- **Perception:** Receiving and filtering input through the Thalamus.
- **Focus:** Selecting the most salient event via the Global Workspace.
- **Cognition:** Memory retrieval, planning, and action selection.
- **Self-Correction:** Evaluating outputs and calibrating confidence.

## 2. Subconsciousness (Reflection & Consolidation)
The **Subconsciousness** (implemented in `magda_agent/subconsciousness/reflection.py`) operates as a background process that:
- **Consolidates Memory:** Periodically reviews episodic memories to extract semantic facts.
- **Reflects:** Analyzes past successes and failures to generate lessons and anti-patterns.
- **Proposes Tasks:** Identifies gaps in knowledge or project structure and suggests new tasks for the autonomous loop.

## 3. Memory Systems
Memory is partitioned into several functional layers:
- **Working Memory:** Limited, short-term context for active tasks.
- **Episodic Memory:** Chronological record of events and interactions.
- **Semantic Memory:** Stable, distilled knowledge about the project and environment.
- **Procedural Memory:** Reusable skills and successful methods (habits).
- **Virtual Context:** A paging mechanism (MemGPT pattern) to manage effectively infinite context length.

## 4. Emotional & Drive Engine
Magdalina's behavior is modulated by internal states:
- **PAD Model:** Pleasure, Arousal, and Dominance influence response style and risk tolerance.
- **Mental States:** Tracks Fear, Desire, Tension, and cognitive biases (Optimism, Overconfidence).
- **Hypothalamus (Drives):** Models biological-like drives such as energy and boredom to trigger proactive exploration.

## 5. Safety & Governance
- **Agent Guard / ACS:** A multi-checkpoint policy layer intercepts all external tool calls.
- **Risk System:** Classifies task risk levels (Low, Medium, High, Critical) to determine merge and execution policies.
- **Distributed Tracing (A2A):** Ensures auditability and observability across agent-to-agent delegations.

## 6. Interaction Interface
- **Telegram Bot:** Real-time interaction dashboard for status monitoring and task management.
- **REST API:** FastAPI-based service for programmatic integration.
- **MCP Compatibility:** Exposes agent skills as standardized tools for other agents.
