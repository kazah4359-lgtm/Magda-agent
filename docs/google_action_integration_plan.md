# Google Action-Oriented Integration Plan

This plan outlines how Jules can leverage Google-style Actions to perform tasks in the real world, similar to how a Rool Machine uses its tools.

## 1. Action Schema
Every task should be treated as an "Action" with:
*   **Trigger:** Input/Event (e.g., new task in `agent_tasks.json`).
*   **Action Name:** Unique identifier for the tool call.
*   **Parameters:** JSON-structured arguments.
*   **Outcome:** Structured response (Success/Failure + Data).

## 2. Rool-to-Google Mapping
*   **Rool Shell Call** -> Google Custom Tool / Vertex AI Action.
*   **Rool File System** -> Cloud Storage / Drive Integration.
*   **Rool Python Exec** -> Cloud Functions / Sandbox Execution.

## 3. Implementation Steps
1.  Map Jules' existing `codex_bridge.py` to a structured Tool Spec.
2.  Define "Observation Actions" (ls, cat, grep).
3.  Define "Transformation Actions" (python, sed, rewrite).
4.  Define "Validation Actions" (pytest, lint, typecheck).
