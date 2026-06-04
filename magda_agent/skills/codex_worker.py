"""
Codex worker skill module.

Exposes a safe skill that renders a Codex-ready task prompt without
launching external Codex processes or creating PRs.
"""

from typing import Optional
from magda_agent.codex_bridge import load_manifest, next_task, task_by_id, render_prompt


def codex_worker(task_id: Optional[str] = None) -> str:
    """
    Renders a Codex-ready task prompt for a given task ID or the next todo task.
    This is a low side-effect prompt-only capability that does not execute shell commands
    or network calls.

    Args:
        task_id: Optional string ID of the task. If not provided, the next todo task is used.

    Returns:
        The rendered prompt text, or an error message if the task is not found or manifest cannot be loaded.
    """
    try:
        manifest = load_manifest()
    except Exception as exc:
        return f"Error loading task manifest: {exc}"

    if task_id:
        task = task_by_id(manifest, task_id)
        if not task:
            return f"Task with ID '{task_id}' not found."
    else:
        task = next_task(manifest)
        if not task:
            return "No 'todo' tasks found in the manifest."

    return render_prompt(task)
