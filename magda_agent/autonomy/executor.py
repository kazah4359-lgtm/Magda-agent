"""Background worker that drives long-running autonomous tasks to completion.

Unlike `Consciousness.process_input` (a single short turn, capped at a handful of
steps), the executor keeps iterating on one goal: plan -> execute steps ->
evaluate against the goal -> refine objective -> repeat, until the goal is met,
the task is cancelled/paused, or a safety iteration budget is exhausted.
"""
import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from magda_agent.autonomy.task_store import TaskStore, TaskStatus

logger = logging.getLogger(__name__)


def _default_planner_factory(llm: Any, skills: Any) -> Callable[[], Any]:
    def factory() -> Any:
        # Imported lazily so the executor module stays importable without the
        # full cognitive/memory dependency stack (useful for tests).
        from magda_agent.planning.planner import Planner

        return Planner(llm=llm, skills=skills)

    return factory


class AutonomousExecutor:
    def __init__(
        self,
        store: TaskStore,
        llm: Any,
        skills: Any,
        *,
        planner_factory: Optional[Callable[[], Any]] = None,
        poll_interval: float = 2.0,
        step_timeout: float = 60.0,
        max_steps_per_iteration: int = 25,
    ):
        self.store = store
        self.llm = llm
        self.skills = skills
        self.planner_factory = planner_factory or _default_planner_factory(llm, skills)
        self.poll_interval = poll_interval
        self.step_timeout = step_timeout
        self.max_steps_per_iteration = max_steps_per_iteration
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("AutonomousExecutor started.")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("AutonomousExecutor stopped.")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                task = await self.store.claim_next_queued()
                if task is None:
                    await asyncio.sleep(self.poll_interval)
                    continue
                await self._execute_task(task.id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep the worker alive across task failures
                logger.exception("Executor loop error: %s", exc)
                await asyncio.sleep(self.poll_interval)

    async def _interrupted(self, task_id: str) -> Optional[str]:
        """Returns a terminal status string if the task was cancelled/paused."""
        current = await self.store.get(task_id)
        if current is None:
            return TaskStatus.CANCELLED
        if current.status == TaskStatus.CANCELLING:
            await self.store.update(task_id, status=TaskStatus.CANCELLED)
            await self.store.append_progress(task_id, "Task cancelled.", event="cancelled")
            return TaskStatus.CANCELLED
        if current.status == TaskStatus.PAUSING:
            await self.store.update(task_id, status=TaskStatus.PAUSED)
            await self.store.append_progress(task_id, "Task paused.", event="paused")
            return TaskStatus.PAUSED
        return None

    async def _execute_task(self, task_id: str) -> None:
        task = await self.store.get(task_id)
        if task is None:
            return

        await self.store.append_progress(task_id, f"Started working on goal: {task.goal}", event="started")
        planner = self.planner_factory()
        objective = task.goal
        last_summary = ""

        for iteration in range(1, task.max_iterations + 1):
            if await self._interrupted(task_id):
                return
            await self.store.update(task_id, iterations=iteration)
            await self.store.append_progress(task_id, f"Iteration {iteration}: planning '{objective}'", event="plan")

            try:
                steps = await planner.generate_plan(objective, user_id=task.user_id)
            except Exception as exc:
                logger.exception("Planning failed: %s", exc)
                steps = []

            if not steps:
                await self.store.append_progress(task_id, "Planner returned no steps.", event="warn")
            else:
                last_summary = await self._execute_plan(task_id, planner)
                if await self._interrupted(task_id):
                    return

            decision = await self._evaluate(task.goal, objective, last_summary)
            await self.store.append_progress(
                task_id, f"Self-check: {decision.get('summary', '')}", event="evaluate"
            )

            if decision.get("done"):
                await self.store.update(task_id, status=TaskStatus.COMPLETED, result=decision.get("summary") or last_summary)
                await self.store.append_progress(task_id, "Goal achieved.", event="completed")
                return

            next_objective = decision.get("next_objective")
            if next_objective:
                objective = next_objective

        # Iteration budget exhausted without an explicit "done".
        await self.store.update(
            task_id,
            status=TaskStatus.COMPLETED,
            result=last_summary or "Reached max iterations without explicit completion.",
        )
        await self.store.append_progress(
            task_id, f"Stopped after {task.max_iterations} iterations (budget reached).", event="completed"
        )

    async def _execute_plan(self, task_id: str, planner: Any) -> str:
        """Execute the planner's current plan sequentially with no artificial cap."""
        steps_executed = 0
        while planner.get_current_plan() and steps_executed < self.max_steps_per_iteration:
            if await self._interrupted(task_id):
                break
            steps_executed += 1
            step = planner.get_current_plan()[0]
            skill_name = step.get("skill")
            kwargs = step.get("skill_kwargs") or {}
            description = step.get("description", "")

            if skill_name:
                try:
                    coro = asyncio.to_thread(self.skills.execute_skill, skill_name, **kwargs)
                    result = await asyncio.wait_for(coro, timeout=self.step_timeout)
                except asyncio.TimeoutError:
                    result = f"Error: skill '{skill_name}' timed out after {self.step_timeout}s."
                except Exception as exc:
                    result = f"Error executing skill '{skill_name}': {exc}"
            else:
                result = "No skill required for this step."

            planner.mark_step_completed(0, str(result))
            await self.store.append_progress(
                task_id, f"Step: {description} (skill={skill_name}) -> {str(result)[:400]}", event="step"
            )

        summary_lines = []
        for i, step in enumerate(planner.completed_steps, start=1):
            summary_lines.append(
                f"{i}. {step.get('description')} (skill={step.get('skill')}) -> {step.get('result')}"
            )
        planner.clear_pending_plan()
        return "\n".join(summary_lines)

    async def _evaluate(self, goal: str, objective: str, results: str) -> Dict[str, Any]:
        """Ask the LLM whether the goal is met; if not, what to do next."""
        system = (
            "You are the self-evaluation module of an autonomous agent. "
            "Given the overall GOAL and the RESULTS of the latest work, decide whether the "
            "goal is fully achieved. Respond ONLY with a JSON object: "
            '{"done": boolean, "summary": string, "next_objective": string}. '
            "'summary' briefly states current progress. If done is false, 'next_objective' "
            "is the concrete next sub-goal to pursue. If done is true, 'next_objective' is empty."
        )
        user = f"GOAL:\n{goal}\n\nCURRENT OBJECTIVE:\n{objective}\n\nRESULTS SO FAR:\n{results or '(no results yet)'}"
        try:
            raw = await self.llm.chat_completion(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
        except Exception as exc:
            logger.exception("Evaluation LLM call failed: %s", exc)
            return {"done": False, "summary": f"Evaluation error: {exc}", "next_objective": objective}

        return self._parse_decision(raw, objective)

    @staticmethod
    def _parse_decision(raw: str, objective: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            data = json.loads(text)
            return {
                "done": bool(data.get("done", False)),
                "summary": str(data.get("summary", "")),
                "next_objective": str(data.get("next_objective", "") or ""),
            }
        except (json.JSONDecodeError, TypeError, AttributeError):
            # If the model didn't return JSON, be conservative and keep going.
            return {"done": False, "summary": text[:300], "next_objective": objective}
