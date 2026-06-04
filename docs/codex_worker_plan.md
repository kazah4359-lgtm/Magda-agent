# Codex Worker Integration Plan

Этот план описывает, как добавить Magda режим работы через Codex: не как замену Jules, а как еще один coding-worker backend для задач разработки, ревью и улучшения проекта.

## Идея

Magda должна уметь формировать и сопровождать coding tasks для Codex:

```text
agent_tasks.json
  -> Magda/Codex bridge selects task
  -> renders focused prompt
  -> Codex implements scoped change
  -> tests/validation
  -> PR
  -> result compressed into lesson
  -> task queue updated
```

Это повторяет полезные идеи Hermes: создать CLI, которым агент может пользоваться, сохранять workflows как skills, запускать long-horizon tasks, ревьюить PR и строить повторяемый improvement loop.

## Official Codex Patterns To Reuse

From official OpenAI Codex use-case docs:

- Create a CLI Codex can use.
- Save repeatable workflows as skills.
- Follow a durable goal for long-running work.
- Review GitHub pull requests.
- Run code migrations in controlled checkpoints.
- Add evals/tests to AI applications.
- Use Codex for bug triage and verified operations.

Magda should adapt these patterns through its own cognitive architecture.

## Surfaces

### 1. Repo Instructions: `AGENTS.md`

`AGENTS.md` is the durable instruction layer for Codex/Jules inside the repository.

It should explain:
- task source priority;
- validation commands;
- risk boundaries;
- files not to touch without explicit permission;
- how to use the Codex bridge.

### 2. Codex Bridge CLI

Magda exposes a lightweight stdlib-only CLI:

```bash
python -m magda_agent.codex_bridge validate
python -m magda_agent.codex_bridge status
python -m magda_agent.codex_bridge next-task
python -m magda_agent.codex_bridge render-prompt
```

This gives Codex a composable command surface without importing the full FastAPI/ChromaDB/Telegram stack.

### 3. Codex Worker Skill

Later, Magda should expose a controlled skill:

```text
codex_worker(task_id, mode="prompt" | "local" | "cloud")
```

Initial implementation should only render task specs. Running Codex or creating PRs should remain gated until policy and credentials are explicit.

### 4. PR Review Worker

Codex can act as a reviewer:

```text
PR diff
  -> Codex review prompt
  -> findings
  -> risk score
  -> missing tests
  -> suggested follow-up tasks
```

This should feed the future `CriticAgent`.

### 5. Workflow Skills

Repeatable Codex workflows should become procedural memory:

- "stabilize failing test";
- "add a brain module";
- "add command registry command";
- "write doctor diagnostic";
- "split a God Method step into processor".

These should be stored as Magda procedure templates, not enabled as executable code automatically.

## Safety Model

Codex-worker tasks must obey the same policy gates as Jules:

- one task id per PR;
- allowed paths only;
- no workflow/dependency/sandbox changes unless task allows them;
- high/critical tasks require human review;
- generated code is never auto-enabled as a skill;
- PRs update `agent_tasks.json`.

## First Implementation Slice

Already included:

- `AGENTS.md`
- `magda_agent/codex_bridge.py`

Next slices:

1. Add focused tests for `codex_bridge`.
2. Add `codex_worker` skill in prompt-only mode.
3. Add `/codex next` command through future command registry.
4. Add PR review prompt renderer.
5. Add task claim/lock field to prevent multiple workers taking the same task.
6. Add trajectory compression from Codex PRs into lessons.

## Command Examples

Get next task:

```bash
python -m magda_agent.codex_bridge next-task
```

Render prompt for the next task:

```bash
python -m magda_agent.codex_bridge render-prompt
```

Render prompt for a specific task:

```bash
python -m magda_agent.codex_bridge render-prompt --task-id planner-schema-validation
```

Check if the queue is running low:

```bash
python -m magda_agent.codex_bridge status
```

