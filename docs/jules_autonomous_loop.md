# Jules Autonomous Improvement Loop

Этот документ описывает, как Jules должен работать без остановки: брать задачи, улучшать проект, пополнять очередь, исправлять свои ошибки и не уходить в режим вопросов, когда работы достаточно.

## Цель

Jules должен поддерживать постоянный цикл улучшения Magda Agent:

```text
read architecture
  -> inspect task manifest
  -> replenish task queue when low
  -> select one safe task
  -> implement small complete change
  -> update manifest
  -> open PR
  -> wait for CI / merge
  -> continue with next task
```

## Главный запрет

Jules не должен спрашивать пользователя "что делать дальше", если он может безопасно:

- взять следующий `todo` из `agent_tasks.json`;
- создать low/medium-risk задачи из архитектурных документов;
- создать fix-задачу из падающего CI;
- создать stabilization-задачу из явного code smell;
- уменьшить риск уже существующей задачи, разбив ее на меньшие шаги.

Вопрос пользователю нужен только для high/critical-risk изменений, доступа к секретам, изменения прав auto-merge или неоднозначной продуктовой цели.

## Источники задач

Приоритет:

1. `agent_tasks.json`
2. `AGENTS.md`
3. CI failures
4. `docs/cognitive_architecture.md`
5. `docs/jules_autonomous_loop.md`
6. `docs/codex_worker_plan.md`
7. `backlog.md`
8. TODO/FIXME comments
9. recurring errors from previous PRs

Если явных задач нет, Jules создает пачку задач сам.

## Replenishment Policy

Jules обязан поддерживать минимум 3 задачи со статусом `todo`.

Если `todo < minimum_todo_tasks`:

1. Не начинать случайный рефакторинг.
2. Найти текущие архитектурные разрывы.
3. Создать `batch_size` новых задач.
4. Каждая задача должна иметь:
   - stable `id`;
   - `risk`;
   - `area`;
   - `allowed_paths`;
   - `acceptance`;
   - короткое описание ожидаемого улучшения.
5. По умолчанию новые задачи должны быть `low` или `medium`.
6. High/critical задачи можно записать, но не выбирать для автономной реализации без review.

## Task Selection

Jules выбирает первую `todo` задачу, но перед реализацией проверяет:

- задача не high/critical без разрешения;
- `allowed_paths` достаточно узкие;
- задачу можно выполнить одним PR;
- acceptance criteria проверяемы;
- изменение уменьшает риск или открывает путь к следующему улучшению.

Если первая задача слишком крупная, Jules не спрашивает пользователя. Он должен:

1. Пометить ее как `blocked` или оставить `todo` с комментарием в description.
2. Создать 2-5 меньших prerequisite tasks перед ней.
3. Взять первую маленькую задачу.

## Risk Gates

Autonomous allowed:

- docs;
- schema/manifest validation;
- pure internal modules;
- bounded refactors with tests;
- bug fixes that do not add new external effects.

Human review required:

- `.github/workflows/**`;
- `requirements.txt`;
- deployment files;
- sandbox policy;
- secrets/env handling;
- external messaging providers;
- auto-merge permissions;
- destructive file operations.

## Stabilization First

Перед крупными когнитивными слоями Jules должен стабилизировать базу:

1. project imports must work on developer machines;
2. plan execution must have max steps and timeouts;
3. LLM-generated plans must be validated;
4. user state must not leak between users;
5. memory layers must not duplicate each other silently;
6. background tasks must stop cleanly;
7. API must not expose internal state without protection.

Only after this should Jules focus on Salience Network, Global Workspace and deeper memory architecture.

## Failure Loop

Если CI падает:

1. Не создавать новую feature-задачу.
2. Создать или выбрать fix-задачу для текущего failure.
3. Ограничить diff только failure area.
4. После исправления добавить lesson в `agent_tasks.json` или docs when relevant.

Если один и тот же failure повторился 2 раза:

- создать stabilization task;
- уменьшить размер следующего PR;
- добавить guardrail, а не только локальный fix.

## PR Shape

Каждый PR должен быть маленьким и завершенным:

- one task id;
- bounded file scope;
- no opportunistic refactors;
- no unrelated backlog churn;
- update `agent_tasks.json`;
- add tests or explain why the task is docs/governance only;
- include the exact validation commands in PR body.

## Done Criteria

Задача считается done только если:

- код/документация изменены в рамках `allowed_paths`;
- acceptance criteria выполнены или уточнены;
- task status обновлен;
- новые обнаруженные проблемы добавлены как tasks;
- PR не увеличивает risk level системы без явной причины.

## Continuous Improvement Metrics

Jules должен постепенно улучшать:

- number of green CI runs;
- smaller PR size;
- fewer repeated failures;
- clearer module contracts;
- less global mutable state;
- fewer duplicated storage systems;
- safer tool execution;
- more explicit task acceptance criteria.

