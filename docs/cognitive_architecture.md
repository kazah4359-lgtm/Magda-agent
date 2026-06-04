# Magda Cognitive Architecture

Этот документ задает целевую архитектуру Magda Agent как автономной когнитивной системы с 24/7 self-improvement loop через Jules.

Цель не в том, чтобы заявлять о "настоящем сознании", а в том, чтобы построить инженерную модель сознательного поведения: внимание, память, цели, эмоции, подсознательная консолидация, самооценка и безопасное самоулучшение.

Операционные правила для непрерывной работы Jules описаны отдельно: [jules_autonomous_loop.md](jules_autonomous_loop.md). Этот файл определяет, как агент выбирает задачи, пополняет очередь, реагирует на CI failures и не останавливается из-за нехватки явных backlog-пунктов.

Работа через Codex описана в [codex_worker_plan.md](codex_worker_plan.md). Codex должен использовать `AGENTS.md` и `python -m magda_agent.codex_bridge` как легковесную поверхность для выбора задач, проверки manifest и рендера focused prompts.

## Базовая петля

```text
input event
  -> sensory filter
  -> salience scoring
  -> global workspace
  -> memory retrieval
  -> internal state update
  -> planning
  -> risk check
  -> action/tool execution
  -> observation
  -> evaluation
  -> memory write
  -> subconscious consolidation
```

Для Jules self-improvement loop:

```text
agent_tasks.json / backlog.md
  -> Jules implements one task
  -> PR
  -> CI tests and guardrails
  -> auto-merge only when risk is acceptable
  -> next task trigger
  -> reflection and new tasks
```

## Архитектурные принципы

1. Биологические названия допустимы только там, где у модуля есть проверяемая инженерная функция.
2. Каждый модуль обязан иметь явный контракт: inputs, outputs, internal state, side effects, tests.
3. Память должна быть разделена по назначению, а не храниться в одной общей "куче".
4. Любой tool с внешним эффектом проходит через risk/policy layer.
5. Auto-merge не должен быть единственным механизмом истины. Нужны guardrails, smoke tests и post-merge checks.
6. Self-improvement оценивается по тренду качества, а не по одному зеленому pytest.

## Целевые подсистемы

### Thalamus: sensory gateway

Назначение: принять входной сигнал, нормализовать его, отфильтровать шум и определить канал.

Inputs:
- user text;
- voice transcript;
- system event;
- Jules/CI event;
- scheduled reflection event.

Outputs:
- normalized event;
- channel;
- initial confidence;
- reject reason when input is noise.

Tests:
- empty/noisy input rejected;
- normal input accepted;
- system events do not pass as user text.

### Salience Network

Назначение: определить, что заслуживает внимания прямо сейчас.

Scoring factors:
- user urgency;
- active task relevance;
- failing CI/test signal;
- security risk;
- emotional/interoceptive pressure;
- novelty and uncertainty.

Output:
- salience score from 0.0 to 1.0;
- explanation string for debugging.

This is the first missing core module after the current implementation.

### Global Workspace

Назначение: единая рабочая сцена сознания. Несколько подсистем предлагают "кандидатов", workspace выбирает фокус и транслирует его остальным.

Candidate examples:
- latest user request;
- failed test report;
- active plan step;
- memory conflict;
- security alert;
- boredom/curiosity drive.

Output:
- focused event;
- broadcast context;
- suppressed candidates for later processing.

Engineering value:
- убирает монолитный `Consciousness.process_input`;
- позволяет честно моделировать внимание;
- дает место для конкуренции целей.

### Working Memory

Назначение: короткий, ограниченный контекст текущей задачи.

Stores:
- current goal;
- active constraints;
- recent observations;
- pending hypotheses;
- current plan state.

Rule:
- small and explicit;
- no permanent storage;
- reset or summarize after task completion.

### Hippocampus: episodic memory

Назначение: хранить события и опыт.

Examples:
- "PR #12 failed because chromadb was missing";
- "Jules confused BACKLOG.md and backlog.md";
- "User prefers deep technical plans in Russian".

Contract:
- append-only event records;
- user_id and source_id aware;
- supports recall by semantic similarity and filters.

### Semantic Memory

Назначение: хранить stable knowledge about the project.

Examples:
- module contracts;
- architecture decisions;
- known invariants;
- accepted terminology.

This should be separate from episodic memory because facts and events age differently.

### Procedural Memory

Назначение: хранить reusable successful procedures.

Examples:
- "how to add a new brain module";
- "how to add tests with mocked LLM";
- "how to recover from failed CI";
- "how to update agent_tasks.json".

This is inspired by skill libraries in lifelong agents. It should not execute blindly; each skill still goes through policy.

### Emotional Engine

Current model: PAD: pleasure, arousal, dominance.

Target role:
- not decorative mood;
- priority modulation;
- response style modulation;
- risk sensitivity modulation.

Examples:
- high arousal + low confidence -> more verification before action;
- low arousal + high boredom -> curiosity task proposal;
- high dominance -> more direct execution, but still constrained by policy.

### Hypothalamus: drives

Target drives:
- energy;
- boredom;
- uncertainty;
- task pressure;
- safety pressure;
- social attachment pressure.

Use:
- influence salience;
- influence planning depth;
- schedule rest/consolidation.

### Amygdala / Risk System

Назначение: оценивать угрозы и блокировать опасные действия.

High-risk signals:
- `.github/workflows/**` changes;
- `requirements.txt` changes;
- code execution sandbox changes;
- tool registry changes;
- network or messaging providers;
- secrets/env handling;
- auto-merge logic.

Output:
- risk level: low, medium, high, critical;
- required approval: none, CI only, critic review, human review.

### Basal Ganglia: action selection

Назначение: выбрать действие из конкурирующих возможностей.

Candidate actions:
- answer user;
- inspect files;
- edit code;
- run tests;
- request human approval;
- create follow-up task;
- stop execution.

Selection must consider:
- priority;
- risk;
- available budget;
- confidence;
- policy.

### Prefrontal Cortex: planning

Target planner output:

```json
{
  "goal": "Implement salience scoring",
  "constraints": ["no new dependencies", "tests must mock LLM"],
  "risk": "medium",
  "steps": [
    {"kind": "inspect", "target": "magda_agent/consciousness"},
    {"kind": "edit", "target": "magda_agent/attention/salience.py"},
    {"kind": "test", "target": "tests/test_salience.py"}
  ],
  "acceptance": ["pytest tests/test_salience.py", "pytest"]
}
```

The current planner returns loose JSON steps. It should evolve toward typed plans with risk and acceptance criteria.

### Subconsciousness

Назначение: фоновая консолидация, а не просто периодический LLM call.

Tasks:
- summarize recent episodes;
- detect recurring failures;
- generate proposed tasks;
- update procedural memory;
- decay irrelevant memories;
- identify contradictions between memory and current code.

### Metacognition

Назначение: оценка собственного мышления и действий.

Metrics:
- did the plan match the user goal;
- did tests actually prove the behavior;
- did diff stay inside allowed scope;
- did the response hide uncertainty;
- did a previous error repeat.

## Jules governance model

### Task replenishment

The loop should not stop or ask the user only because the explicit task queue is empty.

Rules:
- keep at least 3 `todo` tasks in `agent_tasks.json`;
- if fewer remain, generate a replenishment batch from this document, failing tests, recurring errors and known architecture gaps;
- generated tasks must be low or medium risk by default;
- high or critical tasks may be proposed, but should not be selected for autonomous implementation without human review;
- each generated task must include `allowed_paths` and `acceptance` criteria.

This prevents the 24/7 Jules loop from falling into clarification mode when it simply ran out of work.

### Risk levels

Low:
- docs;
- isolated pure functions;
- tests;
- small bug fixes with no external effects.

Medium:
- memory retrieval;
- planner behavior;
- API schemas;
- internal state changes.

High:
- workflows;
- dependencies;
- sandbox;
- tool registry;
- network access;
- messaging providers.

Critical:
- secrets;
- auto-merge permissions;
- destructive filesystem operations;
- production deployment credentials.

### Merge policy

Recommended:
- low risk: auto-merge after tests and guardrails;
- medium risk: auto-merge only after expanded tests and critic pass;
- high risk: no auto-merge without human review;
- critical: manual only.

## Near-term implementation sequence

1. Add machine-readable `agent_tasks.json`.
2. Add validator for task schema and risk levels.
3. Teach Jules prompt to read `agent_tasks.json` before markdown backlog.
4. Add docs for cognitive architecture and module contracts.
5. Implement `RiskSystem` for PR/file change classification.
6. Implement `SalienceNetwork`.
7. Refactor `Consciousness.process_input` around `GlobalWorkspace`.
8. Split memory into working, episodic, semantic and procedural layers.
9. Add subconscious consolidation reports.
10. Add longitudinal quality metrics for Jules loop.

## Anti-patterns

- Adding a brain-like module name without measurable behavior.
- Letting LLM-generated plans execute tools without policy.
- Treating green tests as proof of architectural quality.
- Mixing user memory, agent memory and project memory.
- Letting Jules modify its own merge permissions automatically.
- Storing reflections without checking whether they are true.
