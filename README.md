# Все по полочкам / Everything on Shelves

<div align="center">
  <p>
    <a href="#english">🇬🇧 English</a> •
    <a href="#russian">🇷🇺 Русский</a>
  </p>
</div>

---

<h2 id="english">🇬🇧 English</h2>

**Все по полочкам** is an experimental cognitive agent built around a Telegram interface, a FastAPI consciousness service, memory, emotion, planning, skills, and a Jules-driven self-improvement loop. Originally based on the Magda Agent architecture.

### 🔄 Self-Improvement Loop

- **Target architecture:** [docs/cognitive_architecture.md](docs/cognitive_architecture.md)
- **Codex worker plan:** [docs/codex_worker_plan.md](docs/codex_worker_plan.md)
- **Machine-readable task queue:** [agent_tasks.json](agent_tasks.json)
- **Task manifest validator:** `python scripts/validate_agent_tasks.py agent_tasks.json`

Jules should read `agent_tasks.json` first, implement the first task with status `todo`, keep the task pool replenished according to `replenishment_policy`, and update the task status after completing a PR.

### 🌉 Codex Bridge

The project exposes a lightweight stdlib-only bridge that Codex/Jules can use without importing the full Telegram/FastAPI/memory stack:

```bash
python -m magda_agent.codex_bridge validate
python -m magda_agent.codex_bridge status
python -m magda_agent.codex_bridge next-task
python -m magda_agent.codex_bridge render-prompt
```

### 🧪 Local Checks

```bash
python scripts/validate_agent_tasks.py agent_tasks.json
python -m magda_agent.codex_bridge status
pytest
```

---

<h2 id="russian">🇷🇺 Русский</h2>

**Все по полочкам** — это экспериментальный когнитивный агент, построенный на базе Telegram-интерфейса, сервиса сознания FastAPI, памяти, эмоций, планирования, навыков и цикла самосовершенствования, управляемого агентом Jules. Первоначально основан на архитектуре Magda Agent.

### 🔄 Цикл самосовершенствования

- **Целевая архитектура:** [docs/cognitive_architecture.md](docs/cognitive_architecture.md)
- **План работы Codex:** [docs/codex_worker_plan.md](docs/codex_worker_plan.md)
- **Машиночитаемая очередь задач:** [agent_tasks.json](agent_tasks.json)
- **Валидатор манифеста задач:** `python scripts/validate_agent_tasks.py agent_tasks.json`

Jules должен сначала прочитать `agent_tasks.json`, реализовать первую задачу со статусом `todo`, поддерживать пополнение пула задач согласно `replenishment_policy` и обновлять статус задачи после завершения PR.

### 🌉 Мост Codex (Codex Bridge)

Проект предоставляет легковесный мост только на стандартной библиотеке (stdlib), который Codex/Jules могут использовать без импорта полного стека Telegram/FastAPI/памяти:

```bash
python -m magda_agent.codex_bridge validate
python -m magda_agent.codex_bridge status
python -m magda_agent.codex_bridge next-task
python -m magda_agent.codex_bridge render-prompt
```

### 🧪 Локальные проверки

```bash
python scripts/validate_agent_tasks.py agent_tasks.json
python -m magda_agent.codex_bridge status
pytest
```
