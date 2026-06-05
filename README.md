<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=250&section=header&text=Все%20по%20полочкам&fontSize=60&fontAlignY=38&desc=Everything%20on%20Shelves&descAlignY=55&descAlign=62" alt="Все по полочкам / Everything on Shelves Header"/>

  <p align="center">
    <i>An experimental cognitive agent / Экспериментальный когнитивный агент</i>
  </p>

  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"></a>
    <a href="https://core.telegram.org/"><img src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram"></a>
    <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"></a>
    <a href="https://docs.pytest.org/"><img src="https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest"></a>
  </p>

  <p>
    <a href="#english"><strong>🇬🇧 English</strong></a> •
    <a href="#russian"><strong>🇷🇺 Русский</strong></a>
  </p>
</div>

---

<h2 id="english">🇬🇧 English</h2>

**Все по полочкам** is an experimental cognitive agent built around a Telegram interface, a FastAPI consciousness service, memory, emotion, planning, skills, and a Jules-driven self-improvement loop. Originally based on the Magda Agent architecture.

### 🔄 Self-Improvement Loop

- 🎯 **Target architecture:** [docs/cognitive_architecture.md](docs/cognitive_architecture.md)
- 📝 **Codex worker plan:** [docs/codex_worker_plan.md](docs/codex_worker_plan.md)
- 🤖 **Machine-readable task queue:** [agent_tasks.json](agent_tasks.json)
- ✅ **Task manifest validator:** `python scripts/validate_agent_tasks.py agent_tasks.json`

> **Note:** Jules should read `agent_tasks.json` first, implement the first task with status `todo`, keep the task pool replenished according to `replenishment_policy`, and update the task status after completing a PR.

### 🌉 Codex Bridge

The project exposes a lightweight stdlib-only bridge that Codex/Jules can use without importing the full Telegram/FastAPI/memory stack:

```bash
python -m magda_agent.codex_bridge validate
python -m magda_agent.codex_bridge status
python -m magda_agent.codex_bridge next-task
python -m magda_agent.codex_bridge render-prompt
```

### 🧪 Local Checks

Ensure code quality and task integrity by running local checks:

```bash
python scripts/validate_agent_tasks.py agent_tasks.json
python -m magda_agent.codex_bridge status
pytest
```

---

<h2 id="russian">🇷🇺 Русский</h2>

**Все по полочкам** — это экспериментальный когнитивный агент, построенный на базе Telegram-интерфейса, сервиса сознания FastAPI, памяти, эмоций, планирования, навыков и цикла самосовершенствования, управляемого агентом Jules. Первоначально основан на архитектуре Magda Agent.

### 🔄 Цикл самосовершенствования

- 🎯 **Целевая архитектура:** [docs/cognitive_architecture.md](docs/cognitive_architecture.md)
- 📝 **План работы Codex:** [docs/codex_worker_plan.md](docs/codex_worker_plan.md)
- 🤖 **Машиночитаемая очередь задач:** [agent_tasks.json](agent_tasks.json)
- ✅ **Валидатор манифеста задач:** `python scripts/validate_agent_tasks.py agent_tasks.json`

> **Примечание:** Jules должен сначала прочитать `agent_tasks.json`, реализовать первую задачу со статусом `todo`, поддерживать пополнение пула задач согласно `replenishment_policy` и обновлять статус задачи после завершения PR.

### 🌉 Мост Codex (Codex Bridge)

Проект предоставляет легковесный мост только на стандартной библиотеке (stdlib), который Codex/Jules могут использовать без импорта полного стека Telegram/FastAPI/памяти:

```bash
python -m magda_agent.codex_bridge validate
python -m magda_agent.codex_bridge status
python -m magda_agent.codex_bridge next-task
python -m magda_agent.codex_bridge render-prompt
```

### 🧪 Локальные проверки

Для обеспечения качества кода и целостности задач выполняйте локальные проверки:

```bash
python scripts/validate_agent_tasks.py agent_tasks.json
python -m magda_agent.codex_bridge status
pytest
```

<br>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" alt="Footer"/>
</div>
