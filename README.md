# Magda Agent

Magda Agent is an experimental cognitive agent built around a Telegram interface, a FastAPI consciousness service, memory, emotion, planning, skills and a Jules-driven self-improvement loop.

## Self-Improvement Loop

- Target architecture: [docs/cognitive_architecture.md](docs/cognitive_architecture.md)
- Machine-readable task queue: [agent_tasks.json](agent_tasks.json)
- Task manifest validator: `python scripts/validate_agent_tasks.py agent_tasks.json`

Jules should read `agent_tasks.json` first, implement the first task with status `todo`, keep the task pool replenished according to `replenishment_policy`, and update the task status after completing a PR.

## Local Checks

```bash
python scripts/validate_agent_tasks.py agent_tasks.json
pytest
```
