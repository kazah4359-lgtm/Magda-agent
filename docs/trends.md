# AI Agent Trends — Июнь 2026

Этот файл обновляется периодически. Jules обязан прочитать его перед генерацией новых задач.

## Ключевые игроки и архитектуры

### Hermes Agent (Nous Research) — 177K+ stars
- Self-improving agent с built-in learning loop
- Создаёт skills из опыта, улучшает их при использовании
- Persistent memory: чем дольше работает, тем лучше знает пользователя
- Multi-platform: Telegram, Discord, Slack, WhatsApp, Signal, CLI
- Sub-agents для параллельных задач (RPC, свой контекст каждому)
- Cron scheduler: daily reports, nightly backups, scheduled tasks
- Работает на любой модели: OpenRouter, OpenAI, Anthropic, Nous Portal, локальные
- Открытый стандарт skills: agentskills.io

### OpenClaw — 374K+ stars
- Personal AI assistant, local-first архитектура
- Gateway как control plane + channels (30+ платформов)
- Context Engine plugin interface с lifecycle hooks
- Canvas для live визуализации
- OpenClaw-RL: online reinforcement learning из next-state signals (user replies, tool outputs)
- Trains agent simply by talking — без отдельной разметки

### Claude Agent SDK (Anthropic)
- Agent Teams: multi-agent с git worktree isolation
- Трёхагентная архитектура: Planner, Generator, Evaluator
- 1M token context window (Opus 4.6)
- Task system с dependency graphs
- Subagent spawning, context compression

### OpenAI Agents SDK
- Default model: GPT-5.4-mini
- Multi-agent workflows (27K stars)
- MCP server-prefixed tool names
- Runtime function tool concurrency
- Realtime guardrail fallback

### Google Jules
- Gemini 3 Pro model
- GitHub-native: labels в issues для assignment
- Cloud VM для каждой задачи
- AUTO_CREATE_PR mode

## Протоколы и стандарты

### MCP (Model Context Protocol)
- Стандарт де-факто для agent-to-tool коммуникации
- 177K+ tools создано с 11/2024 по 06/2026
- JSON-RPC client-server interface
- Доля "action" tools выросла с 27% до 65%
- Поддержка: OpenAI, Anthropic, Google, Microsoft, AWS

### A2A (Agent-to-Agent Protocol)
- Открытый стандарт от Google → Linux Foundation
- Agent discovery через Agent Cards
- Peer-to-peer task delegation
- Поддержка: LangGraph, CrewAI, Semantic Kernel
- Async-first, enterprise-ready (auth, tracing, monitoring)

### ACS (Agent Control Specification) — Microsoft, июнь 2026
- Стандарт для runtime safety controls
- 5 validation checkpoints в agentic workflows
- "MCP для безопасности" — как MCP стандартизировал tools, ACS стандартизирует guardrails

## Архитектурные паттерны 2026

### Multi-Agent Orchestration
- Planner / Generator / Evaluator (Claude)
- Magentic-One pattern (Microsoft)
- Hierarchical delegation с isolation
- Git worktree per agent

### Memory Architecture
- Virtual context management (MemGPT/Letta pattern → стандарт)
- Разделение: Working / Episodic / Semantic / Procedural
- Online learning из interactions (OpenClaw-RL)
- Context compression + selective retrieval

### Safety & Guardrails
- Agent Guard: runtime governance layer между агентом и внешним миром
- Prempti (Falco): tool-call interception + audit trail
- MCPKernel: taint tracking + sandboxed execution
- ASSERT (Microsoft): policy-driven evaluation framework
- Принцип: каждый tool call проходит через policy layer

### Self-Improvement Loops
- Skill creation from experience (Hermes)
- Longitudinal quality metrics
- Post-merge smoke tests
- Reflection → new task generation
- Online RL from user feedback

### Evaluation (бенчмарки)
- SWE-bench Verified: Claude 80.9%, GPT-5 88% (Aider)
- SWE-bench Pro: Claude 45.9% (реальная сложность)
- WebArena: web navigation tasks
- AgentBench: multi-domain
- Важно: agent-generated code имеет больший churn vs human

## Тренды для генерации задач Magda

При создании новых задач вдохновляйся:
1. **MCP-совместимость** — Magda должна уметь экспортировать skills как MCP tools
2. **A2A поддержка** — возможность делегировать задачи другим агентам
3. **Online learning** — учиться на каждом диалоге (не только reflections)
4. **Context Engine** — plugin-архитектура для управления контекстом
5. **Guardrails** — policy layer для каждого действия с внешним эффектом
6. **Agent Teams** — порождение sub-agents для параллельных задач
7. **Scheduled operations** — cron-подобные задачи без участия пользователя
8. **Cross-platform reach** — один агент, много каналов
9. **Skill marketplace** — совместимость с agentskills.io или аналогом
10. **Quality metrics** — longitudinal tracking, не одноразовый pytest
