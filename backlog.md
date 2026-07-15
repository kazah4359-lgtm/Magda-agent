# Magda Agent Backlog

## Инструкции для AI-агента

> Бери ПЕРВУЮ задачу со статусом `[ ]` из раздела "Архитектура мозга".
> Каждая задача — это полноценный модуль. Реализуй его целиком: код + тесты + интеграция с существующими модулями.
> После реализации отметь задачу как `[x]` и добавь дату.
> Если обнаружишь баги или зоны роста — добавь их в раздел "Обнаружено".

---

## 🧠 Архитектура мозга

* [x] MODULE: **Mirror Neuron System (Система зеркальных нейронов)** — модуль `magda_agent/emotions/mirror_neurons.py`. (2026-06-04)
  Отвечает за базовую эмпатию: анализирует текст пользователя на наличие позитивных/негативных слов и возвращает сдвиг для PAD (Pleasure, Arousal, Dominance).
  Метод `empathize(text)` — возвращает кортеж (p_shift, a_shift, d_shift) в зависимости от тональности.
  Интеграция: Вызывается в Consciousness при получении сообщения.
  Тесты: проверить логику определения тональности и правильность сдвигов.
* [x] MODULE: **Insula (Инсула)** — модуль `magda_agent/emotions/insula.py`. (2026-06-04)
  Интеграция с внутренними состояниями агента для генерации эмоциональных откликов.
  Метод `process_interoception(energy, boredom)` рассчитывает влияние внутреннего состояния на эмоции.
  Интеграция: Вызывается в Consciousness после обновления Hypothalamus, передавая результаты в EmotionalEngine.
  Тесты: mock LLM, проверить логику сдвигов valence, arousal, dominance в зависимости от energy и boredom.

* [x] MODULE: **Prefrontal Cortex (Планировщик)** — модуль `magda_agent/planning/planner.py`. (2026-06-03)
  Разбивает сложные запросы пользователя на последовательность шагов (план).
  Выбирает какие Skills использовать для каждого шага.
  Поддерживает состояние плана между сообщениями (мультишаговые задачи).
  Интеграция: Consciousness вызывает Planner перед генерацией ответа.
  Тесты: mock LLM, проверить что план генерируется и шаги выполняются последовательно.

* [x] MODULE: **Working Memory** — модуль `magda_agent/memory/working.py`.
  Краткосрочная, ограниченная (bounded) контекстная память в RAM без ChromaDB.
* [x] MODULE: **Episodic Memory (Hippocampus)** — модуль `magda_agent/memory/episodic.py`. (2026-06-03)
  Векторное хранилище (ChromaDB) для всех диалогов и событий.
  Метод `store_event(text, metadata)` — сохранить событие с эмбеддингом.
  Метод `recall_events(query, top_k=5)` — найти релевантные события по семантике.
* [x] MODULE: **Semantic Memory** — модуль `magda_agent/memory/semantic.py`.
  Векторное хранилище (ChromaDB) для фактов о пользователе и мире.
  Метод `store_fact(text, metadata)` — сохранить факт.
  Метод `recall_facts(query, top_k=5)` — семантический поиск фактов.

* [x] MODULE: **Self-Evaluation (Самооценка)** — модуль `magda_agent/metacognition/evaluator.py`. (2026-06-03)
  После каждого ответа агент оценивает качество своего ответа по шкале 1-10.
  Criteria: полезность, точность, полнота, эмоциональная адекватность.
  Результаты сохраняются в Memory. Низкие оценки влияют на system prompt следующего запроса.
  Интеграция: вызывается в Consciousness после отправки ответа пользователю.
  Тесты: mock LLM, проверить что оценка парсится и сохраняется.

* [x] MODULE: **Cerebellum (Обучение на привычках)** — модуль `magda_agent/learning/habits.py`. (2026-06-03)
  Анализирует паттерны: какие Skills часто используются, какие ответы получают высокие оценки.
  Формирует "привычки" — предпочтительные стратегии для типичных запросов.
  Метод `suggest_strategy(input_text)` — предложить стратегию на основе прошлого опыта.
  Интеграция: Planner использует suggest_strategy() для выбора подхода.
  Тесты: проверить что после N успешных использований Skill, он предлагается первым.

* [x] MODULE: **Amygdala (Глубокие эмоции и привязанность)** — модуль `magda_agent/emotions/attachment.py`. (2026-06-03)
  Модель привязанности к пользователю: чем больше общения, тем "ближе" агент.
  Уровни: stranger → acquaintance → friend → close_friend.
  Уровень влияет на стиль общения (формальность, юмор, эмпатия).
  Интеграция: EmotionalEngine учитывает attachment level при генерации тона.
  Тесты: проверить прогрессию уровней, проверить влияние на system prompt.

* [x] MODULE: **Thalamus (Сенсорный фильтр)** — модуль `magda_agent/thalamus/router.py`. (2026-06-04)
  Служит шлюзом для всех входящих сообщений до их обработки сознанием.
  Метод `filter_input(text)` — отбрасывает слишком короткие/пустые или шумовые сообщения.
  Интеграция: API вызывает Thalamus перед отправкой сообщения в Consciousness.
  Тесты: проверить, что шумовые сообщения отклоняются, а нормальные проходят.

* [x] MODULE: **Basal Ganglia (Action Selection)** — модуль `magda_agent/action/selector.py`. (2026-06-04)
  Отвечает за выбор конкретного действия из набора возможных на основе приоритетов.
  Метод `select_action(actions)` — выбирает действие с наибольшим приоритетом.
  Интеграция: Consciousness вызывает Basal Ganglia для финального выбора действия.
  Тесты: mock LLM, проверить выбор действия.

* [x] MODULE: **Pineal Gland (Циркадные ритмы и время)** — модуль `magda_agent/rhythms/pineal_gland.py`. (2026-06-04)
  Управляет внутренними "часами" агента, добавляя в контекст время суток (утро, день, вечер, ночь) и влияя на энергию.
  Метод `get_time_context()` возвращает текущее время суток.
  Метод `get_energy_modifier()` возвращает модификатор энергии на основе времени.
  Интеграция: Consciousness вызывает PinealGland перед LLM Reasoning для добавления временного контекста и перед Hypothalamus для модуляции энергии.
  Тесты: проверить логику определения времени суток по mock datetime и правильность модификаторов энергии.

* [x] MODULE: **Hypothalamus (Homeostatic Drives)** — модуль `magda_agent/drives/hypothalamus.py`. (2026-06-04)
  Отвечает за управление базовыми потребностями (энергия, скука).
  Влияет на внутреннее состояние агента и может служить триггером для действий.
  Интеграция: Consciousness вызывает обновление drives при обработке ввода.
  Тесты: проверить логику обновления (уменьшение энергии, изменение скуки) и границы.

* [x] MODULE: **Brainstem (Ствол мозга - Автономные рефлексы)** — модуль `magda_agent/reflexes/brainstem.py`. (2026-06-04)
  Обрабатывает экстренные команды (stop, help, emergency) для быстрого ответа в обход LLM.
  Метод `process_reflex(text)` — возвращает быстрый ответ или None.
  Интеграция: Вызывается в Consciousness сразу после Thalamus.
  Тесты: проверить, что при экстренных командах возвращается быстрый ответ, а обычные запросы идут дальше.

---

## ✅ Выполнено
- [x] hermes-whatsapp-gateway-v2-e8512f68: Hermes WhatsApp Channel Gateway v2
- [x] mcp-tool-concurrency-integration-e80b41b0: MCP Tool Concurrency Integration v3
- [x] claude-evaluator-multi-agent-teams-v2: Claude SDK Teams Orchestration Evaluator (2026-06-XX)
* [x] FEATURE: Export Skills to MCP Tool Server (mcp-tool-export-skills) (2026-06-XX)
* [x] FEATURE: Agent Guard ACS Checkpoints v6 (agent-guard-acs-checkpoints-v6)
* [x] FEATURE: Online RL from User Feedback v6 (online-rl-user-feedback-v6)
* [x] MODULE: **Virtual Context Manager** — magda_agent/memory/virtual_context.py.
* [x] FEATURE: Hermes Skill Creation from Experience (hermes-skill-creation-8563188e)
* [x] FEATURE: MCP Action Tool Exporter v6 (mcp-action-tool-exporter-v6)
* [x] FEATURE: AgentBench Multi-domain Evaluation (agentbench-multi-domain-eval)
* [x] FEATURE: MCPKernel Taint Tracking Sandbox
* [x] FEATURE: OpenClaw Canvas live visualization support (openclaw-live-canvas-visualizer)
- [x] MCP server-prefixed tools support
* [x] FEATURE: Cross-platform reach across channels (cross-platform-reach-channels)
* [x] FEATURE: Cross-platform reach: Telegram channel v2
* [x] FEATURE: Agent Teams sub-agents spawning (agent-teams-sub-agents)

- [x] a2a-async-auth-tracing: A2A Enterprise-Ready Async Auth and Tracing
- [x] a2a-json-rpc-server: A2A JSON-RPC Server Interface
- [x] acs-runtime-safety-controls-v2: ACS runtime safety controls
* [x] FEATURE: Agent-to-Agent (A2A) Protocol integration
* [x] FEATURE: Online RL from User Feedback v2 (online-rl-user-feedback-v2)
* [x] FEATURE: Longitudinal Quality Metrics (longitudinal-quality-metrics-v2)
* [x] FEATURE: Skill creation from experience (hermes-skill-experience-generation)
- [x] agent-guard-runtime-safety-v4: Agent Guard Runtime Safety Controls
- [x] mcp-action-tool-exporter-v4: MCP Action Tool Exporter
* [x] FEATURE: Guardrails Policy Layer (guardrails-policy-layer)

* [x] FEATURE: Online RL Feedback Integration (online-rl-feedback-integration)
* [x] MODULE: **Agent Teams Git Worktree Isolation** — `magda_agent/agents/isolation.py`. (2026-06-07)
* [x] FEATURE: A2A Peer-to-Peer Task Delegation
* [x] MODULE: **Context Engine Plugin Architecture** — модуль `magda_agent/memory/context_engine.py`. (2026-06-07)
  Inspired by trend #4 in docs/trends.md: Implement a plugin architecture for the Context Engine to manage context dynamically using lifecycle hooks.
* [x] FEATURE: Agent Team Delegation Protocol (agent-team-delegation-protocol)
* [x] FEATURE: Online RL from User Feedback (online-rl-feedback-loop)
* [x] FEATURE: MCP Tools Integration Engine (mcp-tools-integration-v2)

- [x] acs-runtime-safety-controls-v4: ACS Runtime Safety Controls v4
* [x] FEATURE: A2A Agent Discovery v4 (a2a-agent-discovery-v4)
* [x] FEATURE: Agent Teams via Git Worktree Isolation (agent-teams-claude-sdk)
- [x] openclaw-rl-interactive-learning-v3: OpenClaw-RL Interactive Learning
* [x] FEATURE: MCP Action Tools Expansion (mcp-action-tools-v3)
* [x] FEATURE: A2A Task Delegation (v3)
* [x] FEATURE: Longitudinal Quality Metrics CLI
* [x] FEATURE: Agent Skills MCP Export Enhancements (agent-skills-mcp-export)
* [x] FEATURE: Multi-Channel Skills Sync (2026-06-XX)
* [x] FEATURE: A2A Agent Cards Discovery (2026-06-05)
* [x] MODULE: **A2A Agent Discovery Service** - `magda_agent/integration/discovery.py`. Discovers peers via Agent Cards.
* [x] MODULE: **ASSERT Evaluator** — модуль magda_agent/metacognition/assert_evaluator.py.
* [x] FEATURE: Multi-channel gateway (Telegram + Discord + API) (2026-06-05)
* [x] MODULE: **User Model** — модуль `magda_agent/user_model/model.py`. (2026-06-05)
  Builds and maintains a persistent user model: preferences, communication style, expertise level, recurring topics.


* [x] SECURITY: Добавить whitelist для Telegram-пользователей (2026-06-03)
* [x] BUG: Обработка асинхронных ошибок в main.py (2026-06-03)
* [x] FEATURE: Поиск в интернете через DuckDuckGo (2026-06-03)
* [x] ARCHITECTURE: Вынести Consciousness в микросервис (2026-06-03)
* [x] FEATURE: Healthcheck endpoint (2026-06-03)
* [x] SECURITY: Изолировать system_execute_code — subprocess с таймаутом (2026-06-03)

- [x] acs-guardrail-runtime-governance-621d3b4e: ACS Guardrail Checkpoint Refactor (2026-06-15)

---

## 🔍 Обнаружено (добавляется агентом автоматически)
* [x] MODULE: **Conflict Detection Memory** — `magda_agent/memory/semantic.py` и `magda_agent/subconsciousness/reflection.py`. Subconsciousness detects when new information contradicts existing semantic memory and flags conflicts for resolution.
* [x] IMPROVEMENT: Добавить механизм выполнения шагов плана и сбора результатов внутри `Consciousness.process_input`. Сейчас планер интегрирован только для генерации и учета в контексте LLM, но сами функции Skills не вызываются автоматически. (2026-06-03)
* [x] BUG/IMPROVEMENT: В `Evaluator.evaluate_response` можно добавить логику retry на случай, если LLM возвращает невалидный JSON, вместо того, чтобы сразу падать с ошибкой парсинга. (2026-06-04)
* [x] IMPROVEMENT: В модуле `Cerebellum` (`HabitTracker`) сейчас используется точное совпадение текста запроса. Можно улучшить, добавив семантический поиск (через эмбеддинги и ChromaDB) для нечеткого совпадения намерений пользователя. (2026-06-04)
* [x] IMPROVEMENT: Различные модули памяти (например, `LongTermMemory`) в данный момент не учитывают `user_id`. Требуется их обновление для поддержки контекста, специфичного для каждого пользователя (мультипользовательский режим). (2026-06-04)
* [x] IMPROVEMENT: В `MemorySystem` `retrieve_relevant` сейчас использует простой поиск подстрок. Можно улучшить, перейдя на векторный поиск (ChromaDB) для более релевантного и семантического извлечения воспоминаний краткосрочной/долгосрочной памяти в оперативной памяти. (2025-02-14)
* [x] MODULE: **Speech Processing (Голосовой интерфейс)** — модуль `magda_agent/speech/processor.py` (2026-06-05)
  Обработка голосовых сообщений: Speech-to-Text (openai/whisper-tiny) и Text-to-Speech (microsoft/speecht5_tts).
  Методы `stt(audio_path)` и `tts(text, output_path)`.
  Интеграция: main.py перехватывает voice, использует processor, обращается к Consciousness API, и возвращает ответ голосом.
* [x] MODULE: **Subconsciousness (Подсознание)** — модуль `magda_agent/subconsciousness/reflection.py`. (2025-02-14)
  Фоновые процессы рефлексии и консолидации памяти.
  Метод `reflect()` — анализ недавних событий и корректировка эмоций, консолидация памяти.
  Интеграция: Вызывается периодически как фоновая задача.
  Тесты: mock LLM, проверить что рефлексия запускается и `memory.consolidate()` вызывается.
* [x] IMPROVEMENT: Implement gracefully shutdown for `MemorySystem`'s ephemeral ChromaDB client to prevent resource leaks. Add `close()` method to `MemorySystem` and call it in `api.py` lifespan shutdown. (2026-06-04)
* [x] IMPROVEMENT: In `Subconsciousness.reflect`, the LLM prompt for reflection could be more structured to consistently receive PAD adjustments that are then parsed and applied, rather than just hardcoding a dominance increase. (2026-06-04)
* [ ] IMPROVEMENT: `MemorySystem` has a `long_term` list which seems redundant if `LongTermMemory` module is also used. Need to unify long-term memory storage.
* [ ] IMPROVEMENT: In `PinealGland._get_current_time`, consider allowing the timezone to be configured via the user profile rather than relying on system local time.

* [ ] BUG: `Brainstem` integration in `Consciousness.process_input` does not halt long-running skills gracefully upon a 'stop' command. Need to implement a cancellation token or mechanism for active background tasks when a stop reflex is triggered. (2026-06-04)
* [x] FEATURE: Claude Task System with Dependency Graphs v3 (claude-dependency-graph-planner-v3)
## 🛠️ Запланированные Skills
* [x] FEATURE: OpenClaw Canvas Live Visualization (2026-06-15)
* [x] FEATURE: OpenClaw RL Next-State Signals
* [x] FEATURE: Claude Subagent Spawning

* [x] SKILL: **Omnichannel Provider (Работа с провайдерами)** — модуль `magda_agent/skills/omnichannel.py`. (2026-06-04)
  Умение работы с разными провайдерами (Telegram, WhatsApp и другие) для взаимодействия с агентом.
* [ ] SKILL: **Time Management (Управление временем)** — модуль `magda_agent/skills/time_management.py`. Навык установки таймеров, напоминаний и управления расписанием пользователя.
* [ ] SKILL: **Weather Forecasting (Погода)** — модуль `magda_agent/skills/weather.py`. Получение прогноза погоды через внешнее API (например, OpenWeatherMap) по названию города или геолокации.
* [ ] SKILL: **News Aggregator (Агрегатор новостей)** — модуль `magda_agent/skills/news.py`. Сбор актуальных новостей по запросу пользователя из различных RSS-потоков или API.
* [ ] SKILL: **Task Manager (Менеджер задач)** — модуль `magda_agent/skills/task_manager.py`. Интеграция с To-Do листами, создание и отслеживание задач (например, Trello или Notion API).
* [ ] SKILL: **Math Solver (Калькулятор и вычисления)** — модуль `magda_agent/skills/math_solver.py`. Решение сложных математических уравнений и конвертация величин.
* [ ] SKILL: **Translation (Переводчик)** — модуль `magda_agent/skills/translator.py`. Перевод текста между языками с использованием внешних API (Google Translate или DeepL).
* [ ] SKILL: **Image Generator (Генерация изображений)** — модуль `magda_agent/skills/image_gen.py`. Создание изображений по текстовому описанию (интеграция с DALL-E или Stable Diffusion).
* [ ] SKILL: **Smart Home Control (Умный дом)** — модуль `magda_agent/skills/smart_home.py`. Интеграция с системами умного дома (Home Assistant, MQTT) для управления устройствами.
* [ ] SKILL: **Crypto Prices (Курсы криптовалют)** — модуль `magda_agent/skills/crypto.py`. Получение актуальных курсов криптовалют с бирж (Binance, CoinMarketCap).
* [ ] MODULE: **Agent Skills Self-Documentation** — `magda_agent/skills/documentation.py`. Generates natural language documentation for all registered skills.
* [ ] MODULE: **A2A Delegation Cost Estimation** — `magda_agent/integration/a2a_cost_estimator.py`. Predicts token usage and latency for P2P delegation.
* [ ] MODULE: **RL Interaction Loop Entropy Tracking** — `magda_agent/learning/entropy_tracker.py`. Tracks behavior entropy to detect loops.
* [ ] MODULE: **Hermes Online Skill Pruning** — `magda_agent/skills/pruning_v1.py`. Automatically prunes under-used skills from the registry.
* [ ] MODULE: **OpenClaw WebSockets Session Heartbeat** — `magda_agent/visualization/canvas_heartbeat_v1.py`. Keeps WebSocket connections active.
* [ ] MODULE: **ACS Policy Performance Logger** — `magda_agent/safety/acs_perf_logger_v1.py`. Tracks overhead of ACS Checkpoints.

* [x] MODULE: **Curiosity Drive** — `magda_agent/exploration/curiosity.py`. (2026-06-05)
  Proactively suggests exploration tasks when boredom is high.
* [x] FEATURE: MCPKernel Taint Tracking Sandbox
* [x] FEATURE: ASSERT Policy-Driven Evaluation

* [x] MODULE: **Hermes Cron Scheduler v3** — модуль `magda_agent/operations/cron_v3.py`. (2026-06-09)
  Отвечает за cron-подобные фоновые задачи без участия пользователя.
  Интеграция: Вызывается в `api.py` при старте.
  Тесты: mock _get_now, проверить логику scheduler._execute_job.
* [x] FEATURE: Online RL from User Feedback v4 (online-rl-from-feedback-v4)
* [x] FEATURE: Quality metrics longitudinal tracking
* [x] FEATURE: A2A delegation v6
* [x] FEATURE: Online learning v6

* [x] MODULE: **Reflection to task generation** (`magda_agent/planning/reflection_tasks.py`). Inspired by trend: Reflection -> new task generation. Agent should generate new tasks based on reflections from previous failures.
* [x] FEATURE: Task system with dependency graphs (claude-dependency-graph-planner-v2)

* [x] MODULE: **Agent Teams Worktree Isolation** — `claude-subagent-worktree`.
  Implemented SubAgent isolation using GitWorktreeManager.
* [x] FEATURE: MCPKernel: Taint Tracking and Sandboxed Execution (mcpkernel-sandboxed-execution)
* [x] FEATURE: Cross-platform reach channels v2
* [x] FEATURE: OpenClaw Local-First Gateway (2026-06-12)
* [x] FEATURE: A2A Agent Discovery via Cards v2 (a2a-agent-discovery-cards-v2)

- [x] a2a-peer-delegation-discovery-v2: A2A Peer Delegation Discovery
- [x] a2a-peer-delegation-discovery-v3: A2A Peer Delegation Discovery v3
* [x] FEATURE: ACS Workflow Guardrails (acs-workflow-guardrails-v1)
- [x] openclaw-rl-interactive-learning-v5: OpenClaw-RL Interactive Learning v5
* [x] FEATURE: MCP Action Tool Exporter v8 (mcp-action-tool-exporter-v8)
* [x] FEATURE: Agent Teams Parallel Subagents v5 (agent-teams-parallel-subagents-v5)

- [x] openclaw-rl-interactive-learning-v9-a80ac612: OpenClaw-RL Interactive Learning v9

* [x] mcp-dynamic-tool-registry-v4: MCP Dynamic Tool Registry
* [x] FEATURE: A2A Task Delegation Protocol (a2a-task-delegation-v2)
* [x] FEATURE: OpenClaw RL Dynamic Behavior Adjustment v2 (openclaw-rl-dynamic-behavior-adjustment-v2)
* [x] FEATURE: ACS Runtime Safety Controls V6 (acs-guard-runtime-safety-controls-v6)
* [x] FEATURE: MCP Dynamic Action Tool v7 (mcp-dynamic-action-tool-v7)
- [x] a2a-enterprise-integration-v8: A2A Enterprise Integration v8
* [x] FEATURE: MCP Kernel Taint Tracking Policy Engine (mcp-kernel-sandbox-taint-policy-v1-uuid)
* [x] FEATURE: OpenClaw-RL Interactive Feedback Formatter (openclaw-rl-interactive-feedback-v1-uuid)
* [x] FEATURE: OpenClaw-RL Interactive Feedback Formatter v2 (openclaw-rl-interactive-feedback-v2-uuid)
* [x] FEATURE: OpenClaw RL Metrics System v2 (openclaw-rl-metrics-v2-unique)
* [x] FEATURE: A2A Peer-to-Peer Delegation v4 (a2a-peer-delegation-v4-5d34cc13)

* [x] FEATURE: Online Learning from Dialog Interactions (online-learning-interaction-loop-june-2026) (2026-06-08)
* [x] FEATURE: A2A Task Delegation and Discovery (a2a-task-delegation-discovery-june-2026) (2026-06-08)
* [x] FEATURE: A2A Protocol Discovery and Delegation (agent-to-agent-delegation) (2026-06-11)
* [x] FEATURE: OpenClaw RL Metrics Tracker v4 (openclaw-rl-metrics-tracker-v4) (2026-06-10)
* [x] FEATURE: OpenClaw RL Metrics and Visualization v3 (openclaw-rl-metrics-v3-unique) (2026-06-14)
* [x] FEATURE: ACS Control Specification Persistence (acs-guardrail-persistence) (2026-06-12)


## 📡 A2A mDNS & Self-Improvement Loops (June 2026 Trends)
* [x] FEATURE: A2A local network mDNS discovery (a2a-mdns-discovery)
* [ ] FEATURE: A2A mDNS Discovery Secure Token Authentication (a2a-mdns-security-token-v1)
* [ ] FEATURE: OpenClaw RL Trajectory Quality Evaluation (openclaw-rl-trajectory-evaluation-v2)
* [ ] FEATURE: Claude Worktree Pruning and Resource Optimizer (claude-worktree-pruning-optimizer-v1)
