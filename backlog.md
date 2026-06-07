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
## 🛠️ Запланированные Skills

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

* [x] MODULE: **Curiosity Drive** — `magda_agent/exploration/curiosity.py`. (2026-06-05)
  Proactively suggests exploration tasks when boredom is high.
* [ ] FEATURE: MCPKernel Taint Tracking
* [ ] FEATURE: ASSERT Policy-Driven Evaluation
