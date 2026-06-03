# Magda Agent Backlog

## 🔴 Критично
* [x] SECURITY: Добавить whitelist для Telegram-пользователей. (Уже частично реализовано, требуется проверка)
* [x] BUG: `magda_agent/main.py` не обрабатывает асинхронные ошибки.

## 🟡 Важно
* [x] FEATURE: Добавить возможность поиска в интернете.
* [x] ARCHITECTURE: Вынести `Consciousness` в отдельный микросервис.

## ✅ Выполнено
* [x] ARCHITECTURE: Вынести `Consciousness` в отдельный микросервис.
* [x] SECURITY: Изолировать system_execute_code — заменить голый exec() на subprocess с таймаутом 10 сек и без доступа к сети. Код не должен иметь доступ к файловой системе за пределами /tmp/sandbox/ (2026-06-03)
