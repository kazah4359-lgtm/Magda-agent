import json

with open('agent_tasks.json', 'r') as f:
    data = json.load(f)

# Update task
for t in data['tasks']:
    if t['id'] == 'openclaw-context-engine-hooks':
        t['status'] = 'done'

# Add 3 new tasks
new_tasks = [
    {
      "id": "openclaw-context-hooks-v2",
      "status": "todo",
      "area": "architecture",
      "risk": "medium",
      "title": "OpenClaw Context Engine Hooks V2 Integration",
      "description": "Integrate the HookRegistry properly inside ContextEngine so that plugins can register on initialization.",
      "allowed_paths": [
        "magda_agent/memory/context_engine.py",
        "tests/test_context_engine.py",
        "agent_tasks.json"
      ],
      "acceptance": [
        "HookRegistry is used by ContextEngine",
        "Tests verify initialization and hook forwarding"
      ]
    },
    {
      "id": "openclaw-context-hooks-telemetry",
      "status": "todo",
      "area": "architecture",
      "risk": "medium",
      "title": "Context Hooks Telemetry plugin",
      "description": "Inspired by trend: telemetry over hook pipeline. Implement a Telemetry ContextPlugin for tracking hook executions and context sizes.",
      "allowed_paths": [
        "magda_agent/architecture/telemetry_plugin.py",
        "tests/test_telemetry_plugin.py",
        "agent_tasks.json"
      ],
      "acceptance": [
        "TelemetryPlugin registers on before_retrieval and after_retrieval",
        "Metrics are collected properly"
      ]
    }
]

data['tasks'].extend(new_tasks)

with open('agent_tasks.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write('\n')
