import json

with open("agent_tasks.json", "r") as f:
    data = json.load(f)

for task in data["tasks"]:
    if task["id"] == "swe-bench-verified-evaluation-suite-new-9012":
        task["status"] = "done"

new_tasks = [
    {
        "id": "agent-guard-runtime-policy-v5-new",
        "status": "todo",
        "area": "safety",
        "risk": "high",
        "title": "Agent Guard Runtime Policy Check",
        "description": "Implement Agent Guard to intercept and evaluate all tool calls based on runtime policy.",
        "allowed_paths": [
            "magda_agent/safety/guard_runtime_policy_v5.py",
            "tests/test_guard_runtime_policy_v5.py",
            "agent_tasks.json"
        ],
        "acceptance": [
            "AgentGuardRuntimePolicyV5 implemented and intercepts tool calls.",
            "Tests verify that violations are blocked."
        ]
    },
    {
        "id": "context-engine-plugin-interface-new",
        "status": "todo",
        "area": "memory",
        "risk": "medium",
        "title": "Context Engine Plugin Interface",
        "description": "Inspired by OpenClaw: Implement a plugin interface for managing memory context dynamically using hooks.",
        "allowed_paths": [
            "magda_agent/memory/context_engine_v5.py",
            "tests/test_context_engine_v5.py",
            "agent_tasks.json"
        ],
        "acceptance": [
            "ContextEngineV5 is implemented with a DynamicHookRegistry.",
            "Tests verify plugins can be registered and executed."
        ]
    },
    {
        "id": "a2a-agent-discovery-service-new",
        "status": "todo",
        "area": "integration",
        "risk": "medium",
        "title": "A2A Agent Discovery Service",
        "description": "Implement an Agent Discovery Service using AgentCardV3 for peer-to-peer A2A delegation via httpx.",
        "allowed_paths": [
            "magda_agent/integration/a2a_discovery_v3_unique.py",
            "tests/test_a2a_discovery_v3_unique.py",
            "agent_tasks.json"
        ],
        "acceptance": [
            "A2ADiscoveryServiceV3Unique can discover peers by matching capabilities.",
            "Tests mock httpx to verify async peer discovery."
        ]
    }
]

data["tasks"].extend(new_tasks)

with open("agent_tasks.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")
