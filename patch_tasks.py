import json

with open("agent_tasks.json", "r") as f:
    data = json.load(f)

# Update thought-chain-visualization
for task in data["tasks"]:
    if task["id"] == "thought-chain-visualization":
        task["status"] = "done"

new_tasks = [
    {
      "id": "agent-guard-runtime-governance",
      "status": "todo",
      "area": "safety",
      "risk": "medium",
      "title": "Agent Guard: Runtime Governance Layer",
      "description": "Inspired by the Agent Guard and Prempti trend (June 2026), implement a runtime governance layer between the agent and external actions, effectively wrapping the Policy layer into a formalized execution checkpoint.",
      "allowed_paths": [
        "magda_agent/safety/agent_guard.py",
        "tests/test_agent_guard*.py",
        "agent_tasks.json"
      ],
      "acceptance": [
        "A new AgentGuard class evaluates external tool actions",
        "It acts as a middleware wrapping existing risk/policy checks",
        "Tests verify tool calls are appropriately gated by AgentGuard"
      ]
    },
    {
      "id": "mcp-kernel-sandboxed-execution",
      "status": "todo",
      "area": "security",
      "risk": "high",
      "title": "MCPKernel: Taint Tracking and Sandboxed Execution",
      "description": "Inspired by the MCPKernel trend (June 2026), implement an isolated execution environment for code blocks using strict taint tracking to prevent side-channel leaks during execution.",
      "allowed_paths": [
        "magda_agent/security/mcp_kernel.py",
        "tests/test_mcp_kernel*.py",
        "agent_tasks.json"
      ],
      "acceptance": [
        "MCPKernel runs dynamically generated code securely",
        "Code containing unsafe calls (network, uncontrolled FS) is blocked",
        "Tests mock tainted executions to verify blocking"
      ]
    },
    {
      "id": "assert-policy-driven-evaluation",
      "status": "todo",
      "area": "metacognition",
      "risk": "low",
      "title": "ASSERT: Policy-driven evaluation framework",
      "description": "Inspired by Microsoft's ASSERT framework trend (June 2026), augment the Metacognition Evaluator to automatically score outputs based on explicitly defined policy documents or rules, rather than just raw heuristic scores.",
      "allowed_paths": [
        "magda_agent/metacognition/assert_evaluator.py",
        "tests/test_assert_evaluator*.py",
        "agent_tasks.json"
      ],
      "acceptance": [
        "Evaluator accepts a list of policy constraints",
        "LLM evaluates outputs conditionally based on satisfying these constraints",
        "Tests mock the LLM to verify constraint-driven scoring"
      ]
    }
]

data["tasks"].extend(new_tasks)

with open("agent_tasks.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")
