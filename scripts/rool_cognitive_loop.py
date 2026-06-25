"""
Rool Cognitive Loop for Jules.
This script implements the Observe-Orient-Decide-Act pattern.
"""
import os
import subprocess
import json

def observe():
    print("--- OBSERVING ---")
    # Check tasks
    with open("agent_tasks.json", "r") as f:
        tasks = json.load(f)
    todo = [t for t in tasks["tasks"] if t["status"] == "todo"]
    print(f"Found {len(todo)} todo tasks.")
    return todo[0] if todo else None

def orient(task):
    print(f"--- ORIENTING: {task['title']} ---")
    print(f"Goal: {task['description']}")
    # Check if files exist
    for path in task.get("allowed_paths", []):
        exists = os.path.exists(path)
        print(f"  Path {path}: {'EXISTS' if exists else 'MISSING'}")

def decide(task):
    print("--- DECIDING ---")
    # Simple logic: if files are missing, create them. If tests fail, fix them.
    return "verify_and_report"

def act(action, task):
    print(f"--- ACTING: {action} ---")
    if action == "verify_and_report":
        print("Running tests...")
        result = subprocess.run(["pytest"], capture_output=True, text=True)
        print(result.stdout[-500:]) # Last 500 chars

def run_loop():
    task = observe()
    if task:
        orient(task)
        action = decide(task)
        act(action, task)
    else:
        print("No tasks found.")

if __name__ == "__main__":
    run_loop()
