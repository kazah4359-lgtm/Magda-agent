import argparse
import asyncio
import os
import sys

def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key] = val
        except FileNotFoundError:
            pass

async def run_chat_once(message: str):
    from magda_agent.api import consciousness
    response = await consciousness.process_input(message)
    print(response)

async def run_chat_loop():
    print("Chat mode. Type 'exit' or 'quit' to stop.")
    from magda_agent.api import consciousness
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input.strip():
                continue
            response = await consciousness.process_input(user_input)
            print(response)
        except (EOFError, KeyboardInterrupt):
            break

def get_status():
    from magda_agent.api import consciousness
    print(consciousness.get_internal_state())

def get_memory_list():
    from magda_agent.api import memory_system
    memories = memory_system.short_term
    if not memories:
        print("Memory is empty.")
    for m in memories:
        print(f"- {m.content}")

def clear_memory():
    from magda_agent.api import memory_system
    memory_system.working_memory.clear()
    print("Memory cleared.")

def main():
    load_env()
    parser = argparse.ArgumentParser(prog="magda", description="Magda Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("--once", type=str, help="Single request")

    memory_parser = subparsers.add_parser("memory", help="Memory commands")
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command")
    memory_subparsers.add_parser("list", help="List memory")
    memory_subparsers.add_parser("clear", help="Clear memory")

    subparsers.add_parser("status", help="Agent status")

    args = parser.parse_args()

    if args.command == "status":
        get_status()
    elif args.command == "memory":
        if args.memory_command == "list":
            get_memory_list()
        elif args.memory_command == "clear":
            clear_memory()
        else:
            memory_parser.print_help()
    elif args.command == "chat":
        if args.once:
            asyncio.run(run_chat_once(args.once))
        else:
            asyncio.run(run_chat_loop())
    else:
        parser.print_help()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
