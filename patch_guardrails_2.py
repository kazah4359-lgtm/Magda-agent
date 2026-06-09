import re

with open("magda_agent/safety/guardrails.py", "r") as f:
    content = f.read()

# Update execute_with_guardrails to handle exceptions during execution
original_exec = """        if not allow:
            if asyncio.iscoroutinefunction(tool_func):
                async def async_fallback() -> Any:
                    return handle_fallback()
                return async_fallback()
            return handle_fallback()

        return tool_func(**kwargs)"""

new_exec = """        if not allow:
            if asyncio.iscoroutinefunction(tool_func):
                async def async_fallback() -> Any:
                    return handle_fallback()
                return async_fallback()
            return handle_fallback()

        if asyncio.iscoroutinefunction(tool_func):
            async def async_exec() -> Any:
                try:
                    return await tool_func(**kwargs)
                except Exception as e:
                    logging.error(f"Error executing tool '{tool_name}': {e}")
                    # Return sensible fallback on exception
                    return f"Action '{tool_name}' failed during execution: {str(e)}"
            return async_exec()
        else:
            try:
                return tool_func(**kwargs)
            except Exception as e:
                logging.error(f"Error executing tool '{tool_name}': {e}")
                # Return sensible fallback on exception
                return f"Action '{tool_name}' failed during execution: {str(e)}\""""

content = content.replace(original_exec, new_exec)

with open("magda_agent/safety/guardrails.py", "w") as f:
    f.write(content)
