import asyncio
from magda_agent.isolation.cloud_vm import CloudVMManager

async def execute_in_cloud_vm(code: str) -> str:
    """
    Executes code in a fully isolated Cloud VM.

    Args:
        code (str): The code to execute.

    Returns:
        str: The result of the execution.
    """
    manager = CloudVMManager()

    vm_id = await manager.provision_vm()
    try:
        # Avoid f-string shell injection issues by safely framing it.
        output = await manager.run_command(vm_id, "python3 -c " + repr(code))
        return output
    except Exception as e:
        return f"Cloud VM Error: {e}"
    finally:
        await manager.terminate_vm(vm_id)
