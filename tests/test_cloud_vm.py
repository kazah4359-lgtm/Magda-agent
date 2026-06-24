import pytest
import asyncio
from magda_agent.isolation.cloud_vm import CloudVMManager

@pytest.fixture
def vm_manager() -> CloudVMManager:
    return CloudVMManager()

@pytest.mark.asyncio
async def test_provision_vm(vm_manager: CloudVMManager) -> None:
    vm_id = await vm_manager.provision_vm()
    assert vm_id is not None
    assert vm_id in vm_manager.active_vms
    assert vm_manager.active_vms[vm_id]["status"] == "running"

@pytest.mark.asyncio
async def test_run_command(vm_manager: CloudVMManager) -> None:
    vm_id = await vm_manager.provision_vm()
    command = "echo hello"
    output = await vm_manager.run_command(vm_id, command)
    assert f"Simulated output for: {command}" in output

@pytest.mark.asyncio
async def test_run_command_invalid_vm(vm_manager: CloudVMManager) -> None:
    with pytest.raises(ValueError, match="is not available or not running"):
        await vm_manager.run_command("nonexistent-vm", "echo hi")

@pytest.mark.asyncio
async def test_terminate_vm(vm_manager: CloudVMManager) -> None:
    vm_id = await vm_manager.provision_vm()
    await vm_manager.terminate_vm(vm_id)
    assert vm_manager.active_vms[vm_id]["status"] == "terminated"

    with pytest.raises(ValueError, match="is not available or not running"):
        await vm_manager.run_command(vm_id, "echo hi")

@pytest.mark.asyncio
async def test_terminate_invalid_vm(vm_manager: CloudVMManager) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        await vm_manager.terminate_vm("nonexistent-vm")

@pytest.mark.asyncio
async def test_check_boundaries(vm_manager: CloudVMManager) -> None:
    vm_id = await vm_manager.provision_vm()
    assert vm_manager.check_boundaries(vm_id) is True

    assert vm_manager.check_boundaries("nonexistent-vm") is False
