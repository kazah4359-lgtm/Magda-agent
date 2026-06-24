import pytest
from magda_agent.skills.cloud_vm_skill import execute_in_cloud_vm

@pytest.mark.asyncio
async def test_execute_in_cloud_vm():
    code = "print('hello from cloud')"
    output = await execute_in_cloud_vm(code)
    # The output from the mock will contain the executed command representation
    assert "Simulated output for:" in output
    assert "print('hello from cloud')" in output

@pytest.mark.asyncio
async def test_execute_in_cloud_vm_with_quotes():
    code = "print(\"it's a string\")"
    output = await execute_in_cloud_vm(code)
    assert "Simulated output for:" in output
    assert "it\\'s a string" in output or "it's a string" in output
