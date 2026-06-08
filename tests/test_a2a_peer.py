import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.integration.a2a_peer import A2APeerDelegatorV5
from magda_agent.integration.a2a import A2AManager

@pytest.mark.asyncio
async def test_a2a_peer_delegator_success():
    manager = MagicMock(spec=A2AManager)
    manager.delegate_task = AsyncMock(return_value="Delegated to Agent TestAgent: Success")

    delegator = A2APeerDelegatorV5(manager)
    result = await delegator.delegate_to_peer("coding", {"task": "test"})

    assert result == "Delegated to Agent TestAgent: Success"
    manager.delegate_task.assert_called_once_with("coding", {"task": "test"})
