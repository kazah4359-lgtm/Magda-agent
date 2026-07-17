import pytest
import httpx
import respx

from magda_agent.integration.a2a_delegator_v4 import A2ADelegatorV4

@pytest.mark.asyncio
async def test_delegate_task_success():
    delegator = A2ADelegatorV4()
    endpoint = "http://fake-peer/delegate"
    payload = {"task": "do_something"}

    mock_response = {"status": "success", "result": 42}

    with respx.mock:
        respx.post(endpoint).mock(return_value=httpx.Response(200, json=mock_response))

        response = await delegator.delegate_task(endpoint, payload)

        assert response == mock_response

@pytest.mark.asyncio
async def test_delegate_task_failure():
    delegator = A2ADelegatorV4()
    endpoint = "http://fake-peer/delegate"
    payload = {"task": "do_something"}

    with respx.mock:
        respx.post(endpoint).mock(return_value=httpx.Response(500, json={"error": "Internal Server Error"}))

        with pytest.raises(httpx.HTTPStatusError):
            await delegator.delegate_task(endpoint, payload)
