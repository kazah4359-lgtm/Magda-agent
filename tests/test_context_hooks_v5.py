import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.architecture.context_hooks_v5 import HookRegistry

def test_hook_registry_v5_initialization() -> None:
    """Test basic initialization."""
    registry = HookRegistry()
    assert registry._hooks == {}

def test_hook_registry_v5_register_hook() -> None:
    """Test that hooks can be registered."""
    registry = HookRegistry()

    def my_hook(x): return x
    registry.register_hook('before_retrieval', my_hook)

    assert 'before_retrieval' in registry._hooks
    assert registry._hooks['before_retrieval'] == [my_hook]

def test_hook_registry_v5_trigger_hook() -> None:
    """Test synchronous hook triggering."""
    registry = HookRegistry()

    hook1 = MagicMock(return_value="modified1")
    hook2 = MagicMock(return_value="modified2")

    registry.register_hook('before_write', hook1)
    registry.register_hook('before_write', hook2)

    result = registry.trigger_hook('before_write', "initial_context")

    hook1.assert_called_once_with("initial_context")
    hook2.assert_called_once_with("modified1")
    assert result == "modified2"

@pytest.mark.asyncio
async def test_hook_registry_v5_trigger_hook_async() -> None:
    """Test asynchronous hook triggering."""
    registry = HookRegistry()

    hook1 = AsyncMock(return_value="async_mod1")
    hook2 = MagicMock(return_value="sync_mod2")  # Mixed sync/async

    registry.register_hook('ingest', hook1)
    registry.register_hook('ingest', hook2)

    result = await registry.trigger_hook_async('ingest', "content")

    hook1.assert_called_once_with("content")
    hook2.assert_called_once_with("async_mod1")
    assert result == "sync_mod2"

@pytest.mark.asyncio
async def test_hook_registry_v5_trigger_broadcast_async() -> None:
    """Test asynchronous broadcast triggering."""
    registry = HookRegistry()

    hook1 = AsyncMock(return_value="result1")
    hook2 = AsyncMock(return_value="result2")

    registry.register_hook('bootstrap', hook1)
    registry.register_hook('bootstrap', hook2)

    result = await registry.trigger_broadcast_async('bootstrap', {"config": "val"})

    hook1.assert_called_once_with({"config": "val"})
    hook2.assert_called_once_with({"config": "val"})
    assert result == "result2"
