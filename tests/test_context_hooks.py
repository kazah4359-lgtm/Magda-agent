import pytest
from magda_agent.memory.context_hooks import HookRegistry, ContextPluginInterface

def test_hook_registration():
    registry = HookRegistry()
    def sample_hook(query: str, user_id: int) -> str:
        return query + " modified"

    registry.register_hook("before_retrieval", sample_hook)
    assert "before_retrieval" in registry._hooks
    assert len(registry._hooks["before_retrieval"]) == 1
    assert registry._hooks["before_retrieval"][0] == sample_hook

def test_hook_unregistration():
    registry = HookRegistry()
    def sample_hook(query: str, user_id: int) -> str:
        return query + " modified"

    registry.register_hook("before_retrieval", sample_hook)
    assert len(registry._hooks["before_retrieval"]) == 1

    registry.unregister_hook("before_retrieval", sample_hook)
    assert len(registry._hooks["before_retrieval"]) == 0

def test_hook_triggering_pipeline():
    registry = HookRegistry()

    def hook1(query: str, user_id: int) -> str:
        return query + " + hook1"

    def hook2(query: str, user_id: int) -> str:
        return query + " + hook2"

    registry.register_hook("before_retrieval", hook1)
    registry.register_hook("before_retrieval", hook2)

    result = registry.trigger_hook("before_retrieval", "base query", user_id=1)
    assert result == "base query + hook1 + hook2"

def test_hook_triggering_no_hooks():
    registry = HookRegistry()
    result = registry.trigger_hook("before_retrieval", "query", user_id=1)
    assert result == "query"

def test_hook_triggering_no_args():
    registry = HookRegistry()

    executed = False
    def no_arg_hook():
        nonlocal executed
        executed = True

    registry.register_hook("on_event", no_arg_hook)
    registry.trigger_hook("on_event")

    assert executed is True

def test_hook_triggering_side_effect():
    registry = HookRegistry()

    context_updates = []
    def update_hook(new_context: dict, user_id: int) -> None:
        context_updates.append((new_context, user_id))

    registry.register_hook("on_context_update", update_hook)
    registry.trigger_hook("on_context_update", {"key": "value"}, user_id=42)

    assert len(context_updates) == 1
    assert context_updates[0] == ({"key": "value"}, 42)

class DummyPlugin:
    def __init__(self):
        self.hook_called = False

    def my_hook(self, *args, **kwargs):
        self.hook_called = True
        if args:
            return args[0]
        return None

    def attach(self, registry: HookRegistry) -> None:
        registry.register_hook("custom_hook", self.my_hook)

    def detach(self, registry: HookRegistry) -> None:
        registry.unregister_hook("custom_hook", self.my_hook)


def test_plugin_attach_detach():
    registry = HookRegistry()
    plugin = DummyPlugin()

    plugin.attach(registry)
    assert "custom_hook" in registry._hooks
    assert len(registry._hooks["custom_hook"]) == 1

    registry.trigger_hook("custom_hook")
    assert plugin.hook_called is True

    plugin.detach(registry)
    assert len(registry._hooks["custom_hook"]) == 0
