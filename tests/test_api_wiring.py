from magda_agent import api
from magda_agent.safety.policy import PolicyLayer


def test_api_skill_registry_uses_policy_layer() -> None:
    assert isinstance(api.policy_layer, PolicyLayer)
    assert api.skill_registry.policy_layer is api.policy_layer


def test_api_consciousness_has_wired_subsystems() -> None:
    consciousness = api.consciousness

    assert consciousness.confidence_calibrator is api.confidence_calibrator
    assert consciousness.basal_ganglia is api.basal_ganglia
    assert consciousness.online_learner is api.online_learner
    assert consciousness.style_adapter is api.style_adapter
    assert consciousness.user_model is api.user_model
    assert consciousness.brainstem is api.brainstem
    assert consciousness.skill_versioning is api.skill_versioning
