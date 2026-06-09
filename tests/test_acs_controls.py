import pytest
from magda_agent.safety.acs_controls import ACSControls

def test_acs_checkpoints():
    """Tests all 5 ACS checkpoints in the ACSControls module."""
    controls = ACSControls()

    # Checkpoint 1
    assert controls.checkpoint_1_input_validation({})[0] == False
    assert controls.checkpoint_1_input_validation({"action": "test"})[0] == True

    # Checkpoint 2
    assert controls.checkpoint_2_intent_authorization({"action": "unauthorized"})[0] == False
    assert controls.checkpoint_2_intent_authorization({"action": "test"})[0] == True

    # Checkpoint 3
    assert controls.checkpoint_3_tool_policy({"tool": "forbidden"})[0] == False
    assert controls.checkpoint_3_tool_policy({"tool": "allowed"})[0] == True

    # Checkpoint 4
    assert controls.checkpoint_4_state_transition({"current_state": "error", "next_state": "executing"})[0] == False
    assert controls.checkpoint_4_state_transition({"current_state": "idle", "next_state": "executing"})[0] == True

    # Checkpoint 5
    assert controls.checkpoint_5_output_sanitization({"output": "my secret key"})[0] == False
    assert controls.checkpoint_5_output_sanitization({"output": "public info"})[0] == True

    # Validate Workflow
    valid_data = {
        "action": "test",
        "tool": "allowed",
        "current_state": "idle",
        "next_state": "executing",
        "output": "public info"
    }
    invalid_data = {
        "action": "test",
        "tool": "forbidden"
    }

    assert controls.validate_workflow(valid_data) == True
    assert controls.validate_workflow(invalid_data) == False
