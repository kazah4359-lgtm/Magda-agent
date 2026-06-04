import pytest
from magda_agent.drives.hypothalamus import Hypothalamus

def test_hypothalamus_initialization():
    h = Hypothalamus(initial_energy=0.8, initial_boredom=0.2)
    state = h.get_state()
    assert state["energy"] == 0.8
    assert state["boredom"] == 0.2

    # Test clamping on init
    h2 = Hypothalamus(initial_energy=1.5, initial_boredom=-0.5)
    state2 = h2.get_state()
    assert state2["energy"] == 1.0
    assert state2["boredom"] == 0.0

def test_hypothalamus_update_high_activity():
    h = Hypothalamus(initial_energy=1.0, initial_boredom=1.0)

    # High activity
    h.update(1.0)

    state = h.get_state()
    # energy: 1.0 - (1.0 * 0.05) = 0.95
    assert state["energy"] == pytest.approx(0.95)
    # boredom: 1.0 - (1.0 * 0.1) = 0.90
    assert state["boredom"] == pytest.approx(0.90)

def test_hypothalamus_update_low_activity():
    h = Hypothalamus(initial_energy=0.5, initial_boredom=0.5)

    # Low activity (rest mechanics should kick in)
    h.update(0.0)

    state = h.get_state()
    # energy: 0.5 - 0.0 + 0.02 = 0.52
    assert state["energy"] == pytest.approx(0.52)
    # boredom: 0.5 - 0.0 + 0.05 = 0.55
    assert state["boredom"] == pytest.approx(0.55)

def test_hypothalamus_clamping_during_update():
    h = Hypothalamus(initial_energy=0.01, initial_boredom=0.01)

    # Activity that would push below 0
    h.update(1.0)

    state = h.get_state()
    assert state["energy"] == 0.0
    assert state["boredom"] == 0.0

    # Rest that would push above 1
    h2 = Hypothalamus(initial_energy=0.99, initial_boredom=0.99)
    h2.update(0.0)

    state2 = h2.get_state()
    assert state2["energy"] == 1.0
    assert state2["boredom"] == 1.0

def test_hypothalamus_summary():
    h = Hypothalamus(initial_energy=0.75, initial_boredom=0.25)
    summary = h.get_drives_summary()
    assert summary == "Drives - Energy: 0.75, Boredom: 0.25"
