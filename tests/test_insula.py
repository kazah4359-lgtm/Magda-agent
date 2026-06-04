import pytest
from magda_agent.emotions.insula import Insula

def test_process_interoception_low_energy():
    insula = Insula()
    # Test low energy
    v_shift, a_shift, d_shift = insula.process_interoception(energy=0.2, boredom=0.5)
    assert v_shift == -0.05
    assert a_shift == -0.05
    assert d_shift == -0.05

def test_process_interoception_high_energy():
    insula = Insula()
    # Test high energy
    v_shift, a_shift, d_shift = insula.process_interoception(energy=0.9, boredom=0.5)
    assert v_shift == 0.0
    assert a_shift == 0.02
    assert d_shift == 0.02

def test_process_interoception_high_boredom():
    insula = Insula()
    # Test high boredom (normal energy)
    v_shift, a_shift, d_shift = insula.process_interoception(energy=0.5, boredom=0.8)
    assert v_shift == -0.05
    assert a_shift == -0.02
    assert d_shift == -0.02

def test_process_interoception_normal_state():
    insula = Insula()
    # Normal energy and boredom should result in zero shifts
    v_shift, a_shift, d_shift = insula.process_interoception(energy=0.5, boredom=0.5)
    assert v_shift == 0.0
    assert a_shift == 0.0
    assert d_shift == 0.0

def test_process_interoception_low_energy_and_high_boredom():
    insula = Insula()
    # Test combination of low energy and high boredom
    v_shift, a_shift, d_shift = insula.process_interoception(energy=0.2, boredom=0.8)
    # Valence: -0.05 (energy) - 0.05 (boredom) = -0.1
    # Arousal: -0.05 (energy) - 0.02 (boredom) = -0.07
    # Dominance: -0.05 (energy) - 0.02 (boredom) = -0.07
    assert v_shift == pytest.approx(-0.1)
    assert a_shift == pytest.approx(-0.07)
    assert d_shift == pytest.approx(-0.07)
