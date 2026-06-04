import pytest
from magda_agent.emotions.mirror_neurons import MirrorNeurons

@pytest.fixture
def mirror_neurons():
    return MirrorNeurons()

def test_empathize_positive(mirror_neurons):
    text = "I am so happy and excited today!"
    p, a, d = mirror_neurons.empathize(text)
    assert p > 0.0
    assert a > 0.0
    assert d == 0.0

def test_empathize_negative(mirror_neurons):
    text = "I feel terrible and sad about this awful situation."
    p, a, d = mirror_neurons.empathize(text)
    assert p < 0.0
    assert a > 0.0
    assert d > 0.0

def test_empathize_neutral(mirror_neurons):
    text = "I went to the store to buy some milk."
    p, a, d = mirror_neurons.empathize(text)
    assert p == 0.0
    assert a == 0.0
    assert d == 0.0

def test_empathize_empty_string(mirror_neurons):
    p, a, d = mirror_neurons.empathize("")
    assert p == 0.0
    assert a == 0.0
    assert d == 0.0

def test_empathize_mixed(mirror_neurons):
    text = "I am happy but also very sad and angry."
    p, a, d = mirror_neurons.empathize(text)
    # 1 pos (happy), 2 neg (sad, angry) -> negative should win
    assert p < 0.0
    assert a > 0.0
    assert d > 0.0

def test_empathize_case_insensitive(mirror_neurons):
    text = "HAPPY SAD TERRIBLE"
    p, a, d = mirror_neurons.empathize(text)
    # 1 pos, 2 neg -> negative should win
    assert p < 0.0
    assert a > 0.0
    assert d > 0.0
