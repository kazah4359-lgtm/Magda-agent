import pytest
from magda_agent.thalamus.router import Thalamus

def test_thalamus_filter_valid_input():
    thalamus = Thalamus()
    assert thalamus.filter_input("Hello, Magda!") is True
    assert thalamus.filter_input("What is the weather?") is True
    assert thalamus.filter_input("?") is True
    assert thalamus.filter_input("A") is True

def test_thalamus_filter_invalid_input():
    thalamus = Thalamus()
    assert thalamus.filter_input("") is False
    assert thalamus.filter_input("   ") is False
    assert thalamus.filter_input("\n") is False
    assert thalamus.filter_input(None) is False
    assert thalamus.filter_input("-") is False
    assert thalamus.filter_input("*") is False
