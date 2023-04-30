"""Tests for the Random Query Generator."""
import numpy as np
import pytest

from irlpreference.input_models._luce_shepard_choice import LuceShepardChoice

@pytest.fixture
def luce_shepard_choice():
    """Data for grid archive tests."""
    return LuceShepardChoice()

def test_init():
    choice_model = LuceShepardChoice()
    assert choice_model.query == None
    assert 'tell_input' in dir(choice_model)

def test_getting_choice_distribution(luce_shepard_choice):
    query = np.array([
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    omegas = np.array([
        [1,0,0,0],
        [0,0,0,1]
    ])
    p = luce_shepard_choice.get_choice_probabilities(query, omegas)
    print(p)
    assert p[0,0] == 1 / (np.e + 1)
    assert p[0,1] == np.e / (np.e + 1)
    assert p[1,0] == np.e / (np.e + 1)
    assert p[1,1] == 1 / (np.e + 1)

def test_getting_input_probability(luce_shepard_choice):
    query = np.array([
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    omegas = np.array([
        [1,0,0,0],
        [0,0,0,1],
        [0,0,0,0]
    ])
    luce_shepard_choice.tell_input(0, query)
    p = luce_shepard_choice.get_probability_of_input(omegas)
    
    assert p[0] == 1 / (np.e + 1)
    assert p[1] == np.e / (np.e + 1)
    assert p[2] == 0.5
