"""Tests for the Linear Reward Parameterization."""
import numpy as np
import pytest

from irlpreference.reward_parameterizations import MonteCarloLinearReward

@pytest.fixture
def mclr():
    """Data for grid archive tests."""
    return MonteCarloLinearReward(number_dimensions=2, 
                                                number_samples=10_000)


@pytest.mark.parametrize("dimensions", [2,5,10])
@pytest.mark.parametrize("samples", [10, 100, 10_000])
def test_init(dimensions, samples):
    preference_model = MonteCarloLinearReward(number_dimensions=dimensions, 
                                                number_samples=samples)

    assert preference_model.hypothesis_samples.shape[0] == samples
    assert preference_model.hypothesis_samples.shape[1] == dimensions
    assert len(preference_model.hypothesis_samples.shape) == 2
    assert preference_model.hypothesis_log_probabilities.shape[0] == samples
    assert len(preference_model.hypothesis_log_probabilities.shape) == 1

def test_update(mclr):
    #first make sure with no prior the probabilities are all zero
    assert np.all(mclr.hypothesis_log_probabilities == 0)
    
    def p_x(omega):
        '''An arbitrary probability function'''
        return np.max(omega**2) / 2
    mclr.update(p_x)

    #after an update, make sure something has changed
    assert np.any(mclr.hypothesis_log_probabilities != 0)

def test_expected_value(mclr):
    #first make sure with no prior the probabilities are all zero
    assert np.all(mclr.hypothesis_log_probabilities == 0)
    assert mclr.get_expectation().shape[0] == 2 #number of dimensions
    assert np.sum(mclr.get_expectation()**2) < .001 # expectation should be zero

    def p_x(omegas):
        '''probability function that shifts expected omega[0] right'''
        return (omegas[:,0] + 1) / 2
    mclr.update(p_x)

    assert mclr.get_expectation().shape[0] == 2 #number of dimensions
    assert np.sum(mclr.get_expectation()**2) > .001 # expectation should shift