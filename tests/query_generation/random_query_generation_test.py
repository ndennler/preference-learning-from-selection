"""Tests for the Random Query Generator."""
import numpy as np
import pytest

from irlpreference.query_generation._random_query import RandomQueryGenerator

@pytest.fixture
def random_continuous_generator():
    """Data for grid archive tests."""
    return RandomQueryGenerator(bounds=[(-1,1), (-1,1)])

@pytest.fixture
def random_discrete_generator():
    """Instantiation of a random query generator over a discrete set."""
    return RandomQueryGenerator(bounds=[ (-1,1), 
                                         (-1,1) ],
                                discrete_set=np.array([[1,1],
                                                       [1,0],
                                                       [0,1],
                                                       [0,0],
                                                       [-1,0],
                                                       [0,-1],
                                                       [-1, -1],
                                                       [-1, 1]]))

#TODO: test query generator initialization

@pytest.mark.parametrize("num_queries", [2,4,8])
def test_generate_continuous_query(random_continuous_generator, num_queries):
    #first make sure with no prior the probabilities are all zero
    query = random_continuous_generator.get_query(num_queries)
    assert len(query) == num_queries
    for q in query:
        assert np.all(q >= random_continuous_generator.lower_bounds)
        assert np.all(q <= random_continuous_generator.upper_bounds)

@pytest.mark.parametrize("num_queries", [2,4,8])
def test_generate_discrete_query(random_discrete_generator, num_queries):
    #first make sure with no prior the probabilities are all zero
    query = random_discrete_generator.get_query(num_queries)
    assert len(query) == num_queries
    for q in query:
        assert q in random_discrete_generator.space_of_queries
        assert np.all(q >= random_discrete_generator.lower_bounds)
        assert np.all(q <= random_discrete_generator.upper_bounds)
