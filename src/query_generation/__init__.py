"""Query generations create queries to ask a user.
.. autosummary::
    :toctree:
    src.query_generation.RandomQueryGenerator
"""
from src.query_generation._random_query import RandomQueryGenerator
from src.query_generation._information_gain_query import InfoGainQueryGenerator


__all__ = [
    "RandomQueryGenerator",
    "InfoGainQueryGenerator",
]
