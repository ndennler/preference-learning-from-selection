"""Reward Parameterizations describe how the human's reward function is modeled.
They implement a mapping from omega to 
.. autosummary::
    :toctree:
    ribs.archives.GridArchive
    src.reward_parameterizations.MonteCarloLinearReward
"""
from src.reward_parameterizations._linear_reward import MonteCarloLinearReward


__all__ = [
    "MonteCarloLinearReward",
]