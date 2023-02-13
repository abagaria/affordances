import numpy as np


class InitiationLearner:
  def __init__(self) -> None:
    pass

  def __call__(self, states: np.ndarray) -> np.ndarray:
    """Given a batch of states, return their initiation decisions."""
    raise NotImplementedError()

  def add_transition(self, obs, action, reward, next_obs, info):
    raise NotImplementedError()

  def update(self):
    raise NotImplementedError()

