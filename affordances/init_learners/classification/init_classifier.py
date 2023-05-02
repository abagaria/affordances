import collections
import numpy as np

from affordances.init_learners.init_learner import InitiationLearner


class InitiationClassifier(InitiationLearner):
  """Base class for the classification approach to init-learning."""

  def __init__(self, max_n_trajectories):
    super().__init__()

    self.positive_examples = collections.deque([], maxlen=max_n_trajectories)
    self.negative_examples = collections.deque([], maxlen=max_n_trajectories)

  def optimistic_predict(self, states: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

  def pessimistic_predict(self, states: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

  def add_trajectory(self, trajectory):
    raise NotImplementedError()
  
  def update(self):
    raise NotImplementedError()
