import numpy as np


class GoalAttainmentClassifier:
  def __init__(self, goal_space_type: str, use_obs: bool):
    assert goal_space_type in ('discrete', 'continuous'), goal_space_type
    self._goal_space_type = goal_space_type
    self.use_obs = use_obs

  def __call__(self, state, goal):
    """Determine whether the goal is achieved in the current state."""
    raise NotImplementedError()


class InfoAttainmentClassifier(GoalAttainmentClassifier):
  def __init__(self, goal_space_type: str, key_name='pos'):
    self._key = key_name
    super().__init__(goal_space_type, use_obs=False)


class DiscreteInfoAttainmentClassifier(InfoAttainmentClassifier):
  def __init__(self, key_name='pos'):
    super().__init__(goal_space_type='discrete', key_name=key_name)

  def __call__(self, state, goal):
    assert isinstance(state, dict), type(state)
    assert isinstance(goal, dict), type(goal)
    return state[self._key] == goal[self._key]


class ContinuousInfoAttainmentClassifier(InfoAttainmentClassifier):
  def __init__(self, tolerance, key_name='pos'):
    self._tolerance = tolerance
    super().__init__(goal_space_type='continuous', key_name=key_name)

  def __call__(self, state, goal):
    assert isinstance(state, dict), type(state)
    assert isinstance(goal, dict), type(goal)
    return np.linalg.norm(state[self._key] - goal[self._key]) <= self._tolerance
