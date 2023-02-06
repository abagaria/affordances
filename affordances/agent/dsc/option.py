import numpy as np
import collections
from affordances.agent.rainbow.rainbow import Rainbow


class Option:
  def __init__(self, env, option_idx, uvfa_policy: Rainbow, timeout: int):
      self._env = env
      self._timeout = timeout
      self._solver = uvfa_policy
      self._option_idx = option_idx
      self._initiation_classifier = None

      self._is_global_option = option_idx == 0
      
      self.num_goal_hits = 0
      self.num_executions = 0
      self.effect_set = collections.deque(maxlen=20)
      self.success_curve = collections.deque(maxlen=100)
  
  def optimistic_is_init_true(self, state):
    pass

  def pessimistic_is_init_true(self, state):
    pass

  def is_term_true(self, state):
    pass

  def act(self, state, goal):
    augmented_state = self.get_augmeted_state(state, goal)
    return self._solver.act(augmented_state)

  def rollout(self, state, info, goal):
       
    done = False
    reset = False
    reached = False

    n_steps = 0
    trajectory = []  # (s, a, r, s', g, info)

    while not done and not reset and not reached and n_steps < self._timeout:
      action = self.act(state, goal)
      next_state, reward, done, info = self._env.step(action)

      trajectory.append((state, action, reward, next_state, goal, info))
      state = next_state


  def get_augmeted_state(self, state, goal):
    return np.concatenate((state, goal), axis=-1)  # TODO: HWC or CHW?
