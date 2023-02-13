import ipdb
import random
import numpy as np
import collections

from affordances.utils import utils
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.init_learners.init_learner import InitiationLearner
from affordances.goal_attainment.attainment_classifier import GoalAttainmentClassifier


class Option:
  def __init__(self,
    option_idx: int,
    uvfa_policy: Rainbow,
    initiation_learner: InitiationLearner,
    parent_initiation_learner: InitiationLearner,
    goal_attainment_classifier: GoalAttainmentClassifier,
    gestation_period: int,
    timeout: int):
      self._timeout = timeout
      self._solver = uvfa_policy
      self._option_idx = option_idx
      self._gestation_period = gestation_period
      self._goal_attainment_classifier = goal_attainment_classifier
      
      self.initiation_learner = initiation_learner
      self.parent_initiation_learner = parent_initiation_learner

      self._is_global_option = option_idx == 0
      
      self.num_goal_hits = 0
      self.num_executions = 0
      self.effect_set = collections.deque(maxlen=20)
      self.success_curve = collections.deque(maxlen=100)

  @property
  def training_phase(self):
    return 'gestation' if self.num_goal_hits < self._gestation_period else 'mature'
  
  def optimistic_is_init_true(self, state, info):
    if self._is_global_option or self.training_phase == 'gestation':
      return True

    decision1 = self.initiation_learner.optimistic_predict([state]) 
    decision2 = self.initiation_learner.pessimistic_predict([state])
    return decision1 or decision2

  def pessimistic_is_init_true(self, state, info):
    return self.initiation_learner.pessimistic_predict([state])

  def is_term_true(self, state, info):
    if self._option_idx <= 1:
      return self.parent_initiation_learner(info)
    return self.parent_initiation_learner.pessimistic_predict([state])

  def option_reward_func(self, state, info, goal, goal_info):
    inputs = (state, goal) if self._goal_attainment_classifier.use_obs else (info, goal_info)
    reached_goal = self._goal_attainment_classifier(*inputs)
    return reached_goal and self.is_term_true(state, info)

  def act(self, state, goal):
    augmented_state = self.get_augmeted_state(state, goal)
    return self._solver.act(augmented_state)

  def sample_goal(self, state):
    """Sample a goal to pursue from the option's termination region."""

    def get_first_state_in_term_classifier(examples):
      """Given a list of (obs, info) tuples, find the 1st inside the term set."""
      observations = np.array([eg[0] for eg in examples])
      infos = [eg[1] for eg in examples]
      predictions = self.parent_initiation_learner.pessimistic_predict(
        observations)
      sampled_idx = predictions.argmax()  # argmax returns the index of 1st max
      return observations[sampled_idx], infos[sampled_idx]

    if self._option_idx > 1:
      num_tries = 0
      while num_tries < 100:
        num_tries += 1
        trjaectories = self.parent_initiation_learner.positive_examples
        trajectory_idx = random.choice(range(len(trjaectories)))
        sampled_trajectory = trjaectories[trajectory_idx]
        state, info = get_first_state_in_term_classifier(sampled_trajectory)
        if self.parent_initiation_learner.pessimistic_predict([state]):
          return state, info
      print(f'{self} did not find a subgoal to sample.')

  def rollout(self, env, state, info, goal, goal_info, init_replay):
    """Execute the option policy from `state` towards `goal`."""
       
    done = False
    reset = False
    reached = False

    n_steps = 0
    total_reward = 0.
    trajectory = []  # (s, a, r, s', info)

    while not done and not reset and not reached and n_steps < self._timeout:
      action = self.act(state, goal)
      
      next_state, reward, done, info = env.step(action)

      n_steps += 1
      trajectory.append((state, action, reward, next_state, info))
  
      if init_replay is not None:
        init_replay.append(state, action, reward, next_state,
          is_state_terminal=info['terminated'], extra_info=info)
  
      reached = self.option_reward_func(next_state, info, goal, goal_info)

      state = next_state
      reset = info['needs_reset']
      total_reward += reward

    self.success_curve.append(reached)
    self.update_option_after_rollout(goal, goal_info, trajectory, reached)

    return state, info, total_reward, done, reset, reached
  
  def update_option_after_rollout(self, goal, goal_info, transitions, success):
    if success:
      self.num_goal_hits += 1

      final_state = transitions[-1][-2]
      final_info = transitions[-1][-1]
      self.effect_set.append((final_state, final_info))

    self.update_option_policy(transitions, goal, goal_info)

    if not self._is_global_option:
      self.update_option_initiation_classifiers(transitions, success)

  def update_option_policy(self, transitions, goal, goal_info):
    self.experience_replay(transitions, goal, goal_info)
    hindsight_goal, hindsight_goal_info = self.pick_hindsight_goal(
      transitions, goal, goal_info)
    self.experience_replay(transitions, hindsight_goal, hindsight_goal_info)

  def experience_replay(self, transitions, goal, goal_info):
    relabeled_trajectory = []
    for state, action, _, next_state, info in transitions:
      sg = self.get_augmeted_state(state, goal)
      nsg = self.get_augmeted_state(next_state, goal)
      reached = self.option_reward_func(next_state, info, goal, goal_info)
      relabeled_trajectory.append((
        sg, action, float(reached), nsg, reached, info['needs_reset']))
      if reached:
        break
    self._solver.experience_replay(relabeled_trajectory)

  def pick_hindsight_goal(self, transitions, goal, goal_info, method='final'):
    if method == 'final':
      goal_transition = transitions[-1]
      goal = goal_transition[-2]
      goal_info = goal_transition[-1]
      return goal, goal_info
    if method == 'future':
      start_idx = len(transitions) // 2
      goal_idx = random.randint(start_idx, len(transitions) - 1)
      goal_transition = transitions[goal_idx]
      goal = goal_transition[-2]
      goal_info = goal_transition[-1]
      return goal, goal_info
    raise NotImplementedError(method)

  def update_option_initiation_classifiers(self, transitions, success):
    self.initiation_learner.add_trajectory(transitions, success)
    self.initiation_learner.update()

  def get_augmeted_state(self, state, goal):
    assert state.shape == (1, 84, 84), state.shape
    return np.concatenate((state, goal), axis=0)

  def __hash__(self) -> int:
    return self._option_idx

  def __str__(self) -> str:
    return f'option-{self._option_idx}'
  
  def __repr__(self) -> str:
    return str(self)
