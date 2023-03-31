from __future__ import annotations

import ipdb
import random
import numpy as np
import collections

from affordances.utils import utils
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.init_learners.gvf.init_gvf import GoalConditionedInitiationGVF
from affordances.goal_attainment.attainment_classifier import GoalAttainmentClassifier
from affordances.init_learners.classification.binary_init_classifier import ConvInitiationClassifier

import affordances.agent.dsc.goal_sampler as goal_sampler


class Option:
  def __init__(self,
    option_idx: int,
    uvfa_policy: Rainbow,
    initiation_gvf: GoalConditionedInitiationGVF,
    termination_classifier: function,
    parent_initiation_classifier: ConvInitiationClassifier,
    goal_attainment_classifier: GoalAttainmentClassifier,
    gestation_period: int,
    timeout: int,
    start_state_classifier,
    exploration_bonus_scale: float,
    use_her_for_policy_evaluation: bool,
    use_weighted_classifiers: bool):
      self._timeout = timeout
      self._solver = uvfa_policy
      self._option_idx = option_idx
      self._gestation_period = gestation_period
      self._goal_attainment_classifier = goal_attainment_classifier
      self._exploration_bonus_scale = exploration_bonus_scale
      self._use_her_for_policy_evaluation = use_her_for_policy_evaluation
      self._use_weighted_classifiers = use_weighted_classifiers
      
      self.initiation_gvf = initiation_gvf
      self.termination_classifier = termination_classifier
      self.parent_initiation_classifier = parent_initiation_classifier
      
      # function that maps current info to {0, 1}
      self._start_state_classifier = start_state_classifier

      self.is_last_option = False
      self.expansion_classifier = None

      self._is_global_option = option_idx == 0
      
      self.num_goal_hits = 0
      self.num_executions = 0
      self.effect_set = collections.deque(maxlen=20)

      # Information for logging and debugging
      self.debug_log = collections.defaultdict(list)

      self.initiation_classifier = ConvInitiationClassifier(
        device=uvfa_policy.device,
        optimistic_threshold=0.5,
        pessimistic_threshold=0.75,
        n_input_channels=1,
      )

      print(f'Created {self} with bonus_scale={exploration_bonus_scale}',
            f'using_weighted_binary_classifier={use_weighted_classifiers}')

  @property
  def training_phase(self):
    return 'gestation' if self.num_goal_hits < self._gestation_period else 'mature'
  
  def optimistic_is_init_true(self, state, info):
    if self._is_global_option or self.training_phase == 'gestation':
      return True

    if self.is_last_option and self._start_state_classifier(info):
      return True
    
    return self.initiation_classifier.optimistic_predict([state]) or \
           self.initiation_classifier.pessimistic_predict([state])

  def pessimistic_is_init_true(self, state, info):
    return self.initiation_classifier.pessimistic_predict([state])

  def is_term_true(self, state, info):
    if self._option_idx <= 1:
      return self.termination_classifier(info)
    
    # TODO(ab): maybe task goal should be in every option's termination set
    return info['terminated'] or self.termination_classifier(state, info)

  def at_local_goal(self, state, info, goal, goal_info):
    inputs = (state, goal) if self._goal_attainment_classifier.use_obs else (info, goal_info)
    reached_goal = self._goal_attainment_classifier(*inputs)
    return reached_goal

  def act(self, state, goal):
    augmented_state = self.get_augmeted_state(state, goal)
    return self._solver.act(augmented_state)

  def sample_goal(self, state, info):
    """Sample a goal to pursue from the option's termination region."""

    if self._option_idx > 1:
      sample = goal_sampler.reachability_sample(
        state, info,
        self.parent_initiation_classifier.positive_examples,
        self.parent_initiation_classifier.pessimistic_predict,
        self.initiation_gvf.get_values,
        self._goal_attainment_classifier
      )

      if sample is not None:
        return sample
      
      # If we can't find something in the parent's pessimistic, use sth random
      positive_examples = utils.flatten(
        self.parent_initiation_classifier.positive_examples
      )

      if len(positive_examples) > 0:
        print(f'[{self}] Sampling from parent positives')
        return random.choice(positive_examples)
    
    # If we are a root option or if we didn't find any other goal
    if len(self.effect_set) > 0:
      return random.choice(self.effect_set)

  def rollout(self, env, state, info, goal, goal_info):
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
      reached = self.at_local_goal(next_state, info, goal, goal_info)

      state = next_state
      reset = info['needs_reset']
      total_reward += reward

    self.update_option_after_rollout(goal, goal_info, trajectory, reached)
    self.log_progress(info, reached, goal_info)

    return state, info, total_reward, done, reset, reached, n_steps
  
  def update_option_after_rollout(self, goal, goal_info, transitions, success):
    if success:
      self.num_goal_hits += 1

      final_state = transitions[-1][-2]
      final_info = transitions[-1][-1]

      # TODO(ab): Hack - removing keys to not distract goal option's rf
      if self._option_idx <= 1:
        assert isinstance(final_info, dict)
        final_info.pop('has_key')
        final_info.pop('door_open')

      self.effect_set.append((final_state, final_info))

    self.update_option_params(transitions, goal, goal_info)

    if not self._is_global_option:
      self.update_option_initiation_classifiers(transitions, success, goal)

  def update_option_params(self, transitions, goal, goal_info):
    """Update the parameters of the UVFA and the initiation GVF."""

    self._solver.experience_replay(
      self.relabel_trajectory(transitions, goal, goal_info, add_bonus=True))
    self.initiation_gvf.add_trajectory_to_replay(
      self.relabel_trajectory(transitions, goal, goal_info, add_bonus=False))

    hindsight_goal, hindsight_goal_info = self.pick_hindsight_goal(transitions)

    # If the first state in the trajectory doesn't achieve the hindsight goal
    if not self._goal_attainment_classifier(transitions[0][-1], hindsight_goal_info):
      self._solver.experience_replay(
        self.relabel_trajectory(
        transitions, hindsight_goal, hindsight_goal_info, add_bonus=True))
      
      if self._use_her_for_policy_evaluation:
        self.initiation_gvf.add_trajectory_to_replay(
          self.relabel_trajectory(
          transitions, hindsight_goal, hindsight_goal_info, add_bonus=False))

  def relabel_trajectory(self, transitions, goal, goal_info, add_bonus=True):
    relabeled_trajectory = []
    bonus_scale = self._exploration_bonus_scale if add_bonus else 0
    for state, action, _, next_state, info in transitions:
      sg = self.get_augmeted_state(state, goal)
      nsg = self.get_augmeted_state(next_state, goal)
      reached = self._goal_attainment_classifier(info, goal_info)
      reward = float(reached) + (bonus_scale * info['bonus'])
      relabeled_trajectory.append((sg, action, reward, nsg, reached, info))
      if reached:
        break
    return relabeled_trajectory

  def pick_hindsight_goal(self, transitions, method='final'):
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

  def update_option_initiation_classifiers(self, transitions, success, goal):
    self.initiation_classifier.add_trajectory(transitions, success)
    
    if self._use_weighted_classifiers:
      self.initiation_classifier.update(self.initiation_gvf, goal)
    else:
      self.initiation_classifier.update()

  def should_expand_initiation(self, s0: np.ndarray, info0: dict):
    """Check if the option can initiate at the start-state."""
    if self.training_phase != 'gestation':
      return self.pessimistic_is_init_true(s0, info0)
    return False
  
  def get_augmeted_state(self, state, goal):
    assert state.shape == (1, 84, 84), state.shape
    return np.concatenate((state, goal), axis=0)
  
  def log_progress(self, info, success, goal_info):
    self.debug_log['timesteps'].append(info['timestep'])
    self.debug_log['successes'].append(success)
    self.debug_log['goals'].append(goal_info)
    self.debug_log['phases'].append(self.training_phase)

  def __hash__(self) -> int:
    return self._option_idx

  def __str__(self) -> str:
    return f'option-{self._option_idx}'
  
  def __repr__(self) -> str:
    return str(self)
