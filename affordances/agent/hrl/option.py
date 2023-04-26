from __future__ import annotations

import ipdb
import random
import numpy as np
import collections

from pfrl.wrappers import atari_wrappers

from affordances.utils import utils
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.init_learners.gvf.init_gvf import GoalConditionedInitiationGVF
from affordances.goal_attainment.attainment_classifier import GoalAttainmentClassifier
from affordances.init_learners.classification.binary_init_classifier import ConvInitiationClassifier


class Option:
  def __init__(self,
    option_idx: int,
    uvfa_policy: Rainbow,
    initiation_gvf: GoalConditionedInitiationGVF,
    goal_attainment_classifier: GoalAttainmentClassifier,
    gestation_period: int,
    timeout: int,
    exploration_bonus_scale: float,
    use_her_for_policy_evaluation: bool,
    use_weighted_classifiers: bool,
    subgoal_obs: np.ndarray,
    subgoal_info: dict,
    only_reweigh_negative_examples: bool,
    use_gvf_as_initiation_classifier: bool):
      self._timeout = timeout
      self._solver = uvfa_policy
      self._option_idx = option_idx
      self._gestation_period = gestation_period
      self._goal_attainment_classifier = goal_attainment_classifier
      self._exploration_bonus_scale = exploration_bonus_scale
      self._use_her_for_policy_evaluation = use_her_for_policy_evaluation
      self._use_weighted_classifiers = use_weighted_classifiers
      self._use_gvf_as_initiation_classifier = use_gvf_as_initiation_classifier

      if use_gvf_as_initiation_classifier:
        assert not use_weighted_classifiers
        assert not only_reweigh_negative_examples
      
      self.initiation_gvf = initiation_gvf

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
        image_dim=subgoal_obs._frames[0].squeeze().shape[0],  # TODO
        only_reweigh_negative_examples=only_reweigh_negative_examples
      ) if not use_gvf_as_initiation_classifier else None

      self.subgoal_obs = subgoal_obs
      self.subgoal_info = subgoal_info
      self.termination_classifier = lambda info: self._goal_attainment_classifier(info, self.subgoal_info)

      print(f'Created {self} with bonus_scale={exploration_bonus_scale}',
            f'using_weighted_binary_classifier={use_weighted_classifiers}',
            f'with subgoal={subgoal_info}')

  @property
  def training_phase(self):
    return 'gestation' if self.num_goal_hits < self._gestation_period else 'mature'
  
  def optimistic_is_init_true(self, state, info):
    if self._is_global_option or self.training_phase == 'gestation':
      return True
    
    if self._use_gvf_as_initiation_classifier:
      value = self.initiation_gvf.get_values(
        states=np.asarray(state)[np.newaxis, ...],  # (1, 4, 84, 84)
        goals=np.asarray(self.subgoal_obs)[-1:]  # (1, 84, 84)
      )
      return value.item() > 0.5
    
    return self.initiation_classifier.optimistic_predict([state]) or \
           self.initiation_classifier.pessimistic_predict([state])

  def pessimistic_is_init_true(self, state, info):
    # return self.initiation_classifier.pessimistic_predict([state])
    raise NotImplementedError()

  def is_term_true(self, state, info):
    return self.termination_classifier(info)  # or info['terminated']

  def act(self, state, goal):
    augmented_state = self.get_augmented_state(state, goal)
    return self._solver.act(augmented_state)

  def rollout(self, env, state, info):
    """Execute the option policy from `state` towards `goal`."""
       
    done = False
    reset = False
    reached = False

    n_steps = 0
    total_reward = 0.
    trajectory = []  # (s, a, r, s', info)

    while not done and not reset and not reached and n_steps < self._timeout:
      action = self.act(state, self.subgoal_obs)
      
      next_state, reward, done, info = env.step(action)

      n_steps += 1
      trajectory.append((state, action, reward, next_state, info))
      reached = self.is_term_true(next_state, info)

      state = next_state
      reset = info['needs_reset']
      total_reward += reward

    self.log_progress(info, reached, self.subgoal_info)

    return state, info, total_reward, reached, n_steps, trajectory
  
  def update_option_after_rollout(self, transitions, success):
    if success:
      self.num_goal_hits += 1

      final_state = transitions[-1][-2]
      final_info = transitions[-1][-1]

      self.effect_set.append((final_state, final_info))

    self.update_option_params(transitions, self.subgoal_obs, self.subgoal_info)

    if not self._is_global_option and not self._use_gvf_as_initiation_classifier:
      self.initiation_classifier.add_trajectory(transitions, success)

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
      sg = self.get_augmented_state(state, goal)
      nsg = self.get_augmented_state(next_state, goal)
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

  def update_option_initiation_classifiers(self):
    """Perform SGD updates to the option's initiation classifier."""
    
    if self._use_weighted_classifiers:
      self.initiation_classifier.update(self.initiation_gvf, self.subgoal_obs)
    else:
      self.initiation_classifier.update()
  
  def get_augmented_state(self, state, goal):
    assert isinstance(goal, atari_wrappers.LazyFrames), type(goal)
    assert isinstance(state, atari_wrappers.LazyFrames), type(state)
    features = list(state._frames) + [goal._frames[-1]]
    return atari_wrappers.LazyFrames(features, stack_axis=0)
  
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
