"""Create options for visual gridworld."""

from __future__ import annotations

import ipdb

from affordances.agent.hrl.option import Option
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.init_learners.gvf.init_gvf import GoalConditionedInitiationGVF
from affordances.goal_attainment.attainment_classifier import DiscreteInfoAttainmentClassifier
from affordances.domains.minigrid import load_doorkey_subgoal_observations


class AgentOverOptions:
  def __init__(
    self,
    env,
    gestation_period,
    timeout,
    image_dim: int,
    rams: list,
    use_weighted_classifiers: bool,
    only_reweigh_negative_examples: bool,
    use_gvf_as_initiation_classifier: bool,
    uncertainty_type: str,
    gpu: int = 0,
    n_input_channels: int = 4,
    n_goal_channels: int = 1,
    env_steps: int = int(500_000),
    epsilon_decay_steps: int = 25_000,
    final_epsilon: float | None = None,
    optimistic_threshold: float = 0.5,
    n_classifier_training_trajectories: int = 10,
    n_classifier_training_epochs: int = 1
  ):
    self._env = env
    self._timeout = timeout
    self._gestation_period = gestation_period
    self._gpu = gpu
    self._n_input_channels = n_input_channels
    self._n_goal_channels = n_goal_channels
    self._rams = rams
    self._use_weighted_classifiers = use_weighted_classifiers
    self._only_reweigh_negative_examples = only_reweigh_negative_examples
    self._use_gvf_as_initiation_classifier = use_gvf_as_initiation_classifier
    self._optimistic_threshold = optimistic_threshold
    self._uncertainty_type = uncertainty_type
    self._n_classifier_training_trajectories = n_classifier_training_trajectories
    self._n_classifier_training_epochs = n_classifier_training_epochs

    self.image_dim = image_dim
    
    self.uvfa_policy = self.create_uvfa_policy(
      env_steps, epsilon_decay_steps, final_epsilon)
    
    self.initiation_gvf = self.create_initiation_learner()
    self.subgoals = self.door_key_get_goal_classifiers()
    self.options = self.create_options()  # TODO: create global option

  def update_initiation_gvf(self, episode_duration):
    """SGD steps for the initiation GVF."""
    self.initiation_gvf.update(
      n_updates=(episode_duration // self.uvfa_policy.update_interval)
    )

  def create_uvfa_policy(self, env_steps, epsilon_decay_steps, final_eps):
    kwargs = dict(
      n_atoms=51, v_max=10., v_min=-10.,
      noisy_net_sigma=0.5,
      lr=1e-4,
      n_steps=3,
      betasteps=env_steps // 4,
      replay_start_size=1024, 
      replay_buffer_size=int(3e5),
      gpu=self._gpu,
      n_obs_channels=self._n_input_channels + self._n_goal_channels,
      use_custom_batch_states=False,
      final_epsilon=final_eps,
      epsilon_decay_steps=epsilon_decay_steps,
      image_dim=self.image_dim
    )
    return Rainbow(self._env.action_space.n, **kwargs)
  
  def create_initiation_learner(self):
    """Common goal-conditioned initiation learner shared by all options."""
    return GoalConditionedInitiationGVF(
      target_policy=self.uvfa_policy.agent.batch_act,
      n_actions=self._env.action_space.n,
      n_input_channels=self._n_input_channels + self._n_goal_channels,
      optimistic_threshold=0.5,
      pessimistic_threshold=0.75,  # don't need this
      image_dim=self.image_dim,
      uncertainty_type=self._uncertainty_type
    )
  
  def door_key_get_goal_classifiers(self):
    """For each subgoal, create the subgoal_info, subgoal_obs and classifier."""

    goal_attainment_classifiers = []
    pos2obs = load_doorkey_subgoal_observations()
    for state_dict in self._rams:
      pos = state_dict['player_pos']
      clf = DiscreteInfoAttainmentClassifier('player_pos')
      obs = pos2obs[pos]
      goal_attainment_classifiers.append((state_dict, obs, clf))
    return goal_attainment_classifiers

  def create_options(self) -> list:
    options = []
    for i, (subgoal_info, subgoal_obs, goal_clf) in enumerate(self.subgoals):
      option = Option(i + 1,  # Assuming that none of these are global-option
                      self.uvfa_policy,
                      self.initiation_gvf,
                      goal_clf,
                      self._gestation_period,
                      self._timeout,
                      exploration_bonus_scale=0,
                      use_her_for_policy_evaluation=True,
                      use_weighted_classifiers=self._use_weighted_classifiers,
                      subgoal_obs=subgoal_obs,
                      subgoal_info=subgoal_info,
                      only_reweigh_negative_examples=self._only_reweigh_negative_examples,
                      use_gvf_as_initiation_classifier=self._use_gvf_as_initiation_classifier,
                      optimistic_threshold=self._optimistic_threshold,
                      n_classifier_training_trajectories=self._n_classifier_training_trajectories,
                      n_classifier_training_epochs=self._n_classifier_training_epochs)
      options.append(option)
    return options
